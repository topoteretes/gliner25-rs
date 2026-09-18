// COGNEE-EVAL: runner producing the candidate-evaluation contract JSON.
//
// Usage:
//   ORT_DYLIB_PATH=/path/libonnxruntime.so \
//   cargo run --release --example cognee_parity -- \
//       --models <export dir> --scenarios <scenarios.json> --out <out.json>
//
// The prompt layout it builds is the one gliner2 2.0.0 builds for
// `create_schema().entities({name: desc}).relations({name: desc})`:
//
//   group 0 : ( [P] "entities [DESCRIPTION] person: … [DESCRIPTION] …" (
//                   [E] person [E] organization … ) )
//   group k : ( [P] "works_for: Person is employed by …" ( [R] head [R] tail ) )
//
// i.e. every description is folded into the single prompt token at schema
// index 2, and the child-marker list stays bare. Verified against Python by
// monkey-patching `SchemaTransformer._transform_schema`.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use gliner25_rs::{
    BoundaryConfig, BoundaryEngine, BoundaryParams, Chunker, OverlapPolicy, SchemaTask,
    pair_relations,
};
use serde::Deserialize;

/// A JSON object deserialized into insertion-ordered pairs.
///
/// `serde_json`'s default `Map` is a `BTreeMap`, which would silently reorder
/// the label list and change the prompt. The schema order is part of the
/// prompt, so it has to survive parsing.
#[derive(Debug, Default, Clone)]
struct OrderedMap(Vec<(String, String)>);

impl<'de> Deserialize<'de> for OrderedMap {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> serde::de::Visitor<'de> for V {
            type Value = OrderedMap;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("an object of name -> description")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut m: A,
            ) -> Result<OrderedMap, A::Error> {
                let mut out = Vec::new();
                while let Some((k, v)) = m.next_entry::<String, String>()? {
                    out.push((k, v));
                }
                Ok(OrderedMap(out))
            }
        }
        d.deserialize_map(V)
    }
}

#[derive(Debug, Deserialize)]
struct Scenario {
    name: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct Spec {
    model: String,
    threshold: f32,
    chunk_size: usize,
    chunk_overlap: usize,
    overlap_policy: String,
    entity_types: OrderedMap,
    #[serde(default)]
    relation_types: OrderedMap,
    scenarios: Vec<Scenario>,
}

fn arg(name: &str) -> Option<String> {
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        if a == name {
            return it.next();
        }
    }
    None
}

fn main() -> Result<()> {
    let models = arg("--models").ok_or_else(|| anyhow!("--models <dir> is required"))?;
    let scen_path = arg("--scenarios").ok_or_else(|| anyhow!("--scenarios <file> required"))?;
    let out_path = arg("--out").unwrap_or_else(|| "rust_output.json".to_string());
    let model_label = arg("--model-label");

    let spec: Spec = serde_json::from_slice(&std::fs::read(&scen_path)?)
        .with_context(|| format!("parsing {scen_path}"))?;

    gliner25_rs::init("cognee-parity");

    let t0 = Instant::now();
    let mut engine = BoundaryEngine::new(BoundaryConfig::new(&models))?;
    let load_seconds = t0.elapsed().as_secs_f64();

    // ── schema ────────────────────────────────────────────────────────────
    // Group 0: entities, labels bare, descriptions folded into prompt_str.
    // Groups 1..: one per relation, prompt_str "<name>: <description>", roles
    // ["head", "tail"] — exactly what Python's `_process_relations` emits
    // (`prompt=relation_descriptions.get(parent)` ->
    //  `prompt_str = f"{parent}: {prompt}"`).
    let entity_names: Vec<String> = spec.entity_types.0.iter().map(|(k, _)| k.clone()).collect();
    let mut tasks = vec![SchemaTask::Entities(entity_names.clone())];
    let mut descriptions: Vec<Vec<(String, String)>> = vec![spec.entity_types.0.clone()];
    // prompt_str of each relation group -> bare relation name, for the output keys
    let mut rel_key: BTreeMap<String, String> = BTreeMap::new();
    for (name, desc) in &spec.relation_types.0 {
        let prompt = if desc.is_empty() {
            name.clone()
        } else {
            format!("{name}: {desc}")
        };
        rel_key.insert(prompt.clone(), name.clone());
        tasks.push(SchemaTask::Relations(
            prompt,
            vec!["head".to_string(), "tail".to_string()],
        ));
        descriptions.push(Vec::new());
    }

    // Ablation switch: drops both the `[DESCRIPTION]` block on the entity
    // group and the `": <desc>"` suffix on every relation prompt, i.e. what
    // the unpatched crate could express.
    let no_desc = std::env::args().any(|a| a == "--no-descriptions");
    if no_desc {
        descriptions = tasks.iter().map(|_| Vec::new()).collect();
        for t in tasks.iter_mut() {
            if let SchemaTask::Relations(name, _) = t {
                if let Some((bare, _)) = name.clone().split_once(": ") {
                    *name = bare.to_string();
                }
            }
        }
        rel_key = rel_key.values().map(|v| (v.clone(), v.clone())).collect();
    }

    let params = BoundaryParams {
        threshold: spec.threshold,
        overlap_policy: OverlapPolicy::parse(&spec.overlap_policy),
        descriptions,
        ..Default::default()
    };
    let chunker = Chunker::new(spec.chunk_size, spec.chunk_overlap)?;

    let mut rows = Vec::new();
    for sc in &spec.scenarios {
        let t = Instant::now();
        let out = engine.extract_long_with(&sc.text, &tasks, &params, chunker)?;
        let seconds = t.elapsed().as_secs_f64();

        let mut ents: BTreeMap<String, BTreeSet<String>> = entity_names
            .iter()
            .map(|n| (n.clone(), BTreeSet::new()))
            .collect();
        for m in &out.mentions {
            if m.task == "entities" {
                ents.entry(m.field.clone()).or_default().insert(m.text.clone());
            }
        }

        let mut rels: BTreeMap<String, BTreeSet<String>> = spec
            .relation_types
            .0
            .iter()
            .map(|(n, _)| (n.clone(), BTreeSet::new()))
            .collect();
        for (h, t, prompt) in pair_relations(&out.mentions, &tasks) {
            let key = rel_key.get(&prompt).cloned().unwrap_or(prompt);
            rels.entry(key).or_default().insert(format!("{}|{}", h.text, t.text));
        }

        let ent_json: BTreeMap<String, Vec<String>> = ents
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().cloned().collect()))
            .collect();
        let rel_json: BTreeMap<String, Vec<String>> = rels
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().cloned().collect()))
            .collect();
        let entity_count: usize = ent_json.values().map(|v| v.len()).sum();
        let relation_count: usize = rel_json.values().map(|v| v.len()).sum();

        eprintln!(
            "  {:10} {:5} words  {:7.3}s  ents={:3} rels={:3}",
            sc.name,
            sc.text.split_whitespace().count(),
            seconds,
            entity_count,
            relation_count
        );

        rows.push(serde_json::json!({
            "name": sc.name,
            "words": sc.text.split_whitespace().count(),
            "seconds": (seconds * 10000.0).round() / 10000.0,
            "entities": ent_json,
            "relations": rel_json,
            "entity_count": entity_count,
            "relation_count": relation_count,
        }));
    }

    let payload = serde_json::json!({
        "model": model_label.unwrap_or(spec.model),
        "threshold": spec.threshold,
        "chunk_size": spec.chunk_size,
        "chunk_overlap": spec.chunk_overlap,
        "overlap_policy": spec.overlap_policy,
        "load_seconds": (load_seconds * 1000.0).round() / 1000.0,
        "scenarios": rows,
    });
    std::fs::write(&out_path, serde_json::to_string_pretty(&payload)?)?;
    eprintln!("wrote {out_path}");
    Ok(())
}
