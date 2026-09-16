// COGNEE-EVAL: the parity gate.
//
// Runs the GLiNER2.5 boundary engine over the four cognee parity scenarios and
// scores the result against `fixtures/py_reference.json`, the output of the
// Python `gliner2` 2.0.0 pipeline cognee is replacing. This is the promoted
// form of what used to be `examples/cognee_parity.rs`: the same prompt
// construction, now runnable from `cargo test` and self-scoring, so the
// measurement the whole backend choice rests on cannot silently rot.
//
// Link mode: none required. The numbers below were observed byte-identical
// across static `download-binaries` (debug and release), `GLINER2_PRECISION=fp32`,
// and `--features load-dynamic` with `ORT_DYLIB_PATH`. So this test carries no
// `required-features`, needs no `--release`, and runs in the default profile in
// roughly four seconds once the model is on disk.
//
// WHAT THE ASSERTIONS MEAN
//
// * Entities are asserted as **exact set equality** with Python, per scenario.
//   They are not merely scoring 1.000/1.000 — the sets are identical, and were
//   in every configuration tried. That is a real, non-vacuous claim.
//
// * Relations are asserted only as a **macro-average floor** across the four
//   scenarios (`REL_AGREEMENT_*_FLOOR`), and those floors are a drift tripwire,
//   not a quality bar. Two reasons, both load-bearing:
//
//   1. Per scenario the agreement is far below the mean — precision drops to
//      0.205 on `very_long` and 0.400 on `medium` on a *correct* engine. Any
//      per-scenario relation assertion fails a healthy build. Never add one.
//
//   2. Agreement with Python is not correctness. On `long` the two
//      implementations agree on 18 edges of which 7 are factually false
//      (`works_for::Tim Cook|Microsoft`, `produces::Apple|Azure`,
//      `produces::Apple|Windows`, `produces::Microsoft|Apple Watch`,
//      `acquired::Apple Inc.|GitHub`, `acquired::Microsoft|Beats Electronics`,
//      `headquartered_in::Apple Inc.|Redmond, Washington`). A relation floor
//      therefore asserts reproduction of Python's hallucinations along with its
//      hits. It exists so stage 7's neural scorer can be seen to move the
//      behaviour, and a change that *lowers* agreement may well be an
//      improvement — reassess the floors then rather than chasing them. The
//      factual bar is MASTER_PLAN Gate D (hand-annotated F1), not this file.
//
// The full per-scenario table and the set diffs are printed before any
// assertion runs, so a failure on an unfamiliar machine (hardware
// non-determinism) is distinguishable from a genuine regression from the
// `--nocapture` output alone.
//
// ENVIRONMENT
//
//   GLINER25_MODELS              export directory; default `<workspace>/models/
//                                gliner2.5-base-v1-onnx`. Absent -> the test skips.
//   COGNEE_PARITY_OUT            also write the contract JSON here (for
//                                `_shared/compare_rust_python.py`).
//   COGNEE_PARITY_SCENARIOS      score a different spec file instead of the fixture.
//   COGNEE_PARITY_NO_DESCRIPTIONS=1  ablation: drop every schema description.
//
// The last two are ad-hoc exploration switches inherited from the example. They
// change the prompt or the input, so the fixture reference no longer applies and
// the gate assertions are skipped; such a run prints `COGNEE-PARITY-ABLATION`
// instead of the `COGNEE-PARITY-RAN` marker.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use gliner25_rs::{
    BoundaryConfig, BoundaryEngine, BoundaryParams, Chunker, OverlapPolicy, SchemaTask,
    pair_relations,
};
use serde::Deserialize;

/// The scenario spec. Copied from `_shared/`; see `fixtures/README.md`.
const SCENARIOS_JSON: &str = include_str!("fixtures/cognee_parity_scenarios.json");
/// Python `gliner2` 2.0.0 ground truth. Copied from `_shared/`.
const PY_REFERENCE_JSON: &str = include_str!("fixtures/py_reference.json");

/// Share of Python's relation set this engine reproduces. Measured mean: 0.829.
/// A tripwire for drift, **not** a quality bar — see the module comment.
const REL_AGREEMENT_RECALL_FLOOR: f64 = 0.80;
/// Share of this engine's relation set Python agrees with. Measured mean: 0.542.
/// A tripwire for drift, **not** a quality bar — see the module comment.
const REL_AGREEMENT_PRECISION_FLOOR: f64 = 0.50;

/// Fixture order. Asserted, so a truncated run fails instead of passing on a
/// mean taken over one scenario.
const EXPECTED_SCENARIOS: [&str; 4] = ["short", "medium", "long", "very_long"];

// ── fixture types ────────────────────────────────────────────────────────────

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

/// One scenario of `py_reference.json` — only the two sections that are scored.
#[derive(Debug, Deserialize)]
struct RefScenario {
    name: String,
    #[serde(default)]
    entities: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    relations: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct Reference {
    scenarios: Vec<RefScenario>,
}

/// What this engine produced for one scenario, in contract shape.
struct Row {
    name: String,
    words: usize,
    seconds: f64,
    entities: BTreeMap<String, Vec<String>>,
    relations: BTreeMap<String, Vec<String>>,
}

// ── scoring: a direct port of `_shared/compare_rust_python.py` ───────────────
//
// Ported rather than shelled out: the script lives outside this fork, and
// `cargo test` must not need a `python3` on PATH. A port also turns a failure
// into a printed set difference instead of parsed stdout.

/// `"key::value"` over a `{key: [values]}` section — `flat()` in the script.
fn flat(section: &BTreeMap<String, Vec<String>>) -> BTreeSet<String> {
    section
        .iter()
        .flat_map(|(key, values)| values.iter().map(move |v| format!("{key}::{v}")))
        .collect()
}

/// Share of `py` that `rs` reproduces; 1.0 when `py` is empty.
fn recall(py: &BTreeSet<String>, rs: &BTreeSet<String>) -> f64 {
    if py.is_empty() {
        return 1.0;
    }
    py.intersection(rs).count() as f64 / py.len() as f64
}

/// Share of `rs` that `py` agrees with; 1.0 when `rs` is empty.
fn precision(py: &BTreeSet<String>, rs: &BTreeSet<String>) -> f64 {
    if rs.is_empty() {
        return 1.0;
    }
    py.intersection(rs).count() as f64 / rs.len() as f64
}

/// 1.0 when both sides are empty, as in the script.
fn jaccard(py: &BTreeSet<String>, rs: &BTreeSet<String>) -> f64 {
    let union = py.union(rs).count();
    if union == 0 {
        return 1.0;
    }
    py.intersection(rs).count() as f64 / union as f64
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = values.collect();
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

// ── model location ──────────────────────────────────────────────────────────

/// Accepts a directory only when it actually holds an export, so a
/// half-populated one skips instead of failing obscurely inside ONNX Runtime.
fn export_dir(dir: PathBuf) -> Option<PathBuf> {
    let populated =
        dir.join("boundary_manifest.json").is_file() && dir.join("tokenizer.json").is_file();
    populated.then_some(dir)
}

/// Resolution order:
///   1. `$GLINER25_MODELS` — explicit override.
///   2. `<workspace root>/models/gliner2.5-base-v1-onnx` — the documented
///      default. `models/` is gitignored, so symlinking an export there is the
///      intended way to arm this test:
///      `ln -s <export> <workspace>/models/gliner2.5-base-v1-onnx`
///   3. `None` -> the test skips.
///
/// No machine-specific absolute path appears in committed source, and
/// `BoundaryConfig::new` leaves `hub: None` unless `or_download` is called, so a
/// missing model can never turn this test into a 1.5 GB download.
fn models_dir() -> Option<PathBuf> {
    if let Some(explicit) = std::env::var_os("GLINER25_MODELS") {
        return export_dir(PathBuf::from(explicit));
    }
    // Model directories are gitignored and shared by every crate, so they live
    // at the workspace root rather than per crate — same walk as
    // `test_support::find_tokenizer`.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent()?.parent()?;
    export_dir(root.join("models/gliner2.5-base-v1-onnx"))
}

// ── the test ────────────────────────────────────────────────────────────────

#[test]
fn cognee_parity_matches_python_reference() {
    let Some(models) = models_dir() else {
        eprintln!(
            "⚠️  Skipping cognee parity: no GLiNER2.5 export found \
             (set GLINER25_MODELS or symlink <workspace>/models/gliner2.5-base-v1-onnx); \
             test skipped"
        );
        return;
    };
    run(&models).expect("cognee parity harness failed");
}

fn run(models: &Path) -> Result<()> {
    // Ad-hoc switches inherited from the example. Either one invalidates the
    // fixture reference, so they turn the gate assertions off (and say so).
    let scenarios_override = std::env::var_os("COGNEE_PARITY_SCENARIOS");
    let no_desc = std::env::var_os("COGNEE_PARITY_NO_DESCRIPTIONS").is_some();
    let gate = scenarios_override.is_none() && !no_desc;

    let spec: Spec = match &scenarios_override {
        Some(path) => {
            let raw = std::fs::read(path)
                .with_context(|| format!("reading COGNEE_PARITY_SCENARIOS {path:?}"))?;
            serde_json::from_slice(&raw)
                .with_context(|| format!("parsing COGNEE_PARITY_SCENARIOS {path:?}"))?
        }
        None => serde_json::from_str(SCENARIOS_JSON)
            .context("parsing fixtures/cognee_parity_scenarios.json")?,
    };
    let reference: Reference =
        serde_json::from_str(PY_REFERENCE_JSON).context("parsing fixtures/py_reference.json")?;
    let reference: BTreeMap<&str, &RefScenario> = reference
        .scenarios
        .iter()
        .map(|s| (s.name.as_str(), s))
        .collect();

    gliner25_rs::init("cognee-parity");

    let config = BoundaryConfig::new(models);
    // Recorded for the marker line only: fp16-iobinding and fp32 produced
    // byte-identical output here, so precision is not a variable in this gate.
    let resolved_precision = config.precision;
    let link = if cfg!(feature = "load-dynamic") {
        "dynamic"
    } else {
        "static"
    };

    let t0 = Instant::now();
    let mut engine = BoundaryEngine::new(config)
        .with_context(|| format!("loading the GLiNER2.5 export at {}", models.display()))?;
    let load_seconds = t0.elapsed().as_secs_f64();

    // ── schema ──────────────────────────────────────────────────────────────
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
    if no_desc {
        descriptions = tasks.iter().map(|_| Vec::new()).collect();
        for t in tasks.iter_mut() {
            if let SchemaTask::Relations(name, _) = t
                && let Some(bare) = name.split_once(": ").map(|(b, _)| b.to_string())
            {
                *name = bare;
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

    // ── extraction ──────────────────────────────────────────────────────────
    // One engine, one load, all scenarios in a single `#[test]`: split into
    // four, cargo would run them concurrently and reload 1.5 GB each time.
    let mut rows: Vec<Row> = Vec::new();
    for sc in &spec.scenarios {
        let t = Instant::now();
        let out = engine
            .extract_long_with(&sc.text, &tasks, &params, chunker)
            .with_context(|| format!("extracting scenario `{}`", sc.name))?;
        let seconds = t.elapsed().as_secs_f64();

        let mut ents: BTreeMap<String, BTreeSet<String>> = entity_names
            .iter()
            .map(|n| (n.clone(), BTreeSet::new()))
            .collect();
        for m in &out.mentions {
            if m.task == "entities" {
                ents.entry(m.field.clone())
                    .or_default()
                    .insert(m.text.clone());
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
            rels.entry(key)
                .or_default()
                .insert(format!("{}|{}", h.text, t.text));
        }

        rows.push(Row {
            name: sc.name.clone(),
            words: sc.text.split_whitespace().count(),
            seconds,
            entities: ents
                .iter()
                .map(|(k, v)| (k.clone(), v.iter().cloned().collect()))
                .collect(),
            relations: rels
                .iter()
                .map(|(k, v)| (k.clone(), v.iter().cloned().collect()))
                .collect(),
        });
    }

    // ── contract JSON, for `compare_rust_python.py` and stage 6 ─────────────
    if let Some(out_path) = std::env::var_os("COGNEE_PARITY_OUT") {
        let payload = serde_json::json!({
            "model": spec.model,
            "threshold": spec.threshold,
            "chunk_size": spec.chunk_size,
            "chunk_overlap": spec.chunk_overlap,
            "overlap_policy": spec.overlap_policy,
            "load_seconds": (load_seconds * 1000.0).round() / 1000.0,
            "scenarios": rows.iter().map(|r| serde_json::json!({
                "name": r.name,
                "words": r.words,
                "seconds": (r.seconds * 10000.0).round() / 10000.0,
                "entities": r.entities,
                "relations": r.relations,
                "entity_count": r.entities.values().map(|v| v.len()).sum::<usize>(),
                "relation_count": r.relations.values().map(|v| v.len()).sum::<usize>(),
            })).collect::<Vec<_>>(),
        });
        std::fs::write(&out_path, serde_json::to_string_pretty(&payload)?)
            .with_context(|| format!("writing COGNEE_PARITY_OUT {out_path:?}"))?;
        eprintln!("wrote {out_path:?}");
    }

    // ── scoring ─────────────────────────────────────────────────────────────
    struct Scored {
        name: String,
        words: usize,
        py_ents: BTreeSet<String>,
        rs_ents: BTreeSet<String>,
        py_rels: BTreeSet<String>,
        rs_rels: BTreeSet<String>,
    }
    let scored: Vec<Scored> = rows
        .iter()
        .filter_map(|r| {
            let py = reference.get(r.name.as_str())?;
            Some(Scored {
                name: r.name.clone(),
                words: r.words,
                py_ents: flat(&py.entities),
                rs_ents: flat(&r.entities),
                py_rels: flat(&py.relations),
                rs_rels: flat(&r.relations),
            })
        })
        .collect();

    // ── report, always, before any assertion ────────────────────────────────
    eprintln!();
    eprintln!(
        "cognee parity — model {} — load {load_seconds:.3}s — precision {resolved_precision:?} — link {link}",
        models.display()
    );
    eprintln!(
        "{:12} {:>6} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6}",
        "scenario", "words", "ent_J", "ent_R", "ent_P", "rel_J", "rel_R", "rel_P"
    );
    eprintln!("{}", "-".repeat(66));
    for s in &scored {
        eprintln!(
            "{:12} {:6} | {:6.3} {:6.3} {:6.3} | {:6.3} {:6.3} {:6.3}",
            s.name,
            s.words,
            jaccard(&s.py_ents, &s.rs_ents),
            recall(&s.py_ents, &s.rs_ents),
            precision(&s.py_ents, &s.rs_ents),
            jaccard(&s.py_rels, &s.rs_rels),
            recall(&s.py_rels, &s.rs_rels),
            precision(&s.py_rels, &s.rs_rels),
        );
    }
    eprintln!();
    for s in &scored {
        report_diff(&s.name, "entity", &s.py_ents, &s.rs_ents);
        report_diff(&s.name, "relation", &s.py_rels, &s.rs_rels);
    }

    let mean_ent_recall = mean(scored.iter().map(|s| recall(&s.py_ents, &s.rs_ents)));
    let mean_ent_precision = mean(scored.iter().map(|s| precision(&s.py_ents, &s.rs_ents)));
    let mean_rel_recall = mean(scored.iter().map(|s| recall(&s.py_rels, &s.rs_rels)));
    let mean_rel_precision = mean(scored.iter().map(|s| precision(&s.py_rels, &s.rs_rels)));
    eprintln!("mean entity   recall {mean_ent_recall:.3}  precision {mean_ent_precision:.3}");
    eprintln!("mean relation recall {mean_rel_recall:.3}  precision {mean_rel_precision:.3}");
    eprintln!();

    if !gate {
        eprintln!(
            "COGNEE-PARITY-ABLATION scenarios={} scenarios_override={} no_descriptions={no_desc} \
             ent_R={mean_ent_recall:.3} ent_P={mean_ent_precision:.3} \
             rel_R={mean_rel_recall:.3} rel_P={mean_rel_precision:.3} \
             precision={resolved_precision:?} link={link} — gate assertions SKIPPED, \
             this run does not count as parity",
            scored.len(),
            scenarios_override.is_some(),
        );
        return Ok(());
    }

    // ── 1. shape guard: an empty or truncated run must fail, not pass ───────
    let names: Vec<&str> = rows.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(
        names, EXPECTED_SCENARIOS,
        "expected exactly the four fixture scenarios, in order"
    );
    assert_eq!(
        scored.len(),
        EXPECTED_SCENARIOS.len(),
        "every scenario must have a py_reference.json counterpart"
    );
    let total_entities: usize = scored.iter().map(|s| s.rs_ents.len()).sum();
    assert!(
        total_entities > 0,
        "the engine produced no entities at all — parity would be vacuous"
    );
    // Both relation sides must be non-empty, or the floors below are vacuous:
    // recall() returns 1.0 against an empty reference and precision() returns
    // 1.0 against an empty candidate.
    let py_rels: usize = scored.iter().map(|s| s.py_rels.len()).sum();
    let rs_rels: usize = scored.iter().map(|s| s.rs_rels.len()).sum();
    assert!(
        py_rels > 0 && rs_rels > 0,
        "relation sets are empty on at least one side (reference {py_rels}, engine {rs_rels}); \
         a zero on either side means the fixture or the engine is broken, \
         not that agreement is perfect"
    );

    // ── 2. entities: exact set equality, per scenario ───────────────────────
    // Stronger than recall == precision == 1.000, and it cannot pass vacuously.
    for s in &scored {
        assert_eq!(
            s.py_ents, s.rs_ents,
            "entity sets differ for scenario `{}` (see the diff above)",
            s.name
        );
    }

    // ── 3. entity means: redundant given (2), asserted because they are the
    //       numbers the stage brief names ──────────────────────────────────
    assert!(
        mean_ent_recall >= 1.0,
        "mean entity recall {mean_ent_recall:.3} < 1.000"
    );
    assert!(
        mean_ent_precision >= 1.0,
        "mean entity precision {mean_ent_precision:.3} < 1.000"
    );

    // ── 4. relations: macro-average floors ONLY ─────────────────────────────
    // Never assert per scenario (precision is 0.205 on `very_long` when
    // everything is working), and never add an upper bound — an improvement
    // must not fail the gate.
    assert!(
        mean_rel_recall >= REL_AGREEMENT_RECALL_FLOOR,
        "mean relation agreement recall {mean_rel_recall:.3} < {REL_AGREEMENT_RECALL_FLOOR:.3}; \
         compare the per-scenario table above — a uniform drop suggests a regression, \
         one scenario moving suggests hardware non-determinism"
    );
    assert!(
        mean_rel_precision >= REL_AGREEMENT_PRECISION_FLOOR,
        "mean relation agreement precision {mean_rel_precision:.3} < {REL_AGREEMENT_PRECISION_FLOOR:.3}; \
         compare the per-scenario table above — a uniform drop suggests a regression, \
         one scenario moving suggests hardware non-determinism"
    );

    // A skipped Rust test still prints `ok`, so the only evidence this gate
    // actually ran against a model is this line. VERIFY.md must contain it.
    eprintln!(
        "COGNEE-PARITY-RAN scenarios={} ent_R={mean_ent_recall:.3} ent_P={mean_ent_precision:.3} \
         rel_R={mean_rel_recall:.3} rel_P={mean_rel_precision:.3} \
         precision={resolved_precision:?} link={link}",
        scored.len()
    );
    Ok(())
}

/// Prints the two-way difference for one section of one scenario, capped the
/// way `compare_rust_python.py` caps it.
fn report_diff(scenario: &str, section: &str, py: &BTreeSet<String>, rs: &BTreeSet<String>) {
    let only_py: Vec<&String> = py.difference(rs).collect();
    let only_rs: Vec<&String> = rs.difference(py).collect();
    if only_py.is_empty() && only_rs.is_empty() {
        return;
    }
    eprintln!("[{scenario}] {section} diffs");
    for x in only_py.iter().take(12) {
        eprintln!("   python only : {x}");
    }
    for x in only_rs.iter().take(12) {
        eprintln!("   rust only   : {x}");
    }
    if only_py.len() > 12 || only_rs.len() > 12 {
        eprintln!(
            "   ... (+{} py, +{} rs)",
            only_py.len().saturating_sub(12),
            only_rs.len().saturating_sub(12)
        );
    }
    eprintln!();
}
