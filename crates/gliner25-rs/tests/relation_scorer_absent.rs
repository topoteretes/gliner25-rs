//! **Delta 6: an export with no relation scorer is named, not silently skipped.**
//!
//! `BoundaryManifest::enable_relation_scorer` is `#[serde(default)]`, and the
//! relation block in `extract_once` used to be gated on it:
//!
//! ```text
//! if params.decode_relations && self.manifest.enable_relation_scorer {
//! ```
//!
//! With the key absent the whole block was skipped — no error, no warning — and
//! the caller got a `BoundaryOutput` whose `relations` vec was empty, which is
//! indistinguishable from "this text has no relations". cognee measured the
//! consequence on one sentence: the certified `gliner2.5-base-v1` export gives
//! 4 entities / **2 relations**; an export without the scorer gives 3 entities /
//! **0 relations**, and every edge of the knowledge graph vanished in silence.
//!
//! This test pins the two halves of the fix:
//!
//! 1. an entity-only schema against a scorer-less export still works, exactly
//!    as before — the fix must not break the older exports the `#[serde(default)]`
//!    exists for;
//! 2. a schema carrying a `SchemaTask::Relations` group against the same export
//!    is an `IncompleteModelDir` error naming `enable_relation_scorer`.
//!
//! Arm it the same way as `cognee_contract.rs`: `GLINER25_MODELS`, or a
//! `models/gliner2.5-base-v1-onnx` symlink at the workspace root. Without one it
//! prints `⚠️  Skipping …` and returns.
//!
//! ```text
//! cargo test -p gliner25-rs --test relation_scorer_absent -- --nocapture
//! ```

use std::path::{Path, PathBuf};

use gliner25_rs::{BoundaryConfig, BoundaryEngine, BoundaryParams, SchemaTask};

/// Same resolution order as `cognee_contract.rs`, so one export arms both.
fn models_dir() -> Option<PathBuf> {
    let populated = |dir: PathBuf| {
        (dir.join("boundary_manifest.json").is_file() && dir.join("tokenizer.json").is_file())
            .then_some(dir)
    };
    if let Some(explicit) = std::env::var_os("GLINER25_MODELS") {
        return populated(PathBuf::from(explicit));
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent()?.parent()?;
    populated(root.join("models/gliner2.5-base-v1-onnx"))
}

/// A mirror of the real export, by symlink, with the relation scorer removed
/// and `enable_relation_scorer` deleted from the manifest.
///
/// Thirty-odd symlinks and one rewritten 3 KB manifest rather than a copy of
/// 1.5 GB — and the real export, which every other gate in this repo reads, is
/// never written to. The key is *removed* rather than set to `false` because
/// that is what an export predating the scorer actually looks like.
fn mirror_without_a_relation_scorer(src: &Path) -> PathBuf {
    let dir = std::env::temp_dir().join("gliner25-rs-no-relation-scorer");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("creating the mirror");
    for entry in std::fs::read_dir(src).expect("reading the export") {
        let entry = entry.expect("a directory entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("relation_scorer") || name.starts_with("relation_settings") {
            continue;
        }
        if name == "boundary_manifest.json" {
            let text = std::fs::read_to_string(entry.path()).expect("reading the manifest");
            let mut manifest: serde_json::Value =
                serde_json::from_str(&text).expect("parsing the manifest");
            manifest
                .as_object_mut()
                .expect("the manifest is an object")
                .remove("enable_relation_scorer");
            std::fs::write(dir.join(&name), manifest.to_string()).expect("writing the manifest");
            continue;
        }
        std::os::unix::fs::symlink(entry.path(), dir.join(&name)).expect("symlinking a fragment");
    }
    dir
}

#[test]
fn a_relation_schema_against_a_scorerless_export_is_an_error() {
    let Some(models) = models_dir() else {
        eprintln!(
            "⚠️  Skipping the relation-scorer refusal: no GLiNER2.5 export found \
             (set GLINER25_MODELS or symlink <workspace>/models/gliner2.5-base-v1-onnx); \
             test skipped"
        );
        return;
    };
    gliner25_rs::init("gliner25-relation-scorer-absent");

    let mirror = mirror_without_a_relation_scorer(&models);
    // The load itself must succeed — only relations are refused. It also emits
    // the `[gliner25] warning:` line that is the other half of Delta 6.
    let mut engine = BoundaryEngine::new(BoundaryConfig::new(&mirror))
        .expect("a scorer-less export must still load");
    assert!(
        !engine.manifest().enable_relation_scorer,
        "the mirror was built without one"
    );

    let text = "Tim Cook is the chief executive officer of Apple Inc.";
    let params = BoundaryParams::default();
    assert!(
        params.decode_relations,
        "the default is what the silent skip used to run under"
    );

    // 1. Entities alone: unchanged, and this is the behaviour the
    //    `#[serde(default)]` on `enable_relation_scorer` exists to preserve.
    let entities_only = vec![SchemaTask::Entities(vec![
        "person".to_string(),
        "organization".to_string(),
    ])];
    let out = engine
        .extract_with(text, &entities_only, &params)
        .expect("an entity-only schema is unaffected by a missing relation scorer");
    println!(
        "GLINER25-NO-SCORER entity-only mentions={} relations={}",
        out.mentions.len(),
        out.relations.len()
    );
    assert!(!out.mentions.is_empty(), "the export still finds entities");

    // 2. A relation group is the explicit request, and it is refused by name.
    let with_relations = vec![
        SchemaTask::Entities(vec!["person".to_string(), "organization".to_string()]),
        SchemaTask::Relations(
            "works_for: employment".to_string(),
            vec!["head".to_string(), "tail".to_string()],
        ),
    ];
    let err = engine
        .extract_with(text, &with_relations, &params)
        .expect_err("a scorer-less export must refuse a relation schema, not return no edges");
    let msg = format!("{err:#}");
    println!("GLINER25-NO-SCORER {msg}");
    assert!(msg.contains("enable_relation_scorer"), "got {msg}");
    assert!(msg.contains("relation"), "got {msg}");

    let _ = std::fs::remove_dir_all(&mirror);
}
