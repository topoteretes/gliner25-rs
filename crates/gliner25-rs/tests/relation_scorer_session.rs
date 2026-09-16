// COGNEE-EVAL: stage 7, sub-stage 7b-3 — the relation scorer session.
//
// Drives `BoundaryEngine::score_relation_pairs`, the fifth ONNX fragment, and
// pins the two things that cannot be pinned anywhere else.
//
// 1. THE P = 0 GUARD, TESTED WHERE IT LIVES
//
// The exported graph cannot run with zero pairs: `num_pairs >= 1` was baked in
// at tracing time and onnxruntime fails inside `node_add_163`
// ("Can broadcast 0 by 0 or 1. 768 is invalid"). Python never reaches the model
// either — it early-returns at `boundary/engine.py:818-819` and again at
// `relations.py:335-336`.
//
// 7b-2 measured that the generator's own 28 tests **cannot** defend the guard's
// placement: moving it earlier as a pure relocation was 0 red out of 28,
// because inside the generator "the argument pool is empty" and "the proposal
// set is empty" agree on every fixture case. They stop agreeing exactly when
// the answer is used to decide whether to call the session — which is here.
//
// So both P = 0 route tests below **warm the session up with a real P > 0 call
// first**, on the same engine. Only then do they make the empty call. That
// removes the one way this test could pass for the wrong reason: a guard that
// "works" merely because the fragment was never loaded.
//
// The fixture's two P = 0 cases reach zero by different routes —
// `empty_below_threshold_l512` through an empty argument pool,
// `empty_self_span_only_l64` through a valid head and a valid tail that
// `same_span` kills — and each test asserts its own *precondition*, not just
// "no pairs". A guard placed on the pool satisfies the first and still reaches
// the session on the second.
//
// 2. `text_len` IS A LIVE INPUT, AND IT IS THE ONLY CHANNEL PADDING USES
//
// `dist = |tail_start - head_start| / text_len` (`relations.py:374`) is the one
// place the padded length reaches a relation score. Two tests pin both halves
// of that claim in Rust, mirroring the `relation text_len live` and
// `relation text_len override` rows `verify_parity.py` measures in Python:
// changing `text_len` alone moves the logits by ≥ 1e-02, and changing the
// padding alone (same `text_len`, same words, zeroed tail) does not move them
// at all beyond fp32 noise. Together they are why feeding the window's word
// count while the tensor is `[1, bucket, 768]` is exactly the batch-1
// computation rather than an inconsistency — see `score_relation_pairs`.
//
// ENVIRONMENT
//
//   GLINER25_MODELS   export directory; default
//                     `<workspace>/models/gliner2.5-base-v1-onnx`.
//                     Absent -> the tests skip, as in `cognee_contract.rs`.
//
// A run that really reached the session prints `RELATION-SCORER-SESSION-RAN`.

use std::path::{Path, PathBuf};

use gliner25_rs::{
    BoundaryConfig, BoundaryEngine, CandidateView, Carrier, Precision, ProposedPair,
    RelationProposalSettings, RelationScoreInputs, RelationTypeSpec, generate_pairs,
    generate_pairs_detailed,
};

/// The Python-recorded proposal fixture, pinned by the path
/// `onnx_conversion_scripts/fixtures/README.md` records.
const FIXTURE_JSON: &str =
    include_str!("../../../onnx_conversion_scripts/fixtures/relation_pairs.json");

/// Byte length of the fixture as 7a-3 dumped it (sha256
/// `da0d74879fc2011c068b3305514aa8282a776109925a66592742963d019f0d51`, verified
/// in-process by the sibling gate `relation_proposal_identity.rs`, which runs in
/// the same `cargo test`).
const FIXTURE_BYTES: usize = 421_631;

const CASE_NAMES: [&str; 7] = [
    "small_pool_padded_l64",
    "wide_pool_l512",
    "exact_len_l128",
    "tied_scores_total_l64",
    "tied_scores_banded_l128",
    "empty_below_threshold_l512",
    "empty_self_span_only_l64",
];

// ── fixture ─────────────────────────────────────────────────────────────────

struct Case {
    num_words: usize,
    padded_len: usize,
    queries: usize,
    cand_count: usize,
    indices: Vec<i64>,
    logits: Vec<f32>,
    valid: Vec<bool>,
    query_mask: Vec<bool>,
    specs: Vec<RelationTypeSpec>,
}

impl Case {
    fn view(&self) -> CandidateView<'_> {
        CandidateView {
            indices: &self.indices,
            pair_logits: &self.logits,
            valid_mask: &self.valid,
            query_mask: &self.query_mask,
            queries: self.queries,
            cand_count: self.cand_count,
        }
    }
}

fn fixture() -> serde_json::Value {
    assert_eq!(
        FIXTURE_JSON.len(),
        FIXTURE_BYTES,
        "relation_pairs.json is {} bytes, expected {FIXTURE_BYTES} — the fixture changed, so \
         nothing below proves anything",
        FIXTURE_JSON.len()
    );
    let v: serde_json::Value =
        serde_json::from_str(FIXTURE_JSON).expect("relation_pairs.json is not valid JSON");
    assert_eq!(
        v["schema"].as_str(),
        Some("gliner25.relation_pairs.v1"),
        "unexpected fixture schema"
    );
    let names: Vec<&str> = v["cases"]
        .as_array()
        .expect("cases is not an array")
        .iter()
        .map(|c| c["name"].as_str().expect("case without a name"))
        .collect();
    assert_eq!(names, CASE_NAMES, "the fixture's case list changed");
    v
}

fn case(name: &str) -> Case {
    let v = fixture();
    let c = v["cases"]
        .as_array()
        .expect("cases is not an array")
        .iter()
        .find(|c| c["name"].as_str() == Some(name))
        .unwrap_or_else(|| panic!("fixture has no case named {name}"))
        .clone();

    let usize_at = |k: &str| c[k].as_u64().unwrap_or_else(|| panic!("{name}: no {k}")) as usize;
    let arr = |v: &serde_json::Value| v.as_array().expect("expected an array").clone();

    let mut indices = Vec::new();
    for span in arr(&c["inputs"]["cand_indices"]) {
        let pair = arr(&span);
        assert_eq!(
            pair.len(),
            2,
            "{name}: cand_indices row is not (start, end)"
        );
        indices.push(pair[0].as_i64().expect("start is not an integer"));
        indices.push(pair[1].as_i64().expect("end is not an integer"));
    }
    let logits: Vec<f32> = arr(&c["inputs"]["pair_logits"])
        .iter()
        .map(|v| v.as_f64().expect("logit is not a number") as f32)
        .collect();
    let valid: Vec<bool> = arr(&c["inputs"]["cand_valid_mask"])
        .iter()
        .map(|v| v.as_bool().expect("valid is not a bool"))
        .collect();
    let query_mask: Vec<bool> = arr(&c["inputs"]["query_mask"])
        .iter()
        .map(|v| v.as_bool().expect("query mask is not a bool"))
        .collect();

    let specs: Vec<RelationTypeSpec> = arr(&c["relation_specs"])
        .iter()
        .map(|s| RelationTypeSpec {
            relation_type: s["relation_type"]
                .as_str()
                .expect("relation_type is not a string")
                .to_string(),
            head_query_ids: arr(&s["head_query_ids"])
                .iter()
                .map(|q| q.as_u64().expect("query id is not an integer") as usize)
                .collect(),
            tail_query_ids: arr(&s["tail_query_ids"])
                .iter()
                .map(|q| q.as_u64().expect("query id is not an integer") as usize)
                .collect(),
            allow_self: s["allow_self"].as_bool().expect("allow_self is not a bool"),
        })
        .collect();

    Case {
        num_words: usize_at("num_words"),
        padded_len: usize_at("boundary_states_padded_len"),
        queries: usize_at("queries"),
        cand_count: usize_at("cand_count"),
        indices,
        logits,
        valid,
        query_mask,
        specs,
    }
}

// ── engine ──────────────────────────────────────────────────────────────────

/// Same resolution order as `cognee_contract.rs`: `$GLINER25_MODELS`, then
/// `<workspace>/models/gliner2.5-base-v1-onnx`, then skip. `BoundaryConfig::new`
/// leaves `hub: None`, so a missing model can never become a download.
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

/// `None` when there is no export to run against, exactly as the parity gate
/// skips. Anything else is a hard failure.
fn engine(precision: Option<Precision>) -> Option<BoundaryEngine> {
    let models = models_dir()?;
    let mut config = BoundaryConfig::new(models);
    if let Some(p) = precision {
        config = config.with_precision(p);
    }
    Some(BoundaryEngine::new(config).expect("the export is present but the engine did not build"))
}

fn skipped(what: &str) {
    eprintln!(
        "⚠️  Skipping {what}: no GLiNER2.5 export found (set GLINER25_MODELS or symlink \
         <workspace>/models/gliner2.5-base-v1-onnx)."
    );
}

// ── synthetic inputs ────────────────────────────────────────────────────────

const HIDDEN: usize = 768;

/// Deterministic pseudo-random word states in `[-0.5, 0.5)`, then the padding.
///
/// The padded rows are **zero**, which is what `routed_gather` produces
/// (`states * mask`) and what Python's `pad_sequence` produces. A test that
/// padded with noise would be testing a tensor the engine never builds.
fn padded_states(num_words: usize, padded_len: usize, seed: u64) -> Vec<f32> {
    assert!(num_words <= padded_len);
    let mut x = seed | 1;
    let mut out = vec![0.0f32; padded_len * HIDDEN];
    for slot in out.iter_mut().take(num_words * HIDDEN) {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *slot = ((x >> 33) as f32 / (1u64 << 31) as f32) - 0.5;
    }
    out
}

fn carrier(data: Vec<f32>, padded_len: usize) -> Carrier {
    Carrier::Host {
        shape: vec![1, padded_len as i64, HIDDEN as i64],
        data,
    }
}

fn query_states(relations: usize, seed: u64) -> Vec<f32> {
    padded_states(relations, relations, seed)
}

fn pair(relation_index: usize, head: (usize, usize), tail: (usize, usize)) -> ProposedPair {
    ProposedPair {
        relation_index,
        head_start: head.0,
        head_end: head.1,
        tail_start: tail.0,
        tail_end: tail.1,
        head_prob: 0.9,
        tail_prob: 0.9,
        head_query: 0,
        tail_query: 0,
        head_slot: 0,
        tail_slot: 0,
    }
}

// ── tests ───────────────────────────────────────────────────────────────────

#[test]
fn the_fixture_is_the_one_7a_3_dumped() {
    // Hard-fails rather than skips: every other test in this file reads it.
    let v = fixture();
    assert_eq!(v["cases"].as_array().map(Vec::len), Some(7));
}

/// Route 1 to `P = 0`: the argument pool is empty, because nothing in the raw
/// candidate pool clears the 0.2 argument-proposal threshold.
#[test]
fn p0_through_an_empty_argument_pool_never_reaches_the_session() {
    let Some(mut engine) = engine(None) else {
        skipped("relation scorer P=0 (empty pool)");
        return;
    };
    let c = case("empty_below_threshold_l512");
    let settings = RelationProposalSettings::checkpoint();
    let relations = c.specs.len();
    let states = carrier(padded_states(c.num_words, c.padded_len, 11), c.padded_len);
    let qh = query_states(relations, 21);
    let qt = query_states(relations, 22);

    // PRECONDITION: this route really is the empty pool — not one valid head
    // slot and not one valid tail slot anywhere.
    let detailed = generate_pairs_detailed(&c.view(), &c.specs, &settings);
    for proposal in &detailed {
        assert_eq!(
            proposal.heads.iter().filter(|s| s.valid).count(),
            0,
            "{}: expected an empty head pool",
            proposal.relation_type
        );
        assert_eq!(
            proposal.tails.iter().filter(|s| s.valid).count(),
            0,
            "{}: expected an empty tail pool",
            proposal.relation_type
        );
    }

    // Warm the session up, so the empty call below cannot pass merely because
    // the fragment was never loaded.
    let warm = vec![pair(0, (0, 2), (5, 7))];
    let scored = engine
        .score_relation_pairs(RelationScoreInputs {
            text_states: &states,
            padded_words: c.padded_len,
            relation_query_head: &qh,
            relation_query_tail: &qt,
            relations,
            pairs: &warm,
            text_len: c.num_words,
        })
        .expect("the warm-up call must reach the session and succeed");
    assert_eq!(scored.len(), 1, "warm-up did not run the fragment");

    let pairs = generate_pairs(&c.view(), &c.specs, &settings);
    assert!(pairs.is_empty(), "fixture case is no longer P = 0");
    let out = engine
        .score_relation_pairs(RelationScoreInputs {
            text_states: &states,
            padded_words: c.padded_len,
            relation_query_head: &qh,
            relation_query_tail: &qt,
            relations,
            pairs: &pairs,
            text_len: c.num_words,
        })
        .expect(
            "P = 0 must return without calling the session; the exported graph was traced with \
             num_pairs >= 1 and onnxruntime fails in node_add_163",
        );
    assert!(out.is_empty(), "P = 0 must produce no logits");
    eprintln!("RELATION-SCORER-SESSION-RAN route=empty_pool warm_up_pairs=1 guarded_pairs=0");
}

/// Route 2 to `P = 0`: the pool is **not** empty — one valid head and one valid
/// tail survive selection — and `same_span` kills the only pair they form.
///
/// This is the case a guard placed on the argument pool gets wrong. It passes
/// route 1 and reaches the session here, which is the ORT crash.
#[test]
fn p0_through_same_span_never_reaches_the_session() {
    let Some(mut engine) = engine(None) else {
        skipped("relation scorer P=0 (same_span)");
        return;
    };
    let c = case("empty_self_span_only_l64");
    let settings = RelationProposalSettings::checkpoint();
    let relations = c.specs.len();
    let states = carrier(padded_states(c.num_words, c.padded_len, 31), c.padded_len);
    let qh = query_states(relations, 41);
    let qt = query_states(relations, 42);

    // PRECONDITION: exactly one valid head and one valid tail, sharing a span.
    // Asserting "no pairs" alone would make this test identical to route 1 and
    // it would stop discriminating between guard placements.
    let detailed = generate_pairs_detailed(&c.view(), &c.specs, &settings);
    assert_eq!(detailed.len(), 1);
    let heads: Vec<_> = detailed[0].heads.iter().filter(|s| s.valid).collect();
    let tails: Vec<_> = detailed[0].tails.iter().filter(|s| s.valid).collect();
    assert_eq!(
        heads.len(),
        1,
        "the head pool must NOT be empty on this route"
    );
    assert_eq!(
        tails.len(),
        1,
        "the tail pool must NOT be empty on this route"
    );
    assert_eq!(
        (heads[0].start, heads[0].end),
        (tails[0].start, tails[0].end),
        "the surviving head and tail must share a span — that is what same_span kills"
    );
    assert!(!c.specs[0].allow_self);

    let warm = vec![pair(0, (2, 4), (9, 11))];
    let scored = engine
        .score_relation_pairs(RelationScoreInputs {
            text_states: &states,
            padded_words: c.padded_len,
            relation_query_head: &qh,
            relation_query_tail: &qt,
            relations,
            pairs: &warm,
            text_len: c.num_words,
        })
        .expect("the warm-up call must reach the session and succeed");
    assert_eq!(scored.len(), 1, "warm-up did not run the fragment");

    let pairs = generate_pairs(&c.view(), &c.specs, &settings);
    assert!(pairs.is_empty(), "fixture case is no longer P = 0");
    let out = engine
        .score_relation_pairs(RelationScoreInputs {
            text_states: &states,
            padded_words: c.padded_len,
            relation_query_head: &qh,
            relation_query_tail: &qt,
            relations,
            pairs: &pairs,
            text_len: c.num_words,
        })
        .expect(
            "P = 0 reached through same_span must return without calling the session — a guard \
             on the argument pool would have called it here",
        );
    assert!(out.is_empty(), "P = 0 must produce no logits");
    eprintln!("RELATION-SCORER-SESSION-RAN route=same_span warm_up_pairs=1 guarded_pairs=0");
}

#[test]
fn the_session_returns_one_finite_logit_per_pair() {
    let Some(mut engine) = engine(Some(Precision::Fp32)) else {
        skipped("relation scorer shape");
        return;
    };
    let (num_words, padded) = (40usize, 64usize);
    let states = carrier(padded_states(num_words, padded, 7), padded);
    let qh = query_states(2, 8);
    let qt = query_states(2, 9);
    let pairs: Vec<ProposedPair> = (0..17)
        .map(|i| pair(i % 2, (i % 5, i % 5 + 2), (i % 7 + 10, i % 7 + 13)))
        .collect();

    let out = engine
        .score_relation_pairs(RelationScoreInputs {
            text_states: &states,
            padded_words: padded,
            relation_query_head: &qh,
            relation_query_tail: &qt,
            relations: 2,
            pairs: &pairs,
            text_len: num_words,
        })
        .expect("the fragment must run");
    assert_eq!(out.len(), pairs.len());
    assert!(out.iter().all(|v| v.is_finite()), "logits: {out:?}");
    eprintln!("RELATION-SCORER-SESSION-RAN pairs={} L={padded}", out.len());
}

/// `text_len` is a genuine input, not a constant the exporter baked in.
///
/// Same states, same pairs, same graph; only the denominator changes. The
/// Python-side twin of this row (`relation text_len live` in
/// `verify_parity.py`) measures 7.805e-02 and uses the same 1e-02 floor.
#[test]
fn text_len_is_a_live_input_and_moves_the_logits() {
    let Some(mut engine) = engine(Some(Precision::Fp32)) else {
        skipped("relation scorer text_len");
        return;
    };
    let (num_words, padded) = (40usize, 512usize);
    let states = carrier(padded_states(num_words, padded, 3), padded);
    let qh = query_states(1, 4);
    let qt = query_states(1, 5);
    // Distances have to be non-zero, or `dist` is 0 whatever the denominator.
    let pairs: Vec<ProposedPair> = (0..8)
        .map(|i| pair(0, (i, i + 2), (i + 20, i + 22)))
        .collect();
    let run = |engine: &mut BoundaryEngine, text_len: usize| {
        engine
            .score_relation_pairs(RelationScoreInputs {
                text_states: &states,
                padded_words: padded,
                relation_query_head: &qh,
                relation_query_tail: &qt,
                relations: 1,
                pairs: &pairs,
                text_len,
            })
            .expect("the fragment must run")
    };
    let at_words = run(&mut engine, num_words);
    let at_bucket = run(&mut engine, padded);
    let swing = at_words
        .iter()
        .zip(&at_bucket)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        swing >= 1e-2,
        "feeding the word count and feeding the bucket produced the same logits (max |Δ| = \
         {swing:.3e}); text_len is not reaching the graph, so the padded-length decision would \
         be silently unobservable"
    );
    eprintln!("RELATION-SCORER-SESSION-RAN text_len_swing={swing:.3e} (40 vs 512)");
}

/// The other half of the claim: with `text_len` held fixed, **padding alone
/// changes nothing**.
///
/// This is why feeding the window's own word count while the tensor is
/// `[1, bucket, 768]` is the batch-1 computation and not an inconsistency: the
/// bucket is invisible to the fragment except through `text_len`.
#[test]
fn padding_past_the_word_count_does_not_move_a_logit() {
    let Some(mut engine) = engine(Some(Precision::Fp32)) else {
        skipped("relation scorer padding transparency");
        return;
    };
    let num_words = 40usize;
    let qh = query_states(1, 14);
    let qt = query_states(1, 15);
    let pairs: Vec<ProposedPair> = (0..8)
        .map(|i| pair(0, (i, i + 2), (i + 20, i + 22)))
        .collect();
    let mut run = |padded: usize| {
        let states = carrier(padded_states(num_words, padded, 13), padded);
        engine
            .score_relation_pairs(RelationScoreInputs {
                text_states: &states,
                padded_words: padded,
                relation_query_head: &qh,
                relation_query_tail: &qt,
                relations: 1,
                pairs: &pairs,
                text_len: num_words,
            })
            .expect("the fragment must run")
    };
    let short = run(64);
    let long = run(512);
    let drift = short
        .iter()
        .zip(&long)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        drift <= 1e-5,
        "padding 40 words to 64 and to 512 moved the logits by {drift:.3e} at a fixed text_len; \
         the fragment is not padding-transparent and the text_len decision needs revisiting"
    );
    eprintln!("RELATION-SCORER-SESSION-RAN padding_drift={drift:.3e} (L=64 vs L=512)");
}

/// `score.masked_fill(~pair_valid, 0.0)` — `relations.py:407`. Exactly `0.0`,
/// not `-inf`, which a port could reasonably have assumed.
///
/// Only the above-range half is reachable from Rust: `ProposedPair::relation_index`
/// is a `usize`, so a negative index is unrepresentable by construction.
#[test]
fn an_out_of_range_relation_index_scores_exactly_zero() {
    let Some(mut engine) = engine(Some(Precision::Fp32)) else {
        skipped("relation scorer out-of-range index");
        return;
    };
    let (num_words, padded) = (40usize, 64usize);
    let states = carrier(padded_states(num_words, padded, 17), padded);
    let qh = query_states(2, 18);
    let qt = query_states(2, 19);
    let pairs = vec![
        pair(0, (0, 2), (10, 12)),
        pair(2, (0, 2), (10, 12)),
        pair(7, (3, 5), (20, 22)),
    ];
    let out = engine
        .score_relation_pairs(RelationScoreInputs {
            text_states: &states,
            padded_words: padded,
            relation_query_head: &qh,
            relation_query_tail: &qt,
            relations: 2,
            pairs: &pairs,
            text_len: num_words,
        })
        .expect("the fragment must run");
    assert_eq!(out.len(), 3);
    assert!(out[0] != 0.0, "the in-range pair should have a real logit");
    assert_eq!(out[1], 0.0, "relation_index == R must score exactly 0.0");
    assert_eq!(out[2], 0.0, "relation_index > R must score exactly 0.0");
}

/// `relations.py:341-343`: no relation queries means every pair scores 0.0.
/// The graph cannot run `R = 0` either, so this must not reach the session.
#[test]
fn no_relation_queries_scores_every_pair_zero() {
    let Some(mut engine) = engine(Some(Precision::Fp32)) else {
        skipped("relation scorer R=0");
        return;
    };
    let (num_words, padded) = (40usize, 64usize);
    let states = carrier(padded_states(num_words, padded, 23), padded);
    let pairs = vec![pair(0, (0, 2), (10, 12)), pair(0, (3, 5), (20, 22))];
    let out = engine
        .score_relation_pairs(RelationScoreInputs {
            text_states: &states,
            padded_words: padded,
            relation_query_head: &[],
            relation_query_tail: &[],
            relations: 0,
            pairs: &pairs,
            text_len: num_words,
        })
        .expect("R = 0 must return zeros rather than running the graph");
    assert_eq!(out, vec![0.0, 0.0]);
}
