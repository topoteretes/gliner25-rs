// COGNEE-EVAL: stage 7, sub-stage 7b-2 — the proposal-identity gate.
//
// Drives `generate_pairs` / `generate_pairs_detailed` (the port of `gliner2`'s
// `TypedRelationPairGenerator`) from the Python-recorded fixture
// `onnx_conversion_scripts/fixtures/relation_pairs.json` and asserts
// **index-for-index identity** with Python across all seven cases.
//
// WHY THIS GATE EXISTS, AND WHY IT IS SHAPED THE WAY IT IS
//
// Python's `select()` re-keys the candidate pool by span `(start, end)` through
// two stable argsorts *before* the descending score sort, so equal scores break
// by position and not by flat index. A natural Rust port that sorts a
// flat-index-ordered vector by score picks a different top-32 and a different
// top-64 and **fails silently**: fragment parity passes, entity parity stays
// 1.000/1.000, every other test stays green, and the relation numbers simply
// never improve. Nothing downstream can detect it. This file is the only thing
// that can.
//
// A GATE THAT ONLY COMPARED PAIRS WOULD BE VACUOUS — MEASURED, NOT THEORISED
//
// 7a-3's `--self-check` ran three deliberately wrong ports against this
// fixture. On the three *ordinary* cases (`small_pool_padded_l64`,
// `wide_pool_l512`, `exact_len_l128`) a port with no `(start, end, flat)`
// re-key moves 33–85 argument slots and still produces **byte-identical final
// pairs**. Only the two tied-score cases make the pairs themselves diverge
// (121/128 and 115/128 rows).
//
// So every case gets three separate tests, deliberately not merged:
//
//   * `<case>_argument_slots`  — `.heads` / `.tails` slot-for-slot. This is the
//     assertion that actually catches the re-key bug on the ordinary cases.
//   * `<case>_pairs`           — only the fields Python's compacted
//     `RelationPairBatch` carries. This is what a naive gate would have
//     checked; on the ordinary cases it is *expected* to be blind to a broken
//     re-key, and keeping it separate is what makes that blindness visible
//     rather than hidden inside a bigger passing test.
//   * `<case>_pair_slots`      — `head_slot` / `tail_slot` per pair, the ranks
//     into the argument lists, so a mismatch localises to argument selection
//     rather than only showing up as different spans.
//
// COMPARISON RULES
//
// `prob` gets a small tolerance: torch's f32 `sigmoid` is not bit-identical to
// `1/(1+exp(-x))` (measured ≤ 2 ulp). **Everything else is exact** —
// `flat_index`, `query`, `cand_slot`, the raw i64 spans, `valid`, `num_pairs`
// and all ordering. A tolerance is never applied to an index or to an order.
//
// THE TWO P=0 CASES REACH ZERO BY DIFFERENT ROUTES
//
// `empty_below_threshold_l512` has an empty argument pool; `empty_self_span_only_l64`
// has a valid head and a valid tail that `same_span` kills. The second exists to
// catch an early return placed **too early** — a guard keyed on "no eligible
// candidates" passes the first and still reaches the session on the second,
// which is the ORT crash 7a-2 found (`num_pairs >= 1` baked in at tracing).
// Two dedicated tests assert the *routes*, not merely that both are empty.
//
// The fixture is hash-pinned in-process (see `sha256_hex`, self-tested against
// the NIST vectors) so a run against edited or truncated bytes fails loudly
// instead of quietly proving nothing. There is no skip path in this file.

use gliner25_rs::{
    ArgumentSlot, CandidateView, ProposedPair, RelationProposal, RelationProposalSettings,
    RelationTypeSpec, generate_pairs, generate_pairs_detailed,
};
use serde::Deserialize;

/// The Python-recorded fixture. Not copied under `crates/` — 7a-3 wrote it on
/// the Python lane while this lane was live, and the README records this exact
/// `include_str!` path as the contract.
const FIXTURE_JSON: &str =
    include_str!("../../../onnx_conversion_scripts/fixtures/relation_pairs.json");

/// `sha256sum onnx_conversion_scripts/fixtures/relation_pairs.json`, as recorded
/// in `onnx_conversion_scripts/fixtures/README.md` and in the stage-7 brief.
const FIXTURE_SHA256: &str = "da0d74879fc2011c068b3305514aa8282a776109925a66592742963d019f0d51";

/// Byte length of the pinned fixture. Redundant with the hash, and cheap enough
/// to keep: it turns a truncated include into an obvious failure.
const FIXTURE_BYTES: usize = 421_631;

/// Absolute tolerance on `prob` only. Comfortably above the measured ≤ 2 ulp
/// sigmoid gap near 1.0 (~2.4e-7) and far below any selection-relevant
/// difference. Never applied to an index, a span, a flag or an order.
const PROB_TOL: f32 = 1e-6;

/// The fixture's case order, asserted, so a truncated or reordered file fails
/// instead of silently gating fewer cases.
const EXPECTED_CASES: [&str; 7] = [
    "small_pool_padded_l64",
    "wide_pool_l512",
    "exact_len_l128",
    "tied_scores_total_l64",
    "tied_scores_banded_l128",
    "empty_below_threshold_l512",
    "empty_self_span_only_l64",
];

// ── fixture types ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct Fixture {
    schema: String,
    settings: FixtureSettings,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct FixtureSettings {
    heads_per_relation: usize,
    tails_per_relation: usize,
    pair_cap: usize,
    argument_threshold: f32,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    queries: usize,
    cand_count: usize,
    flat_pool_size: usize,
    inputs: Inputs,
    relation_specs: Vec<FixtureSpec>,
    proposals: Vec<FixtureProposal>,
    pair_batch: PairBatch,
}

#[derive(Deserialize)]
struct Inputs {
    /// Raw i64 half-open `(start, end)` spans, one per flat candidate.
    cand_indices: Vec<[i64; 2]>,
    /// Raw f32 **logits** — sigmoid is applied inside the generator.
    pair_logits: Vec<f32>,
    cand_valid_mask: Vec<bool>,
    query_mask: Vec<bool>,
}

#[derive(Deserialize)]
struct FixtureSpec {
    relation_index: usize,
    relation_type: String,
    head_query_ids: Vec<usize>,
    tail_query_ids: Vec<usize>,
    allow_self: bool,
}

#[derive(Deserialize)]
struct FixtureProposal {
    relation_index: usize,
    relation_type: String,
    heads: Vec<FixtureSlot>,
    tails: Vec<FixtureSlot>,
    num_pairs: usize,
    pairs: Vec<FixturePair>,
}

#[derive(Deserialize)]
struct FixtureSlot {
    slot: usize,
    flat_index: usize,
    query: usize,
    cand_slot: usize,
    start: i64,
    end: i64,
    prob: f32,
    valid: bool,
}

#[derive(Deserialize)]
struct FixturePair {
    pair_index: usize,
    /// Index into the 64 kept pairs *before* compaction dropped the invalid
    /// survivors. Carried so a mismatch in which kept pair survived compaction
    /// is distinguishable from a mismatch in the keep sort itself.
    keep_position: usize,
    head_slot: usize,
    tail_slot: usize,
    head_start: usize,
    head_end: usize,
    tail_start: usize,
    tail_end: usize,
    head_prob: f32,
    tail_prob: f32,
    head_query: usize,
    tail_query: usize,
}

#[derive(Deserialize)]
struct PairBatch {
    num_pairs: usize,
    rows: Vec<Row>,
}

#[derive(Deserialize)]
struct Row {
    index: usize,
    relation_index: usize,
    head_start: usize,
    head_end: usize,
    tail_start: usize,
    tail_end: usize,
    head_prob: f32,
    tail_prob: f32,
    pair_mask: bool,
    relation_type: String,
}

// ── loading ──────────────────────────────────────────────────────────────────

/// Parses the fixture, failing loudly if it is missing, empty, truncated or
/// edited. There is deliberately no branch here that skips.
fn fixture() -> Fixture {
    assert!(
        !FIXTURE_JSON.is_empty(),
        "relation_pairs.json is empty — regenerate it with \
         onnx_conversion_scripts/dump_relation_pairs.py; this gate never skips"
    );
    assert_eq!(
        FIXTURE_JSON.len(),
        FIXTURE_BYTES,
        "relation_pairs.json is {} bytes, expected {FIXTURE_BYTES}: the fixture was edited or \
         truncated, and the comparison below would no longer be the recorded Python output",
        FIXTURE_JSON.len()
    );
    assert_eq!(
        sha256_hex(FIXTURE_JSON.as_bytes()),
        FIXTURE_SHA256,
        "relation_pairs.json does not hash to the pinned sha256 — do not loosen this, \
         regenerate the fixture and update the hash in fixtures/README.md and the brief"
    );
    let parsed: Fixture =
        serde_json::from_str(FIXTURE_JSON).expect("relation_pairs.json is not valid JSON");
    assert_eq!(
        parsed.schema, "gliner25.relation_pairs.v1",
        "unexpected fixture schema"
    );
    assert_eq!(
        parsed.cases.len(),
        EXPECTED_CASES.len(),
        "fixture case count changed"
    );
    parsed
}

/// Everything one case needs to drive the port, with the borrowed slices owned
/// here so `CandidateView` can point at them.
struct Driver {
    name: String,
    indices: Vec<i64>,
    logits: Vec<f32>,
    valid_mask: Vec<bool>,
    query_mask: Vec<bool>,
    queries: usize,
    cand_count: usize,
    schema: Vec<RelationTypeSpec>,
    settings: RelationProposalSettings,
    case: Case,
}

impl Driver {
    fn view(&self) -> CandidateView<'_> {
        CandidateView {
            indices: &self.indices,
            pair_logits: &self.logits,
            valid_mask: &self.valid_mask,
            query_mask: &self.query_mask,
            queries: self.queries,
            cand_count: self.cand_count,
        }
    }

    fn detailed(&self) -> Vec<RelationProposal> {
        generate_pairs_detailed(&self.view(), &self.schema, &self.settings)
    }

    fn flat(&self) -> Vec<ProposedPair> {
        generate_pairs(&self.view(), &self.schema, &self.settings)
    }
}

fn driver(name: &str) -> Driver {
    let parsed = fixture();
    let settings = RelationProposalSettings {
        heads_per_relation: parsed.settings.heads_per_relation,
        tails_per_relation: parsed.settings.tails_per_relation,
        pair_cap: parsed.settings.pair_cap,
        argument_threshold: parsed.settings.argument_threshold,
    };
    let case = parsed
        .cases
        .into_iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("fixture has no case `{name}`"));

    let flat_len = case.queries * case.cand_count;
    assert_eq!(
        case.flat_pool_size, flat_len,
        "{name}: flat_pool_size disagrees with queries * cand_count"
    );
    assert_eq!(
        case.inputs.cand_indices.len(),
        flat_len,
        "{name}: cand_indices length"
    );
    assert_eq!(
        case.inputs.pair_logits.len(),
        flat_len,
        "{name}: pair_logits length"
    );
    assert_eq!(
        case.inputs.cand_valid_mask.len(),
        flat_len,
        "{name}: cand_valid_mask length"
    );
    assert_eq!(
        case.inputs.query_mask.len(),
        case.queries,
        "{name}: query_mask length"
    );
    assert_eq!(
        case.relation_specs.len(),
        case.proposals.len(),
        "{name}: one proposal per relation spec"
    );

    let mut indices = Vec::with_capacity(flat_len * 2);
    for span in &case.inputs.cand_indices {
        indices.push(span[0]);
        indices.push(span[1]);
    }
    let schema = case
        .relation_specs
        .iter()
        .enumerate()
        .map(|(i, spec)| {
            assert_eq!(
                spec.relation_index, i,
                "{name}: relation_specs are not in relation_index order"
            );
            RelationTypeSpec {
                relation_type: spec.relation_type.clone(),
                head_query_ids: spec.head_query_ids.clone(),
                tail_query_ids: spec.tail_query_ids.clone(),
                allow_self: spec.allow_self,
            }
        })
        .collect();

    Driver {
        name: case.name.clone(),
        indices,
        logits: case.inputs.pair_logits.clone(),
        valid_mask: case.inputs.cand_valid_mask.clone(),
        query_mask: case.inputs.query_mask.clone(),
        queries: case.queries,
        cand_count: case.cand_count,
        schema,
        settings,
        case,
    }
}

// ── the three per-case assertions ────────────────────────────────────────────

/// `.heads` / `.tails`, slot-for-slot. Every field exact except `prob`.
///
/// This is the assertion the gate turns on: on the three ordinary cases it is
/// the *only* one that reacts to a missing `(start, end, flat)` re-key.
fn assert_argument_slots(name: &str) {
    let d = driver(name);
    let got = d.detailed();
    assert_eq!(
        got.len(),
        d.case.proposals.len(),
        "{name}: relation count (Rust {} vs Python {})",
        got.len(),
        d.case.proposals.len()
    );

    for (r, (rust, py)) in got.iter().zip(&d.case.proposals).enumerate() {
        assert_eq!(rust.relation_index, r, "{name}: relation_index at {r}");
        assert_eq!(
            py.relation_index, r,
            "{name}: fixture relation_index at {r}"
        );
        assert_eq!(
            rust.relation_type, py.relation_type,
            "{name} rel {r}: relation_type"
        );
        assert_eq!(
            rust.heads.len(),
            d.settings.heads_per_relation,
            "{name} rel {r}: Rust kept {} head slots, not the requested {}",
            rust.heads.len(),
            d.settings.heads_per_relation
        );
        assert_eq!(
            rust.tails.len(),
            d.settings.tails_per_relation,
            "{name} rel {r}: Rust kept {} tail slots, not the requested {}",
            rust.tails.len(),
            d.settings.tails_per_relation
        );
        assert_slots(name, r, "heads", &rust.heads, &py.heads);
        assert_slots(name, r, "tails", &rust.tails, &py.tails);
    }
}

fn assert_slots(name: &str, r: usize, which: &str, rust: &[ArgumentSlot], py: &[FixtureSlot]) {
    assert_eq!(
        rust.len(),
        py.len(),
        "{name} rel {r} {which}: slot count (Rust {} vs Python {})",
        rust.len(),
        py.len()
    );
    for (i, (g, w)) in rust.iter().zip(py).enumerate() {
        assert_eq!(
            w.slot, i,
            "{name} rel {r} {which}: fixture slot numbering is not 0..n"
        );
        // Order is part of the claim: `slot` is a rank, never a tolerance.
        assert_eq!(
            g.flat_index, w.flat_index,
            "{name} rel {r} {which}[{i}]: flat_index — Rust {} vs Python {}. A whole-list \
             permutation here with identical final pairs is the silent re-key failure this \
             gate exists for.",
            g.flat_index, w.flat_index
        );
        assert_eq!(
            g.query, w.query,
            "{name} rel {r} {which}[{i}]: query — Rust {} vs Python {}",
            g.query, w.query
        );
        assert_eq!(
            g.cand_slot, w.cand_slot,
            "{name} rel {r} {which}[{i}]: cand_slot — Rust {} vs Python {}",
            g.cand_slot, w.cand_slot
        );
        assert_eq!(
            g.start, w.start,
            "{name} rel {r} {which}[{i}]: raw span start — Rust {} vs Python {}",
            g.start, w.start
        );
        assert_eq!(
            g.end, w.end,
            "{name} rel {r} {which}[{i}]: raw span end — Rust {} vs Python {}",
            g.end, w.end
        );
        assert_eq!(
            g.valid, w.valid,
            "{name} rel {r} {which}[{i}]: valid — Rust {} vs Python {}",
            g.valid, w.valid
        );
        assert!(
            (g.prob - w.prob).abs() <= PROB_TOL,
            "{name} rel {r} {which}[{i}]: prob — Rust {} vs Python {} (tolerance {PROB_TOL:e})",
            g.prob,
            w.prob
        );
    }
}

/// The flat proposal sequence against Python's compacted `RelationPairBatch`,
/// index for index, using only the fields Python's batch actually carries.
///
/// Deliberately kept separate from [`assert_argument_slots`]: on the three
/// ordinary cases this assertion is *blind* to a broken re-key, and that
/// blindness is a documented measurement, not an accident.
fn assert_pairs(name: &str) {
    let d = driver(name);
    let rust = d.flat();
    let py = &d.case.pair_batch;

    assert_eq!(
        py.rows.len(),
        py.num_pairs,
        "{name}: fixture num_pairs disagrees with its own row count"
    );
    assert_eq!(
        rust.len(),
        py.num_pairs,
        "{name}: num_pairs — Rust proposed {} pairs, Python {}",
        rust.len(),
        py.num_pairs
    );

    for (i, (g, w)) in rust.iter().zip(&py.rows).enumerate() {
        assert_eq!(w.index, i, "{name}: fixture row index is not 0..n");
        assert!(
            w.pair_mask,
            "{name} pair {i}: compacted output must carry only valid pairs"
        );
        assert_eq!(
            g.relation_index, w.relation_index,
            "{name} pair {i}: relation_index — Rust {} vs Python {} (relation-major order broke)",
            g.relation_index, w.relation_index
        );
        assert_eq!(
            d.schema[g.relation_index].relation_type, w.relation_type,
            "{name} pair {i}: relation_type"
        );
        assert_eq!(
            (g.head_start, g.head_end),
            (w.head_start, w.head_end),
            "{name} pair {i}: head span — Rust ({}, {}) vs Python ({}, {})",
            g.head_start,
            g.head_end,
            w.head_start,
            w.head_end
        );
        assert_eq!(
            (g.tail_start, g.tail_end),
            (w.tail_start, w.tail_end),
            "{name} pair {i}: tail span — Rust ({}, {}) vs Python ({}, {})",
            g.tail_start,
            g.tail_end,
            w.tail_start,
            w.tail_end
        );
        assert!(
            (g.head_prob - w.head_prob).abs() <= PROB_TOL,
            "{name} pair {i}: head_prob — Rust {} vs Python {}",
            g.head_prob,
            w.head_prob
        );
        assert!(
            (g.tail_prob - w.tail_prob).abs() <= PROB_TOL,
            "{name} pair {i}: tail_prob — Rust {} vs Python {}",
            g.tail_prob,
            w.tail_prob
        );
    }
}

/// `head_slot` / `tail_slot` per pair — the ranks into the argument lists.
///
/// These have no counterpart in Python's `RelationPairBatch`; the fixture
/// carries them precisely so a mismatch localises to *argument selection*
/// instead of only showing up as a different span. Also checks that the
/// per-relation `pairs` of `generate_pairs_detailed` concatenate, in order,
/// into exactly what `generate_pairs` returns.
fn assert_pair_slots(name: &str) {
    let d = driver(name);
    let detailed = d.detailed();
    let flat = d.flat();

    let concatenated: Vec<&ProposedPair> = detailed.iter().flat_map(|p| p.pairs.iter()).collect();
    assert_eq!(
        concatenated.len(),
        flat.len(),
        "{name}: generate_pairs is not the concatenation of generate_pairs_detailed"
    );
    for (i, (a, b)) in concatenated.iter().zip(&flat).enumerate() {
        assert_eq!(
            *a, b,
            "{name} pair {i}: generate_pairs disagrees with generate_pairs_detailed"
        );
    }

    for (r, (rust, py)) in detailed.iter().zip(&d.case.proposals).enumerate() {
        assert_eq!(
            py.pairs.len(),
            py.num_pairs,
            "{name} rel {r}: fixture proposal num_pairs disagrees with its own pair list"
        );
        assert_eq!(
            rust.pairs.len(),
            py.pairs.len(),
            "{name} rel {r}: kept pair count — Rust {} vs Python {}",
            rust.pairs.len(),
            py.pairs.len()
        );
        for (i, (g, w)) in rust.pairs.iter().zip(&py.pairs).enumerate() {
            assert_eq!(
                w.pair_index, i,
                "{name} rel {r}: fixture pair_index is not 0..n"
            );
            assert!(
                w.keep_position < d.settings.pair_cap,
                "{name} rel {r} pair {i}: fixture keep_position {} is outside the pair cap {}",
                w.keep_position,
                d.settings.pair_cap
            );
            assert_eq!(
                g.head_slot, w.head_slot,
                "{name} rel {r} pair {i} (Python keep_position {}): head_slot — Rust {} vs \
                 Python {}. The spans may still agree; the argument selection does not.",
                w.keep_position, g.head_slot, w.head_slot
            );
            assert_eq!(
                g.tail_slot, w.tail_slot,
                "{name} rel {r} pair {i} (Python keep_position {}): tail_slot — Rust {} vs \
                 Python {}",
                w.keep_position, g.tail_slot, w.tail_slot
            );
            assert!(
                (g.head_prob - w.head_prob).abs() <= PROB_TOL,
                "{name} rel {r} pair {i}: head_prob — Rust {} vs Python {}",
                g.head_prob,
                w.head_prob
            );
            assert!(
                (g.tail_prob - w.tail_prob).abs() <= PROB_TOL,
                "{name} rel {r} pair {i}: tail_prob — Rust {} vs Python {}",
                g.tail_prob,
                w.tail_prob
            );
            assert_eq!(
                g.head_query, w.head_query,
                "{name} rel {r} pair {i}: head_query — Rust {} vs Python {}",
                g.head_query, w.head_query
            );
            assert_eq!(
                g.tail_query, w.tail_query,
                "{name} rel {r} pair {i}: tail_query — Rust {} vs Python {}",
                g.tail_query, w.tail_query
            );
            // Cross-check: the slot the pair claims must be the slot that holds
            // the span, so a consistent-but-wrong pairing cannot hide here.
            assert_eq!(
                (rust.heads[g.head_slot].start, rust.heads[g.head_slot].end),
                (w.head_start as i64, w.head_end as i64),
                "{name} rel {r} pair {i}: head_slot {} does not hold the head span Python recorded",
                g.head_slot
            );
            assert_eq!(
                (rust.tails[g.tail_slot].start, rust.tails[g.tail_slot].end),
                (w.tail_start as i64, w.tail_end as i64),
                "{name} rel {r} pair {i}: tail_slot {} does not hold the tail span Python recorded",
                g.tail_slot
            );
        }
    }
    assert_eq!(d.name, name, "driver loaded the wrong case");
}

// ── per-case tests ───────────────────────────────────────────────────────────

macro_rules! case_tests {
    ($slots:ident, $pairs:ident, $pair_slots:ident, $case:literal) => {
        #[test]
        fn $slots() {
            assert_argument_slots($case);
        }
        #[test]
        fn $pairs() {
            assert_pairs($case);
        }
        #[test]
        fn $pair_slots() {
            assert_pair_slots($case);
        }
    };
}

// Q*C = 18 < 32, so `select()` pads and the padded slots carry candidate 0's
// span (`F.pad` fills the *flat index* with 0), not `(0, 0)`.
case_tests!(
    small_pool_padded_l64_argument_slots,
    small_pool_padded_l64_pairs,
    small_pool_padded_l64_pair_slots,
    "small_pool_padded_l64"
);

// take == 32, no padding; 1024 pairs truncated to pair_cap = 64; carries an
// out-of-range query id that must be dropped, never clamped onto query 0.
case_tests!(
    wide_pool_l512_argument_slots,
    wide_pool_l512_pairs,
    wide_pool_l512_pair_slots,
    "wide_pool_l512"
);

// The contrast case: padded length equals num_words.
case_tests!(
    exact_len_l128_argument_slots,
    exact_len_l128_pairs,
    exact_len_l128_pair_slots,
    "exact_len_l128"
);

// LOAD-BEARING. Every logit is 0.0, so every prob is exactly 0.5 and every pair
// score exactly 0.25 in f32 — no tolerance is involved anywhere. Which 32 of
// the 50 candidates and which 64 of the 1024 pairs survive is decided
// *entirely* by `prob DESC -> start ASC -> end ASC -> flat ASC` and then
// `pair prob DESC -> head rank ASC -> tail rank ASC`. This is one of the two
// cases whose *pairs* diverge under a broken port.
case_tests!(
    tied_scores_total_l64_argument_slots,
    tied_scores_total_l64_pairs,
    tied_scores_total_l64_pair_slots,
    "tied_scores_total_l64"
);

// LOAD-BEARING. Three exact logit bands, so ties are bit-exact within a band
// whatever implementation computes the sigmoid, while the bands still order the
// list. The lowest band sits just above the 0.2 argument threshold and must
// stay eligible, which exercises the `>=`.
case_tests!(
    tied_scores_banded_l128_argument_slots,
    tied_scores_banded_l128_pairs,
    tied_scores_banded_l128_pair_slots,
    "tied_scores_banded_l128"
);

// P = 0 via an empty argument pool.
case_tests!(
    empty_below_threshold_l512_argument_slots,
    empty_below_threshold_l512_pairs,
    empty_below_threshold_l512_pair_slots,
    "empty_below_threshold_l512"
);

// P = 0 despite a non-empty pool — see the dedicated route test below.
case_tests!(
    empty_self_span_only_l64_argument_slots,
    empty_self_span_only_l64_pairs,
    empty_self_span_only_l64_pair_slots,
    "empty_self_span_only_l64"
);

// ── the two P=0 routes, asserted as routes ───────────────────────────────────

/// Route 1: nothing clears the 0.2 argument threshold, so the argument pool
/// itself is empty. Every slot comes back invalid and floored — not removed.
#[test]
fn empty_below_threshold_reaches_zero_pairs_through_an_empty_argument_pool() {
    let d = driver("empty_below_threshold_l512");
    let detailed = d.detailed();
    assert!(
        d.flat().is_empty(),
        "empty_below_threshold_l512 must propose no pairs"
    );
    assert!(!detailed.is_empty(), "the case has relation specs");
    for p in &detailed {
        assert_eq!(
            p.heads.iter().filter(|s| s.valid).count(),
            0,
            "rel {}: the argument pool must be empty on this route",
            p.relation_index
        );
        assert_eq!(
            p.tails.iter().filter(|s| s.valid).count(),
            0,
            "rel {}: the argument pool must be empty on this route",
            p.relation_index
        );
        // Floored, not dropped: the slots still exist and still carry real spans.
        assert_eq!(p.heads.len(), d.settings.heads_per_relation);
        assert_eq!(p.tails.len(), d.settings.tails_per_relation);
        assert!(p.pairs.is_empty());
    }
    // And the route really is the threshold, not a mask: the best probability in
    // the whole pool is below it.
    let best = d
        .logits
        .iter()
        .map(|&x| 1.0f32 / (1.0 + (-x).exp()))
        .fold(f32::NEG_INFINITY, f32::max);
    assert!(
        best < d.settings.argument_threshold,
        "this case is supposed to die at the argument threshold: best prob {best} vs threshold {}",
        d.settings.argument_threshold
    );
}

/// Route 2, the one that matters: the pool is **not** empty — one valid head and
/// one valid tail survive the threshold — and the single scoreable pair is a
/// span with itself, killed by `same_span` (`relations.py:216` compares *both*
/// endpoints).
///
/// A Rust early return keyed on "no eligible candidates" passes route 1 and
/// still reaches the scorer session here, which is the ORT crash 7a-2 found.
/// 7b-3's guard must sit **after** the `same_span` filter and after compaction.
#[test]
fn empty_self_span_only_reaches_zero_pairs_through_same_span_not_an_empty_pool() {
    let d = driver("empty_self_span_only_l64");
    let detailed = d.detailed();
    assert!(
        d.flat().is_empty(),
        "empty_self_span_only_l64 must propose no pairs"
    );
    assert_eq!(detailed.len(), 1, "the case has one relation spec");
    let p = &detailed[0];

    let heads: Vec<&ArgumentSlot> = p.heads.iter().filter(|s| s.valid).collect();
    let tails: Vec<&ArgumentSlot> = p.tails.iter().filter(|s| s.valid).collect();
    assert_eq!(
        heads.len(),
        1,
        "the argument pool must NOT be empty on this route — that is the whole point of the case"
    );
    assert_eq!(tails.len(), 1, "exactly one valid tail is expected");
    assert_eq!(
        (heads[0].start, heads[0].end),
        (tails[0].start, tails[0].end),
        "the surviving head and tail must be the same span, so that `same_span` is what \
         produces P = 0 rather than an empty pool"
    );
    assert!(
        p.pairs.is_empty(),
        "a span must never relate to itself while allow_self is false"
    );
    assert!(
        !d.schema[0].allow_self,
        "the fixture spec must have allow_self = false for this case to mean anything"
    );
}

// ── fixture-integrity tests ──────────────────────────────────────────────────

#[test]
fn fixture_is_present_non_empty_and_hash_pinned() {
    let parsed = fixture();
    assert_eq!(parsed.cases.len(), EXPECTED_CASES.len());
}

#[test]
fn fixture_carries_all_seven_cases_in_the_recorded_order() {
    let parsed = fixture();
    let names: Vec<&str> = parsed.cases.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(
        names, EXPECTED_CASES,
        "the seven cases are not the recorded ones — a gate run over fewer cases proves less \
         than it appears to"
    );
}

/// The caps are the ones the checkpoint carries, not `gliner2`'s library
/// defaults. `pair_cap` 64 vs 128 and `argument_threshold` 0.2 vs 0.0 are the
/// two that differ, and both change the output without failing anything else.
#[test]
fn fixture_settings_are_the_checkpoint_values() {
    let parsed = fixture();
    let from_fixture = RelationProposalSettings {
        heads_per_relation: parsed.settings.heads_per_relation,
        tails_per_relation: parsed.settings.tails_per_relation,
        pair_cap: parsed.settings.pair_cap,
        argument_threshold: parsed.settings.argument_threshold,
    };
    assert_eq!(
        from_fixture,
        RelationProposalSettings::checkpoint(),
        "Python recorded the fixture under different caps than the Rust port uses"
    );
}

/// The tied cases have to actually tie, or the two cases that make the *pairs*
/// diverge under a broken port stop doing so and the gate quietly weakens.
#[test]
fn the_tied_cases_really_tie() {
    for name in ["tied_scores_total_l64", "tied_scores_banded_l128"] {
        let d = driver(name);
        let detailed = d.detailed();
        let mut max_run = 0usize;
        for p in &detailed {
            let mut scores: Vec<f32> = p
                .pairs
                .iter()
                .map(|pair| pair.head_prob * pair.tail_prob)
                .collect();
            scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let mut run = 0usize;
            let mut i = 0usize;
            while i < scores.len() {
                let mut j = i;
                // Bit-exact equality on purpose: these ties are constructed, not
                // approximate, so no tolerance belongs here.
                while j < scores.len() && scores[j].to_bits() == scores[i].to_bits() {
                    j += 1;
                }
                run = run.max(j - i);
                i = j;
            }
            max_run = max_run.max(run);
        }
        assert!(
            max_run >= 2,
            "{name}: maximum pair-score multiplicity is {max_run}; this case is supposed to be \
             decided by the tie-break chain"
        );
    }
}

// ── a dependency-free sha256, so the fixture can be pinned in-process ────────

/// Self-test of [`sha256_hex`] against the NIST FIPS 180-4 vectors, so a bug in
/// the hasher below cannot turn the fixture pin into a rubber stamp.
#[test]
fn sha256_helper_matches_the_known_vectors() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// Plain FIPS 180-4 SHA-256. Vendored rather than pulled in as a dev-dependency:
/// the fork is kept byte-close to upstream, and a new crate in the dependency
/// graph is a bigger change than forty lines of well-known arithmetic that this
/// file self-tests above.
fn sha256_hex(data: &[u8]) -> String {
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (slot, word) in w.iter_mut().zip(block.chunks_exact(4)) {
            *slot = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let a = w[i - 15];
            let b = w[i - 2];
            let s0 = a.rotate_right(7) ^ a.rotate_right(18) ^ (a >> 3);
            let s1 = b.rotate_right(17) ^ b.rotate_right(19) ^ (b >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for (k, wi) in SHA256_K.iter().zip(w.iter()) {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(*k)
                .wrapping_add(*wi);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (acc, add) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *acc = acc.wrapping_add(add);
        }
    }

    let mut out = String::with_capacity(64);
    for word in h {
        out.push_str(&format!("{word:08x}"));
    }
    out
}
