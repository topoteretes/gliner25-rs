// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! Typed, capped relation-pair **proposal**, ported from `gliner2`'s
//! `TypedRelationPairGenerator` (`gliner2/models/boundary/relations.py:93`).
//!
//! This module is weightless: it is index arithmetic over the boundary head's
//! *raw* outputs (`pair_logits`, `cand_indices`, `cand_valid`), and it produces
//! the `(head, tail)` pairs that the relation scorer will later score. Nothing
//! here runs a model, and nothing here decodes — see the sub-stage split in
//! `docs`: scoring and decoding are separate work.
//!
//! ## Why this is ported rather than exported
//!
//! The generator is pure tensor code in Python, so exporting it with the rest
//! of the graph would be the obvious move. It cannot be done: the selection
//! depends on `torch.argsort(..., stable=True)`, and `torch.export` lowers
//! `argsort` through `_unstable_sort`, which drops the stability guarantee. A
//! different tie-break picks a different top-32 and a different top-64, and
//! **nothing downstream notices** — the pipeline still runs and the numbers are
//! merely worse. Rust's [`slice::sort_by`] / [`slice::sort_by_key`] *are*
//! guaranteed stable, so the port is the faithful implementation and the export
//! is not.
//!
//! For that reason this module never uses `sort_unstable_*`. Do not "optimise"
//! a sort here.
//!
//! ## Inputs are the raw pool, before any decoding
//!
//! [`CandidateView`] must be built from the boundary head's outputs *before*
//! abstention, *before* the decode threshold and *before* overlap resolution.
//! Relation endpoints enter the pool at
//! [`RelationProposalSettings::argument_threshold`] (0.2 for this checkpoint),
//! which is a **separate knob** from the decode threshold (0.5) applied to the
//! relation scorer's output. Both coexist; feeding this module the already
//! decoded `BoundaryOutput::mentions` would apply the decode threshold twice
//! and silently lose the low-probability endpoints relations depend on.

use std::cmp::Ordering;

use crate::runtime::sigmoid;

/// The proposal caps and the argument threshold.
///
/// There is deliberately **no `Default`**. The `gliner2` library's defaults
/// (`relations.py:25-29`) disagree with the `fastino/gliner2.5-base-v1`
/// checkpoint on two of the four fields, and picking the wrong ones changes the
/// output without failing anything. Use [`RelationProposalSettings::checkpoint`]
/// or, better, [`RelationProposalSettings::from_relation_settings_json`] against
/// the `relation_settings.json` the exporter writes next to the model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelationProposalSettings {
    /// `Rh` — head slots kept per relation type. Checkpoint: 32.
    pub heads_per_relation: usize,
    /// `Rt` — tail slots kept per relation type. Checkpoint: 32.
    pub tails_per_relation: usize,
    /// `pair_cap` — pairs kept per relation type. Checkpoint: **64**; the
    /// library default is 128.
    pub pair_cap: usize,
    /// Minimum `sigmoid(pair_logit)` for a candidate to be *eligible as an
    /// endpoint*. Checkpoint: **0.2**; the library default is 0.0. This is not
    /// the relation decode threshold.
    pub argument_threshold: f32,
}

impl RelationProposalSettings {
    /// The values carried by `fastino/gliner2.5-base-v1`, as written to
    /// `models/<export>/relation_settings.json` by the exporter.
    ///
    /// Two of them are **not** the library defaults, and the difference is not
    /// cosmetic: `pair_cap` 64 vs 128 halves the proposal set, and
    /// `argument_threshold` 0.2 vs 0.0 is what keeps the pool from degenerating
    /// into the full cross product.
    pub const fn checkpoint() -> Self {
        Self {
            heads_per_relation: 32,
            tails_per_relation: 32,
            pair_cap: 64,
            argument_threshold: 0.2,
        }
    }

    /// Reads the four fields out of an exporter-written `relation_settings.json`.
    ///
    /// Prefer this over [`Self::checkpoint`] wherever the model directory is at
    /// hand: it keeps the constants in one place — the export — instead of two.
    /// A missing key is an error rather than a silent fallback, because a silent
    /// fallback is how the library defaults get back in.
    pub fn from_relation_settings_json(json: &str) -> anyhow::Result<Self> {
        let v: serde_json::Value = serde_json::from_str(json)?;
        let usize_at = |key: &str| -> anyhow::Result<usize> {
            v.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| anyhow::anyhow!("relation_settings.json: missing `{key}`"))
        };
        let f32_at = |key: &str| -> anyhow::Result<f32> {
            v.get(key)
                .and_then(serde_json::Value::as_f64)
                .map(|n| n as f32)
                .ok_or_else(|| anyhow::anyhow!("relation_settings.json: missing `{key}`"))
        };
        Ok(Self {
            heads_per_relation: usize_at("relation_heads_per_type")?,
            tails_per_relation: usize_at("relation_tails_per_type")?,
            pair_cap: usize_at("relation_pair_cap")?,
            argument_threshold: f32_at("relation_argument_proposal_threshold")?,
        })
    }
}

/// One relation type and the entity queries allowed at each end.
///
/// `gliner2`'s `model.py` builds exactly one of these per relation group, with a
/// single head query id and a single tail query id. The multi-query form is
/// implemented here anyway because it costs nothing — but note that a passing
/// proposal-identity test against this checkpoint's fixtures does **not** cover
/// multi-query routing, because the fixtures never exercise it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelationTypeSpec {
    /// The relation group's name, carried through to the decoded edge.
    pub relation_type: String,
    /// Query ids eligible as the head. Ids outside `0..queries` are dropped,
    /// as in `relations.py:143`.
    pub head_query_ids: Vec<usize>,
    /// Query ids eligible as the tail.
    pub tail_query_ids: Vec<usize>,
    /// Whether a mention may relate to itself. `false` for every spec `gliner2`
    /// builds (`relations.py:39`, and `model.py` never overrides it).
    pub allow_self: bool,
}

impl RelationTypeSpec {
    /// A spec with `allow_self = false`, which is what the engine always wants.
    pub fn new(
        relation_type: impl Into<String>,
        head_query_ids: Vec<usize>,
        tail_query_ids: Vec<usize>,
    ) -> Self {
        Self {
            relation_type: relation_type.into(),
            head_query_ids,
            tail_query_ids,
            allow_self: false,
        }
    }
}

/// A borrowed view of the boundary head's raw candidate outputs.
///
/// Flat candidate index is `j = query * cand_count + slot`, matching
/// `relations.py`'s `reshape(bsz, queries * cand_count, …)`. Batch size is
/// fixed at one: this engine runs a single window per call.
#[derive(Debug, Clone, Copy)]
pub struct CandidateView<'a> {
    /// `[Q*C*2]` half-open `(start, end)` word spans, `i64` as the head emits
    /// them — the ordering keys below are taken from these raw values, so they
    /// are deliberately not narrowed to `usize` here.
    pub indices: &'a [i64],
    /// `[Q*C]` raw logits. Sigmoid is applied here, not by the caller.
    pub pair_logits: &'a [f32],
    /// `[Q*C]` candidate validity as the head reports it.
    pub valid_mask: &'a [bool],
    /// `[Q]` query validity.
    pub query_mask: &'a [bool],
    pub queries: usize,
    pub cand_count: usize,
}

impl CandidateView<'_> {
    fn flat_len(&self) -> usize {
        self.queries * self.cand_count
    }

    fn start_of(&self, j: usize) -> i64 {
        self.indices[j * 2]
    }

    fn end_of(&self, j: usize) -> i64 {
        self.indices[j * 2 + 1]
    }
}

/// One selected head-or-tail slot, in `select()` order.
///
/// Exposed so a proposal-identity test can compare the *argument* selection
/// index-for-index against a Python fixture, not only the final pairs. Exactly
/// `requested` of these come back per relation type; the ones past the real
/// pool are zero-padded and carry `valid = false`, as in `relations.py:199-203`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArgumentSlot {
    /// `j = query * cand_count + cand_slot`.
    pub flat_index: usize,
    pub query: usize,
    pub cand_slot: usize,
    /// Raw span, half-open, exactly as the head emitted it.
    pub start: i64,
    pub end: i64,
    pub prob: f32,
    pub valid: bool,
}

/// One proposed `(head, tail)` pair, before any scoring.
///
/// The sequence returned by [`generate_pairs`] is the whole contract: it is
/// ordered, and 7b-2 asserts it index-for-index against Python's compacted
/// `RelationPairBatch`.
#[derive(Debug, Clone, PartialEq)]
pub struct ProposedPair {
    /// Index into the `schema` slice handed to [`generate_pairs`].
    pub relation_index: usize,
    pub head_start: usize,
    pub head_end: usize,
    pub tail_start: usize,
    pub tail_end: usize,
    pub head_prob: f32,
    pub tail_prob: f32,
    pub head_query: usize,
    pub tail_query: usize,
    /// Rank of this head within the relation's selected head slots, and of the
    /// tail within its tail slots. Not part of Python's `RelationPairBatch` —
    /// carried so a failing proposal-identity diff can say *which* selection
    /// step drifted instead of only that the spans differ.
    pub head_slot: usize,
    pub tail_slot: usize,
}

/// The whole proposal for one relation type, kept inspectable.
///
/// [`generate_pairs`] flattens these; [`generate_pairs_detailed`] does not, so a
/// test can reach the intermediate top-`Rh` / top-`Rt` slot lists.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationProposal {
    pub relation_index: usize,
    pub relation_type: String,
    /// Exactly `settings.heads_per_relation` slots, padded with invalid ones.
    pub heads: Vec<ArgumentSlot>,
    /// Exactly `settings.tails_per_relation` slots, padded with invalid ones.
    pub tails: Vec<ArgumentSlot>,
    /// The kept pairs, in `keep` order, already filtered to the valid ones
    /// (`compact = True` in `relations.py:249-252`).
    pub pairs: Vec<ProposedPair>,
}

/// The `floor` of `relations.py:168` — `torch.finfo(dtype).min`, used to push
/// invalid entries to the bottom of a *descending* sort without removing them.
/// They still occupy slots; only their `valid` flag says otherwise.
const FLOOR: f32 = f32::MIN;

/// Generates typed, capped relation-pair proposals.
///
/// Port of `TypedRelationPairGenerator.generate_batched(..., compact=True)`
/// (`relations.py:91-290`) for a batch of one. The returned sequence is
/// relation-major, and within a relation it is `keep` order.
pub fn generate_pairs(
    candidates: &CandidateView<'_>,
    schema: &[RelationTypeSpec],
    settings: &RelationProposalSettings,
) -> Vec<ProposedPair> {
    generate_pairs_detailed(candidates, schema, settings)
        .into_iter()
        .flat_map(|proposal| proposal.pairs)
        .collect()
}

/// As [`generate_pairs`], but keeps the per-relation argument selection.
pub fn generate_pairs_detailed(
    candidates: &CandidateView<'_>,
    schema: &[RelationTypeSpec],
    settings: &RelationProposalSettings,
) -> Vec<RelationProposal> {
    let flat_len = candidates.flat_len();
    // `rel_count == 0` short-circuits in `relations.py:110-121`; an empty pool
    // is not a case Python has to handle (a batch always has candidates) but it
    // would index out of bounds here.
    if schema.is_empty() || flat_len == 0 {
        return Vec::new();
    }

    // `probs = torch.sigmoid(candidates.pair_logits)` — `relations.py:152`.
    let probs: Vec<f32> = candidates
        .pair_logits
        .iter()
        .copied()
        .map(sigmoid)
        .collect();

    // `base_valid = valid_mask & query_mask.unsqueeze(-1)` — `relations.py:153`.
    // `threshold = flat_prob >= s.argument_threshold` — `relations.py:165`, a
    // `>=`, not a `>`.
    let eligible: Vec<bool> = (0..flat_len)
        .map(|j| {
            candidates.valid_mask[j]
                && candidates.query_mask[j / candidates.cand_count]
                && probs[j] >= settings.argument_threshold
        })
        .collect();

    // ── the positional key, `secondary` ──────────────────────────────────
    // Python builds it with two stable argsorts over the identity permutation:
    //   secondary = argsort(end,   stable=True)   relations.py:179-182
    //   secondary = argsort(start, stable=True)   relations.py:186-189
    // Stability composes, so the result is exactly `0..Q*C` ordered by
    // (start ASC, end ASC, flat index ASC). That is one stable sort here, with
    // the composed key written out so the equivalence can be checked by eye.
    // `sort_by_key` is stable; `sort_unstable_by_key` would not be, and would
    // change which candidates survive the score sort below.
    let mut secondary: Vec<usize> = (0..flat_len).collect();
    secondary.sort_by_key(|&j| (candidates.start_of(j), candidates.end_of(j), j as i64));

    let mut out = Vec::with_capacity(schema.len());
    for (relation_index, spec) in schema.iter().enumerate() {
        // `head_member` / `tail_member`, `relations.py:141-149`. Query ids
        // outside the layout are dropped rather than clamped.
        let mut head_member = vec![false; candidates.queries];
        let mut tail_member = vec![false; candidates.queries];
        for &q in &spec.head_query_ids {
            if q < candidates.queries {
                head_member[q] = true;
            }
        }
        for &q in &spec.tail_query_ids {
            if q < candidates.queries {
                tail_member[q] = true;
            }
        }

        let head_valid: Vec<bool> = (0..flat_len)
            .map(|j| eligible[j] && head_member[j / candidates.cand_count])
            .collect();
        let tail_valid: Vec<bool> = (0..flat_len)
            .map(|j| eligible[j] && tail_member[j / candidates.cand_count])
            .collect();

        let heads = select(
            candidates,
            &secondary,
            &probs,
            &head_valid,
            settings.heads_per_relation,
        );
        let tails = select(
            candidates,
            &secondary,
            &probs,
            &tail_valid,
            settings.tails_per_relation,
        );
        let pairs = pair(&heads, &tails, relation_index, spec.allow_self, settings);

        out.push(RelationProposal {
            relation_index,
            relation_type: spec.relation_type.clone(),
            heads,
            tails,
            pairs,
        });
    }
    out
}

/// Port of the inner `select()` of `relations.py:171-210`.
///
/// `order` is `secondary` — the flat indices already in
/// `(start ASC, end ASC, flat ASC)` order. The score sort runs over **positions
/// in `order`**, not over flat indices, which is the whole point: a stable
/// descending sort then breaks ties by span position rather than by `j`. A port
/// that sorts a flat-index-ordered vector by score picks a different top-`Rh`
/// and fails silently.
///
/// Always returns exactly `requested` slots. Slots past `take = min(requested,
/// Q*C)` are the zero-pad of `relations.py:199-203`: flat index 0, probability
/// 0, `valid = false`. They are kept because the pair arithmetic below divides
/// by `tails_per_relation`, not by `take`.
fn select(
    candidates: &CandidateView<'_>,
    order: &[usize],
    probs: &[f32],
    valid: &[bool],
    requested: usize,
) -> Vec<ArgumentSlot> {
    let n = order.len();
    let take = requested.min(n);

    // `rank_in_secondary = argsort(score.masked_fill(~valid, floor),
    //  descending=True, stable=True)` — relations.py:191-195.
    let key = |position: usize| -> f32 {
        let j = order[position];
        if valid[j] { probs[j] } else { FLOOR }
    };
    let mut rank: Vec<usize> = (0..n).collect();
    // `sort_by`, never `sort_unstable_by`: torch's `stable=True` keeps equal
    // scores in `secondary` order, and so must this.
    rank.sort_by(|&a, &b| key(b).partial_cmp(&key(a)).unwrap_or(Ordering::Equal));

    let mut slots: Vec<ArgumentSlot> = Vec::with_capacity(requested);
    for &position in &rank[..take] {
        let j = order[position];
        slots.push(slot_at(candidates, j, probs[j], valid[j]));
    }
    // `F.pad(ranked, (0, pad))` pads the *flat index* with 0, so the padded
    // slots carry candidate 0's span — not `(0, 0)`. It never matters, because
    // `valid = false` kills every pair they take part in, but reproducing it
    // keeps a fixture comparison on `heads` / `tails` honest.
    while slots.len() < requested {
        slots.push(slot_at(candidates, 0, 0.0, false));
    }
    slots
}

fn slot_at(
    candidates: &CandidateView<'_>,
    flat_index: usize,
    prob: f32,
    valid: bool,
) -> ArgumentSlot {
    // `qslot = ranked // cand_count`, `cslot = ranked - qslot * cand_count`,
    // both clamped — relations.py:205-208. The clamps cannot bite for a flat
    // index inside the pool; they are Python guarding its own padding.
    let query = (flat_index / candidates.cand_count).min(candidates.queries.saturating_sub(1));
    let cand_slot =
        (flat_index % candidates.cand_count).min(candidates.cand_count.saturating_sub(1));
    ArgumentSlot {
        flat_index,
        query,
        cand_slot,
        start: candidates.start_of(flat_index),
        end: candidates.end_of(flat_index),
        prob,
        valid,
    }
}

/// Port of the pair stage, `relations.py:212-247`.
///
/// `pair_score[h][t] = head_prob[h] * tail_prob[t]`, flattened row-major so the
/// flat pair index is `h * Rt + t` with `Rt = settings.tails_per_relation` (the
/// *requested* count, which is why `select` pads). The keep sort is again
/// stable and descending, so ties break by `h` then `t` — and those are already
/// in `select()` order, making the full tie chain
/// `pair prob DESC → head rank → tail rank → (start, end, flat index)`.
fn pair(
    heads: &[ArgumentSlot],
    tails: &[ArgumentSlot],
    relation_index: usize,
    allow_self: bool,
    settings: &RelationProposalSettings,
) -> Vec<ProposedPair> {
    let rt = settings.tails_per_relation;
    let total = heads.len() * rt;
    let mut score = vec![0.0f32; total];
    let mut valid = vec![false; total];
    for (h, head) in heads.iter().enumerate() {
        for (t, tail) in tails.iter().enumerate() {
            let flat = h * rt + t;
            score[flat] = head.prob * tail.prob;
            // `same_span` compares **both** endpoints (`.all(-1)`,
            // relations.py:216), not just the start.
            let same_span = head.start == tail.start && head.end == tail.end;
            valid[flat] = head.valid && tail.valid && (allow_self || !same_span);
        }
    }

    let key = |flat: usize| -> f32 { if valid[flat] { score[flat] } else { FLOOR } };
    let mut keep: Vec<usize> = (0..total).collect();
    keep.sort_by(|&a, &b| key(b).partial_cmp(&key(a)).unwrap_or(Ordering::Equal));
    let take = settings.pair_cap.min(total);

    // `compact = True` drops the invalid survivors instead of carrying a mask.
    let mut pairs = Vec::new();
    for &flat in &keep[..take] {
        if !valid[flat] {
            continue;
        }
        let head = &heads[flat / rt];
        let tail = &tails[flat % rt];
        pairs.push(ProposedPair {
            relation_index,
            // Only valid slots reach here, and a valid candidate's span comes
            // from the head as a non-negative half-open word range — the same
            // assumption `boundary.rs` decoding already makes.
            head_start: head.start as usize,
            head_end: head.end as usize,
            tail_start: tail.start as usize,
            tail_end: tail.end as usize,
            head_prob: head.prob,
            tail_prob: tail.prob,
            head_query: head.query,
            tail_query: tail.query,
            head_slot: flat / rt,
            tail_slot: flat % rt,
        });
    }
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a view over `spans` (half-open `(start, end)` per candidate) and
    /// `probs`, with one query per `queries` split. `logit` inverts sigmoid so a
    /// test can state the probability it means.
    struct Pool {
        indices: Vec<i64>,
        logits: Vec<f32>,
        valid: Vec<bool>,
        query_mask: Vec<bool>,
        queries: usize,
        cand_count: usize,
    }

    impl Pool {
        /// `spans[q][c] = (start, end, prob)`.
        fn new(spans: &[Vec<(i64, i64, f32)>]) -> Self {
            let queries = spans.len();
            let cand_count = spans[0].len();
            let mut indices = Vec::new();
            let mut logits = Vec::new();
            for row in spans {
                assert_eq!(row.len(), cand_count, "ragged pool");
                for &(s, e, p) in row {
                    indices.push(s);
                    indices.push(e);
                    logits.push(logit(p));
                }
            }
            Self {
                indices,
                logits,
                valid: vec![true; queries * cand_count],
                query_mask: vec![true; queries],
                queries,
                cand_count,
            }
        }

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

    fn logit(p: f32) -> f32 {
        (p / (1.0 - p)).ln()
    }

    fn settings(rh: usize, rt: usize, cap: usize, threshold: f32) -> RelationProposalSettings {
        RelationProposalSettings {
            heads_per_relation: rh,
            tails_per_relation: rt,
            pair_cap: cap,
            argument_threshold: threshold,
        }
    }

    fn spec(heads: Vec<usize>, tails: Vec<usize>) -> Vec<RelationTypeSpec> {
        vec![RelationTypeSpec::new("works_at", heads, tails)]
    }

    #[test]
    fn checkpoint_values_are_not_the_library_defaults() {
        let s = RelationProposalSettings::checkpoint();
        assert_eq!(
            s.pair_cap, 64,
            "the library default 128 is wrong for this checkpoint"
        );
        assert_eq!(
            s.argument_threshold, 0.2,
            "the library default 0.0 is wrong"
        );
        assert_eq!(s.heads_per_relation, 32);
        assert_eq!(s.tails_per_relation, 32);
    }

    #[test]
    fn settings_load_from_the_exporter_json() {
        let json = r#"{
            "relation_heads_per_type": 32,
            "relation_tails_per_type": 32,
            "relation_pair_cap": 64,
            "relation_argument_proposal_threshold": 0.2
        }"#;
        let s = RelationProposalSettings::from_relation_settings_json(json).expect("parses");
        assert_eq!(s, RelationProposalSettings::checkpoint());
        assert!(RelationProposalSettings::from_relation_settings_json("{}").is_err());
    }

    // ── the tie-break tests ──────────────────────────────────────────────
    //
    // Every one of these is built so that a score-only sort over flat-index
    // order gives a *different* answer. They are the reason this module exists.

    /// End-to-end: the `(start, end, flat)` re-key lives inside
    /// `generate_pairs_detailed`, not only inside `select`. Every head here has
    /// the same probability, so only the re-key decides which two survive.
    /// Deleting it makes this say `[(9, 10), (3, 4)]`.
    #[test]
    fn the_pool_is_re_keyed_by_span_before_the_score_sort() {
        let pool = Pool::new(&[
            vec![(9, 10, 0.6), (3, 4, 0.6), (7, 8, 0.6), (1, 2, 0.6)],
            vec![(0, 1, 0.6), (0, 1, 0.6), (0, 1, 0.6), (0, 1, 0.6)],
        ]);
        let out = generate_pairs_detailed(
            &pool.view(),
            &spec(vec![0], vec![1]),
            &settings(2, 1, 8, 0.0),
        );
        let heads: Vec<(i64, i64)> = out[0].heads.iter().map(|s| (s.start, s.end)).collect();
        assert_eq!(
            heads,
            vec![(1, 2), (3, 4)],
            "the two left-most spans win the tie"
        );
    }

    /// End-to-end companion for the second key: equal starts order by `end ASC`.
    /// Deleting the re-key makes this say `[(5, 9), (5, 6)]`.
    #[test]
    fn equal_starts_are_ordered_by_span_end_end_to_end() {
        let pool = Pool::new(&[
            vec![(5, 9, 0.6), (5, 6, 0.6), (5, 7, 0.6)],
            vec![(0, 1, 0.6), (0, 1, 0.6), (0, 1, 0.6)],
        ]);
        let out = generate_pairs_detailed(
            &pool.view(),
            &spec(vec![0], vec![1]),
            &settings(2, 1, 8, 0.0),
        );
        let heads: Vec<(i64, i64)> = out[0].heads.iter().map(|s| (s.start, s.end)).collect();
        assert_eq!(
            heads,
            vec![(5, 6), (5, 7)],
            "shortest span at the same start first"
        );
    }

    /// All four candidates share a probability. Python re-keys by
    /// `(start, end, flat)` before the score sort, so the survivors are the two
    /// left-most spans — not the two lowest flat indices.
    #[test]
    fn equal_scores_break_by_span_start_not_by_flat_index() {
        // flat order:            j=0        j=1        j=2        j=3
        // spans, deliberately not sorted by position:
        let pool = Pool::new(&[vec![(9, 10, 0.8), (3, 4, 0.8), (7, 8, 0.8), (1, 2, 0.8)]]);
        let view = pool.view();
        let all_valid = [true; 4];
        let mut secondary: Vec<usize> = (0..4).collect();
        secondary.sort_by_key(|&j| (view.start_of(j), view.end_of(j), j as i64));
        assert_eq!(
            secondary,
            vec![3, 1, 2, 0],
            "secondary is (start, end, j) order"
        );

        let slots = select(&view, &secondary, &[0.8, 0.8, 0.8, 0.8], &all_valid, 2);
        assert_eq!(
            slots.iter().map(|s| s.flat_index).collect::<Vec<_>>(),
            vec![3, 1],
            "ties must break by span position; a flat-index port would say [0, 1]"
        );
    }

    /// Same start, different ends: the second key decides, and it is `end ASC`.
    #[test]
    fn equal_scores_and_equal_starts_break_by_span_end() {
        let pool = Pool::new(&[vec![(5, 9, 0.5), (5, 6, 0.5), (5, 7, 0.5)]]);
        let view = pool.view();
        let mut secondary: Vec<usize> = (0..3).collect();
        secondary.sort_by_key(|&j| (view.start_of(j), view.end_of(j), j as i64));
        assert_eq!(secondary, vec![1, 2, 0]);

        let slots = select(&view, &secondary, &[0.5, 0.5, 0.5], &[true; 3], 2);
        assert_eq!(
            slots.iter().map(|s| (s.start, s.end)).collect::<Vec<_>>(),
            vec![(5, 6), (5, 7)],
            "shortest span at the same start wins the tie"
        );
    }

    /// Identical spans are the only case where the flat index decides, and it
    /// decides ascending.
    #[test]
    fn identical_spans_break_by_flat_index_ascending() {
        let pool = Pool::new(&[vec![(2, 3, 0.4), (2, 3, 0.4), (2, 3, 0.4)]]);
        let view = pool.view();
        let mut secondary: Vec<usize> = (0..3).collect();
        secondary.sort_by_key(|&j| (view.start_of(j), view.end_of(j), j as i64));
        let slots = select(&view, &secondary, &[0.4, 0.4, 0.4], &[true; 3], 2);
        assert_eq!(
            slots.iter().map(|s| s.flat_index).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    /// The score sort still dominates: position only breaks ties.
    #[test]
    fn a_higher_score_beats_an_earlier_span() {
        let pool = Pool::new(&[vec![(1, 2, 0.3), (8, 9, 0.9)]]);
        let view = pool.view();
        let mut secondary: Vec<usize> = (0..2).collect();
        secondary.sort_by_key(|&j| (view.start_of(j), view.end_of(j), j as i64));
        let slots = select(&view, &secondary, &[0.3, 0.9], &[true; 2], 1);
        assert_eq!(slots[0].flat_index, 1);
    }

    /// Invalid entries are pushed to the floor but still occupy slots, and the
    /// result is always exactly `requested` long.
    #[test]
    fn invalid_entries_are_floored_not_removed_and_slots_are_padded() {
        let pool = Pool::new(&[vec![(1, 2, 0.9), (3, 4, 0.9)]]);
        let view = pool.view();
        let secondary = vec![0usize, 1];
        let slots = select(&view, &secondary, &[0.9, 0.9], &[false, true], 4);
        assert_eq!(slots.len(), 4, "always `requested` slots");
        assert_eq!(slots[0].flat_index, 1, "the only valid entry ranks first");
        assert!(slots[0].valid);
        assert_eq!(slots[1].flat_index, 0, "the invalid one keeps a slot");
        assert!(!slots[1].valid);
        // the zero-pad carries candidate 0's span, not `(0, 0)`
        assert_eq!((slots[2].start, slots[2].end), (1, 2));
        assert!(!slots[2].valid && !slots[3].valid);
        assert_eq!(slots[2].prob, 0.0);
    }

    /// The pair keep-sort breaks ties by head rank then tail rank, and those
    /// ranks are `select()`'s, so a whole-chain regression shows up here.
    #[test]
    fn equal_pair_scores_break_by_head_rank_then_tail_rank() {
        // Two heads and two tails, all at 0.5, so all four pair scores are 0.25.
        // Head query 0 (flat 0, 1), tail query 1 (flat 2, 3).
        // Spans are laid out so `select` order is the reverse of flat order.
        let pool = Pool::new(&[
            vec![(7, 8, 0.5), (3, 4, 0.5)],  // heads: select order = [1, 0]
            vec![(9, 10, 0.5), (5, 6, 0.5)], // tails: select order = [3, 2]
        ]);
        let out = generate_pairs_detailed(
            &pool.view(),
            &spec(vec![0], vec![1]),
            &settings(2, 2, 4, 0.0),
        );
        let heads: Vec<usize> = out[0].heads.iter().map(|s| s.flat_index).collect();
        let tails: Vec<usize> = out[0].tails.iter().map(|s| s.flat_index).collect();
        assert_eq!(heads, vec![1, 0]);
        assert_eq!(tails, vec![3, 2]);
        let spans: Vec<(usize, usize, usize, usize)> = out[0]
            .pairs
            .iter()
            .map(|p| (p.head_start, p.head_end, p.tail_start, p.tail_end))
            .collect();
        assert_eq!(
            spans,
            vec![
                (3, 4, 5, 6),  // head rank 0, tail rank 0
                (3, 4, 9, 10), // head rank 0, tail rank 1
                (7, 8, 5, 6),  // head rank 1, tail rank 0
                (7, 8, 9, 10), // head rank 1, tail rank 1
            ],
            "equal pair scores keep (head rank, tail rank) order"
        );
    }

    /// `pair_cap` truncates *after* the keep sort, so a cap of 1 keeps the
    /// tie-break winner rather than the first flat pair.
    #[test]
    fn pair_cap_truncates_after_the_keep_sort() {
        let pool = Pool::new(&[
            vec![(7, 8, 0.5), (3, 4, 0.5)],
            vec![(9, 10, 0.5), (5, 6, 0.5)],
        ]);
        let pairs = generate_pairs(
            &pool.view(),
            &spec(vec![0], vec![1]),
            &settings(2, 2, 1, 0.0),
        );
        assert_eq!(pairs.len(), 1);
        assert_eq!((pairs[0].head_start, pairs[0].tail_start), (3, 5));
    }

    /// The argument threshold is `>=`, it is applied to the *raw pool*, and it
    /// is 0.2 — an endpoint at 0.3 must survive, which is the whole point of
    /// not reusing the 0.5 decode threshold.
    #[test]
    fn argument_threshold_admits_endpoints_below_the_decode_threshold() {
        let pool = Pool::new(&[
            vec![(1, 2, 0.30), (3, 4, 0.10)],
            vec![(5, 6, 0.90), (7, 8, 0.90)],
        ]);
        let s = RelationProposalSettings::checkpoint();
        let pairs = generate_pairs(&pool.view(), &spec(vec![0], vec![1]), &s);
        let heads: Vec<(usize, usize)> = pairs.iter().map(|p| (p.head_start, p.head_end)).collect();
        assert!(
            heads.contains(&(1, 2)),
            "a 0.30 head must be proposed at threshold 0.2"
        );
        assert!(!heads.contains(&(3, 4)), "0.10 is below 0.2");
    }

    /// The comparison is `flat_prob >= threshold` (`relations.py:165`), not
    /// `>`. `logit(0.5) == 0.0` and `sigmoid(0.0) == 0.5` are both exact in
    /// `f32`, so this pins the boundary without a tolerance.
    #[test]
    fn the_argument_threshold_is_inclusive() {
        let pool = Pool::new(&[vec![(1, 2, 0.5)], vec![(5, 6, 0.9)]]);
        let schema = spec(vec![0], vec![1]);
        let at_threshold = generate_pairs(&pool.view(), &schema, &settings(4, 4, 8, 0.5));
        assert_eq!(
            at_threshold.len(),
            1,
            "a probability equal to the threshold is admitted"
        );
        let above = generate_pairs(&pool.view(), &schema, &settings(4, 4, 8, 0.5000001));
        assert!(above.is_empty());
    }

    /// `allow_self` is false, and the self-check compares both endpoints.
    #[test]
    fn a_span_never_relates_to_itself_unless_allow_self() {
        // One query serving as both head and tail: the (1,2)/(1,2) pair is the
        // same span and must be dropped; (1,2)/(1,5) shares a start and must not.
        let pool = Pool::new(&[vec![(1, 2, 0.9), (1, 5, 0.8)]]);
        let pairs = generate_pairs(
            &pool.view(),
            &spec(vec![0], vec![0]),
            &settings(2, 2, 8, 0.0),
        );
        let seen: Vec<(usize, usize, usize, usize)> = pairs
            .iter()
            .map(|p| (p.head_start, p.head_end, p.tail_start, p.tail_end))
            .collect();
        assert_eq!(seen, vec![(1, 2, 1, 5), (1, 5, 1, 2)]);

        let mut allowing = spec(vec![0], vec![0]);
        allowing[0].allow_self = true;
        let with_self = generate_pairs(&pool.view(), &allowing, &settings(2, 2, 8, 0.0));
        assert_eq!(with_self.len(), 4);
    }

    /// Ineligible queries contribute nothing, and an out-of-range query id is
    /// dropped rather than clamped onto query 0.
    #[test]
    fn routing_respects_query_membership_and_drops_out_of_range_ids() {
        let pool = Pool::new(&[vec![(1, 2, 0.9)], vec![(3, 4, 0.9)], vec![(5, 6, 0.9)]]);
        let pairs = generate_pairs(
            &pool.view(),
            &spec(vec![0], vec![2]),
            &settings(4, 4, 8, 0.0),
        );
        assert_eq!(pairs.len(), 1);
        assert_eq!((pairs[0].head_start, pairs[0].tail_start), (1, 5));
        assert_eq!((pairs[0].head_query, pairs[0].tail_query), (0, 2));

        let none = generate_pairs(
            &pool.view(),
            &spec(vec![7], vec![2]),
            &settings(4, 4, 8, 0.0),
        );
        assert!(
            none.is_empty(),
            "query 7 does not exist and must not fall back to 0"
        );
    }

    /// `query_mask` and `valid_mask` both gate the pool.
    #[test]
    fn masks_gate_the_pool() {
        let mut pool = Pool::new(&[
            vec![(1, 2, 0.9), (3, 4, 0.9)],
            vec![(5, 6, 0.9), (7, 8, 0.9)],
        ]);
        pool.valid[1] = false;
        let pairs = generate_pairs(
            &pool.view(),
            &spec(vec![0], vec![1]),
            &settings(4, 4, 8, 0.0),
        );
        assert_eq!(pairs.len(), 2, "one head times two tails");
        assert!(
            pairs.iter().all(|p| p.head_start == 1),
            "candidate 1 was masked out"
        );

        pool.query_mask[0] = false;
        let pairs = generate_pairs(
            &pool.view(),
            &spec(vec![0], vec![1]),
            &settings(4, 4, 8, 0.0),
        );
        assert!(pairs.is_empty());
    }

    /// The output is relation-major: every pair of relation 0 precedes every
    /// pair of relation 1, matching Python's `[B, R, pair_cap]` reshape.
    #[test]
    fn output_is_relation_major() {
        let pool = Pool::new(&[vec![(1, 2, 0.9)], vec![(3, 4, 0.9)]]);
        let schema = vec![
            RelationTypeSpec::new("a", vec![0], vec![1]),
            RelationTypeSpec::new("b", vec![1], vec![0]),
        ];
        let pairs = generate_pairs(&pool.view(), &schema, &settings(2, 2, 8, 0.0));
        assert_eq!(
            pairs.iter().map(|p| p.relation_index).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(pairs[0].head_start, 1);
        assert_eq!(pairs[1].head_start, 3);
    }

    #[test]
    fn an_empty_schema_or_pool_proposes_nothing() {
        let pool = Pool::new(&[vec![(1, 2, 0.9)]]);
        assert!(generate_pairs(&pool.view(), &[], &settings(2, 2, 8, 0.0)).is_empty());
        let empty = CandidateView {
            indices: &[],
            pair_logits: &[],
            valid_mask: &[],
            query_mask: &[],
            queries: 0,
            cand_count: 0,
        };
        assert!(
            generate_pairs(&empty, &spec(vec![0], vec![0]), &settings(2, 2, 8, 0.0)).is_empty()
        );
    }

    /// The module must not acquire an unstable sort. `sort_unstable_*` would
    /// compile, pass every other test here, and quietly change the proposal set.
    #[test]
    fn this_module_uses_no_unstable_sort() {
        // Split so this very assertion does not trip the check it performs.
        let needle = concat!("sort_", "unstable");
        let offenders: Vec<&str> = include_str!("relations.rs")
            .lines()
            .map(str::trim_start)
            .filter(|line| !line.starts_with("//"))
            .filter(|line| line.contains(needle))
            .collect();
        assert!(
            offenders.is_empty(),
            "relations.rs must sort stably everywhere: {offenders:?}"
        );
    }
}
