// COGNEE-EVAL: the four de-duplication stages, one at a time.
//
// `_deduplicate_relation_edges` (`boundary/engine.py:900-1003`) is one static
// method in Python. It is ported as four functions, and it is tested as four,
// because the stages fail for different reasons and a single test over the
// composed pipeline cannot say *which* rule drifted.
//
// Every test below is built so that **removing its stage from
// `deduplicate_relation_edges` changes the assertion**, not merely the internal
// bookkeeping. That is harder than it sounds for stages 1 and 3, which are a
// span-level and a token-level version of the same idea and routinely rescue
// each other's edge *count*. The discriminator in those tests is therefore the
// surviving edge's **score and span**, not how many survived — the two stages
// keep different edges.
//
// No model, no ONNX, no fixture: these are pure functions over hand-built
// edges, so they run in microseconds and fail for exactly one reason.

use gliner25_rs::boundary::{RelationEdge, RelationEndpoint};
use gliner25_rs::relation_decode::{
    EdgeDecoder, OrderedMap, canonical_mentions, canonicalise_containment, casefold,
    collapse_semantic_duplicates, deduplicate_relation_edges, drop_dominated_by_token_superset,
    semantic_text, sort_edges,
};
use gliner25_rs::relations::ProposedPair;

// ── helpers ──────────────────────────────────────────────────────────────────

/// An endpoint whose surface is read out of `text`, so a test cannot quietly
/// disagree with itself about what sits at those offsets.
fn end_at(text: &str, start: usize, end: usize) -> RelationEndpoint {
    RelationEndpoint {
        text: text[start..end].trim().to_string(),
        char_start: start,
        char_end: end,
        word_start: 0,
        word_end: 0,
    }
}

fn edge(score: f32, head: RelationEndpoint, tail: RelationEndpoint) -> RelationEdge {
    RelationEdge {
        relation: "rel".to_string(),
        score,
        head,
        tail,
    }
}

/// Byte range per whitespace-separated word, the shape
/// `ProcessedRecord::word_to_char_maps` has.
fn word_map(text: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut cursor = 0usize;
    for word in text.split(' ') {
        out.push((cursor, cursor + word.len()));
        cursor += word.len() + 1;
    }
    out
}

fn pair(relation_index: usize, hs: usize, he: usize, ts: usize, te: usize) -> ProposedPair {
    ProposedPair {
        relation_index,
        head_start: hs,
        head_end: he,
        tail_start: ts,
        tail_end: te,
        head_prob: 0.9,
        tail_prob: 0.9,
        head_query: 0,
        tail_query: 1,
        head_slot: 0,
        tail_slot: 0,
    }
}

/// The logit whose `sigmoid` is `p`. Lets a test name a probability and let the
/// decoder do the arithmetic, instead of hard-coding a magic logit.
fn logit_for(p: f32) -> f32 {
    (p / (1.0 - p)).ln()
}

fn surfaces(edges: &[RelationEdge]) -> Vec<String> {
    edges
        .iter()
        .map(|e| format!("{}|{}", e.head.text, e.tail.text))
        .collect()
}

// ── STAGE 1 — containment canonicalisation ───────────────────────────────────

/// `Apple` is collapsed into `Apple Inc.`, and the collapsed edge keeps the
/// **better score of the two**, which is the part stage 3 cannot reproduce.
///
/// Remove stage 1 and this still returns one edge — stage 3 drops
/// `{apple} ⊂ {apple, inc.}` with an identical tail — but it is the *other*
/// edge: score 0.50 instead of 0.90. That difference is the assertion.
#[test]
fn stage1_collapses_a_contained_mention_and_keeps_its_score() {
    let text = "Apple Inc. sells the iPhone worldwide, and Apple is loved.";
    // "Apple" 0..5 is contained by "Apple Inc." 0..10; "iPhone" 21..27.
    assert_eq!(&text[0..5], "Apple");
    assert_eq!(&text[0..10], "Apple Inc.");
    assert_eq!(&text[21..27], "iPhone");

    let strong = edge(0.90, end_at(text, 0, 5), end_at(text, 21, 27));
    let weak = edge(0.50, end_at(text, 0, 10), end_at(text, 21, 27));

    let out = deduplicate_relation_edges(vec![strong, weak], text);
    assert_eq!(surfaces(&out), vec!["Apple Inc.|iPhone".to_string()]);
    assert_eq!(out.len(), 1);
    assert!(
        (out[0].score - 0.90).abs() < 1e-6,
        "stage 1 must canonicalise `Apple` to `Apple Inc.` and then keep the higher-scoring \
         edge under the shared key; got score {} (0.50 means stage 1 did not run and stage 3 \
         dropped the wrong edge instead)",
        out[0].score
    );
    assert_eq!(out[0].head.char_start, 0);
    assert_eq!(out[0].head.char_end, 10);
}

/// The containment pool is built **per side**: a head is never canonicalised
/// against a tail, however well the spans would contain each other.
#[test]
fn stage1_never_canonicalises_a_head_against_a_tail() {
    let text = "Apple Inc. and Apple parted ways.";
    // head pool: only 15..20 "Apple"; tail pool: only 0..10 "Apple Inc.".
    let only = edge(0.8, end_at(text, 15, 20), end_at(text, 0, 10));
    let out = canonicalise_containment(vec![only], text);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].head.text, "Apple");
    assert_eq!(out[0].head.char_start, 15);
}

/// Python's `max()` returns the **first** maximal element; Rust's
/// `max_by_key` returns the last. With two equally long containing mentions the
/// key `(length, -start)` settles it — the one starting earlier wins — and this
/// pins that the port did not inherit Rust's tie rule.
#[test]
fn stage1_picks_the_longest_containing_mention_then_the_earliest() {
    let text = "AA BB CC";
    // "AA BB" 0..5 and "BB CC" 3..8 both have length 5; only 0..5 contains 3..5.
    let ends = [end_at(text, 3, 5), end_at(text, 0, 5), end_at(text, 3, 8)];
    let canonical = canonical_mentions(ends.iter(), text);
    let picked = canonical
        .get(&(3, 5))
        .expect("the queried span is always in its own pool");
    assert_eq!((picked.char_start, picked.char_end), (0, 5));
    // A mention that nothing longer contains stays itself.
    let self_picked = canonical.get(&(3, 8)).expect("present");
    assert_eq!((self_picked.char_start, self_picked.char_end), (3, 8));
}

/// The exact collapse replaces on a **strictly** greater score, so the first
/// edge seen wins a tie.
#[test]
fn stage1_keeps_the_first_edge_on_an_exact_score_tie() {
    let text = "Apple Inc. sells the iPhone.";
    let first = RelationEdge {
        relation: "first".to_string(),
        ..edge(0.7, end_at(text, 0, 10), end_at(text, 21, 27))
    };
    let second = RelationEdge {
        relation: "second".to_string(),
        ..edge(0.7, end_at(text, 0, 10), end_at(text, 21, 27))
    };
    let out = canonicalise_containment(vec![first, second], text);
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].relation, "first");
}

// ── STAGE 2 — one edge per pair of surface forms ─────────────────────────────

/// Two mentions of the same pair in different sentences collapse to one, and
/// the survivor is the **closest** pair, not the highest scoring one.
///
/// Stage 1 cannot do this (the spans are disjoint, so neither contains the
/// other) and stage 3 cannot either (the token sets are equal, and domination
/// needs a *strict* subset). Remove stage 2 and two edges come back.
#[test]
fn stage2_collapses_repeated_mentions_and_keeps_the_closest() {
    let text = "Microsoft builds Azure. Later on, Microsoft also shipped Azure again.";
    assert_eq!(&text[0..9], "Microsoft");
    assert_eq!(&text[17..22], "Azure");
    assert_eq!(&text[34..43], "Microsoft");
    assert_eq!(&text[57..62], "Azure");

    let near = edge(0.60, end_at(text, 0, 9), end_at(text, 17, 22)); // gap 8 chars
    let far = edge(0.95, end_at(text, 34, 43), end_at(text, 57, 62)); // gap 14 chars

    let out = deduplicate_relation_edges(vec![near, far], text);
    assert_eq!(out.len(), 1, "one semantic edge, got {:?}", surfaces(&out));
    assert_eq!(
        out[0].head.char_start, 0,
        "rank orders on distance first, so the near pair wins"
    );
    assert!(
        (out[0].score - 0.60).abs() < 1e-6,
        "and it wins despite the lower score"
    );
}

/// `rank` is `(distance, -score, head start, tail start)`, so a distance tie
/// falls through to the **higher** score.
#[test]
fn stage2_breaks_a_distance_tie_on_the_higher_score() {
    let text = "Acme hires Bob. Acme hires Bob.";
    assert_eq!(&text[0..4], "Acme");
    assert_eq!(&text[11..14], "Bob");
    assert_eq!(&text[16..20], "Acme");
    assert_eq!(&text[27..30], "Bob");

    let low = edge(0.55, end_at(text, 0, 4), end_at(text, 11, 14));
    let high = edge(0.85, end_at(text, 16, 20), end_at(text, 27, 30));
    // Both gaps are " hires " — 7 characters — so distance ties.
    let out = collapse_semantic_duplicates(vec![low, high], text);
    assert_eq!(out.len(), 1);
    assert!(
        (out[0].score - 0.85).abs() < 1e-6,
        "the distance tie must fall through to -score"
    );
}

/// The key is case- and whitespace-folded, so `IBM` and `ibm` are one edge.
#[test]
fn stage2_keys_on_the_folded_surface_not_the_raw_one() {
    let text = "IBM    hires Ann. ibm hires Ann.";
    let a = edge(0.9, end_at(text, 0, 3), end_at(text, 13, 16));
    let b = edge(0.8, end_at(text, 18, 21), end_at(text, 28, 31));
    assert_eq!(collapse_semantic_duplicates(vec![a, b], text).len(), 1);
}

// ── STAGE 3 — a partial argument beside a complete one ──────────────────────

/// `Apple|Mac` is dropped beside `Apple Inc.|Mac` although the two head spans
/// are **disjoint**, which is precisely what stage 1 cannot see.
///
/// Remove stage 3 and both edges come back: stage 1 leaves them alone (neither
/// span contains the other) and stage 2 leaves them alone (the surfaces differ).
#[test]
fn stage3_drops_a_token_subset_argument_whose_spans_never_overlap() {
    let text = "Apple Inc. builds the Mac. Apple sells it everywhere.";
    assert_eq!(&text[0..10], "Apple Inc.");
    assert_eq!(&text[22..25], "Mac");
    assert_eq!(&text[27..32], "Apple");

    let partial = edge(0.90, end_at(text, 27, 32), end_at(text, 22, 25));
    let complete = edge(0.80, end_at(text, 0, 10), end_at(text, 22, 25));

    let out = deduplicate_relation_edges(vec![partial, complete], text);
    assert_eq!(
        surfaces(&out),
        vec!["Apple Inc.|Mac".to_string()],
        "the strictly-partial head must be dropped even though it scored higher"
    );
}

/// Domination needs a **strict** subset: equal token sets dominate nothing, or
/// two edges with the same surfaces would annihilate each other.
#[test]
fn stage3_leaves_equal_token_sets_alone() {
    let text = "Acme hires Bob. Acme hires Bob.";
    let a = edge(0.9, end_at(text, 0, 4), end_at(text, 11, 14));
    let b = edge(0.8, end_at(text, 16, 20), end_at(text, 27, 30));
    assert_eq!(drop_dominated_by_token_superset(vec![a, b]).len(), 2);
}

/// Only one side may shrink: a subset head with a *different* tail is not
/// dominated.
#[test]
fn stage3_requires_the_opposite_endpoint_to_be_identical() {
    let text = "Apple Inc. builds the Mac. Apple sells the iPhone.";
    let a = edge(0.9, end_at(text, 27, 32), end_at(text, 43, 49)); // Apple | iPhone
    let b = edge(0.8, end_at(text, 0, 10), end_at(text, 22, 25)); // Apple Inc. | Mac
    assert_eq!(drop_dominated_by_token_superset(vec![a, b]).len(), 2);
}

// ── STAGE 4 — the output order ───────────────────────────────────────────────

/// The output is ordered by `(head start, tail start, -score)`, whatever order
/// the earlier stages left it in.
///
/// The inputs below are fed back-to-front, and stages 1-3 preserve input order,
/// so without stage 4 the first edge out is `Beta`.
#[test]
fn stage4_orders_the_output_by_head_then_tail_then_score() {
    let text = "Zeta works for Acme. Beta works for Acme.";
    assert_eq!(&text[0..4], "Zeta");
    assert_eq!(&text[15..19], "Acme");
    assert_eq!(&text[21..25], "Beta");
    assert_eq!(&text[36..40], "Acme");

    let late = edge(0.9, end_at(text, 21, 25), end_at(text, 36, 40));
    let early = edge(0.6, end_at(text, 0, 4), end_at(text, 15, 19));

    let out = deduplicate_relation_edges(vec![late, early], text);
    assert_eq!(
        surfaces(&out),
        vec!["Zeta|Acme".to_string(), "Beta|Acme".to_string()],
        "stage 4 must sort by head start; input order was Beta first"
    );
}

/// Equal spans fall through to `-score`, i.e. the higher score comes first, and
/// the sort is stable beyond that.
#[test]
fn stage4_breaks_a_span_tie_on_the_higher_score() {
    let text = "Acme hires Bob.";
    let low = RelationEdge {
        relation: "low".to_string(),
        ..edge(0.4, end_at(text, 0, 4), end_at(text, 11, 14))
    };
    let high = RelationEdge {
        relation: "high".to_string(),
        ..edge(0.9, end_at(text, 0, 4), end_at(text, 11, 14))
    };
    let out = sort_edges(vec![low, high]);
    assert_eq!(out[0].relation, "high");
}

// ── the composition ─────────────────────────────────────────────────────────

/// `len < 2` returns the input untouched — canonicalisation included.
#[test]
fn a_single_edge_is_returned_verbatim() {
    let text = "Apple Inc. sells the iPhone.";
    let one = edge(0.7, end_at(text, 0, 5), end_at(text, 21, 27));
    let out = deduplicate_relation_edges(vec![one.clone()], text);
    assert_eq!(out, vec![one]);
}

/// The four stages together on the shape that actually shows up in the parity
/// scenarios: a repeated company under two surface forms, related to two people.
#[test]
fn the_four_stages_together_reproduce_the_apple_inc_shape() {
    let text = "Apple Inc. was founded by Steve Jobs and Steve Wozniak. Apple grew fast.";
    assert_eq!(&text[0..5], "Apple");
    assert_eq!(&text[0..10], "Apple Inc.");
    assert_eq!(&text[26..36], "Steve Jobs");
    assert_eq!(&text[41..54], "Steve Wozniak");
    assert_eq!(&text[56..61], "Apple");

    let edges = vec![
        edge(0.71, end_at(text, 0, 5), end_at(text, 26, 36)), // Apple      | Jobs
        edge(0.93, end_at(text, 0, 10), end_at(text, 26, 36)), // Apple Inc. | Jobs
        edge(0.64, end_at(text, 56, 61), end_at(text, 26, 36)), // Apple (2nd)| Jobs
        edge(0.88, end_at(text, 0, 10), end_at(text, 41, 54)), // Apple Inc. | Wozniak
        edge(0.52, end_at(text, 0, 5), end_at(text, 41, 54)), // Apple      | Wozniak
    ];
    let out = deduplicate_relation_edges(edges, text);
    assert_eq!(
        surfaces(&out),
        vec![
            "Apple Inc.|Steve Jobs".to_string(),
            "Apple Inc.|Steve Wozniak".to_string()
        ],
        "five cross-product edges over two surface forms collapse to two semantic edges"
    );
}

// ── casefold, the one string primitive that is easy to get wrong ────────────

/// `casefold()` is not `to_lowercase()`: the difference is a *full* fold that
/// can lengthen the string.
#[test]
fn casefold_is_not_to_lowercase() {
    assert_eq!(casefold("Straße"), "strasse");
    assert_ne!(casefold("Straße"), "Straße".to_lowercase());
    assert_eq!(casefold("STRASSE"), "strasse");
    // Both spellings therefore land on the same stage-2 key.
    assert_eq!(semantic_text("Straße"), semantic_text("STRASSE"));
    // Folds `to_lowercase` leaves alone.
    assert_eq!(casefold("ſ"), "s");
    assert_eq!(casefold("ΟΔΟΣ").chars().last(), Some('σ'));
    assert_eq!(casefold("ﬁt"), "fit");
    // And the ordinary path still works.
    assert_eq!(casefold("Apple Inc."), "apple inc.");
}

/// `semantic_text` also normalises runs of whitespace, including newlines and
/// tabs, exactly as `" ".join(x.split())` does.
#[test]
fn semantic_text_collapses_every_kind_of_whitespace() {
    assert_eq!(semantic_text("  Apple \t\n Inc.  "), "apple inc.");
    assert_eq!(semantic_text("Apple Inc."), semantic_text("Apple\n\nInc."));
}

// ── decode ──────────────────────────────────────────────────────────────────

/// The decode threshold is **0.5**, not the 0.2 that selected the arguments.
///
/// A pair scoring 0.30 is above the argument-proposal threshold and below the
/// decode threshold, and must not produce an edge.
#[test]
fn decode_abstains_at_the_decode_threshold_not_the_proposal_threshold() {
    let text = "Apple Inc. sells the iPhone today";
    let map = word_map(text);
    let names = vec!["produces".to_string()];
    let decoder = EdgeDecoder {
        relation_names: &names,
        text,
        word_to_char: &map,
        num_words: map.len(),
        threshold: 0.5,
        temperature: 1.0,
    };
    // words: 0 Apple 1 Inc. 2 sells 3 the 4 iPhone 5 today
    let pairs = vec![pair(0, 0, 2, 4, 5)];
    assert!(decoder.decode(&pairs, &[logit_for(0.30)]).is_empty());
    assert_eq!(decoder.decode(&pairs, &[logit_for(0.30)]).len(), 0);
    let kept = decoder.decode(&pairs, &[logit_for(0.80)]);
    assert_eq!(surfaces(&kept), vec!["Apple Inc.|iPhone".to_string()]);
    assert!((kept[0].score - 0.80).abs() < 1e-5);
}

/// A pair whose `relation_index` is outside `0..R` scores exactly 0.0 in the
/// graph, and `sigmoid(0.0)` is 0.5 — which is *not* below a 0.5 threshold.
/// Such a pair is dropped by index, before the threshold can let it through.
#[test]
fn decode_drops_a_pair_whose_relation_index_is_out_of_range() {
    let text = "Apple Inc. sells the iPhone today";
    let map = word_map(text);
    let names = vec!["produces".to_string()];
    let decoder = EdgeDecoder {
        relation_names: &names,
        text,
        word_to_char: &map,
        num_words: map.len(),
        threshold: 0.5,
        temperature: 1.0,
    };
    let pairs = vec![pair(7, 0, 2, 4, 5)];
    assert!(
        decoder.decode(&pairs, &[0.0]).is_empty(),
        "sigmoid(0.0) == 0.5 clears a `>= 0.5` test; the index check has to come first"
    );
}

/// Spans reaching past the window's word count are dropped, as
/// `0 <= hs < he <= text_len` does in Python.
#[test]
fn decode_drops_a_span_past_the_word_count() {
    let text = "Apple Inc. sells the iPhone today";
    let map = word_map(text);
    let names = vec!["produces".to_string()];
    let decoder = EdgeDecoder {
        relation_names: &names,
        text,
        word_to_char: &map,
        num_words: 4, // the window really only has four words
        threshold: 0.5,
        temperature: 1.0,
    };
    let pairs = vec![pair(0, 0, 2, 4, 5)];
    assert!(decoder.decode(&pairs, &[logit_for(0.99)]).is_empty());
}

/// De-duplication runs **per relation type**: two different relations between
/// the same two entities are both kept, because Python groups by label first.
#[test]
fn decode_deduplicates_within_a_relation_type_not_across_them() {
    let text = "Apple Inc. sells the iPhone today";
    let map = word_map(text);
    let names = vec!["produces".to_string(), "sells".to_string()];
    let decoder = EdgeDecoder {
        relation_names: &names,
        text,
        word_to_char: &map,
        num_words: map.len(),
        threshold: 0.5,
        temperature: 1.0,
    };
    let pairs = vec![
        pair(0, 0, 2, 4, 5),
        pair(1, 0, 2, 4, 5),
        pair(0, 0, 1, 4, 5),
    ];
    let out = decoder.decode(&pairs, &[logit_for(0.9), logit_for(0.8), logit_for(0.7)]);
    assert_eq!(
        out.len(),
        2,
        "one per relation type, got {:?}",
        surfaces(&out)
    );
    assert_eq!(out[0].relation, "produces");
    assert_eq!(out[1].relation, "sells");
    // Within `produces`, `Apple` (words 0..1) was canonicalised into
    // `Apple Inc.` (words 0..2) and the higher score survived.
    assert_eq!(out[0].head.text, "Apple Inc.");
    assert!((out[0].score - 0.9).abs() < 1e-5);
}

/// `temperature` divides the logit before the sigmoid.
#[test]
fn decode_applies_the_relation_temperature() {
    let text = "Apple Inc. sells the iPhone today";
    let map = word_map(text);
    let names = vec!["produces".to_string()];
    let hot = EdgeDecoder {
        relation_names: &names,
        text,
        word_to_char: &map,
        num_words: map.len(),
        threshold: 0.5,
        temperature: 4.0,
    };
    // sigmoid(2.0) = 0.881 clears the threshold; sigmoid(2.0 / 4.0) = 0.622 too,
    // but the reported score has to be the tempered one.
    let out = hot.decode(&[pair(0, 0, 2, 4, 5)], &[2.0]);
    assert_eq!(out.len(), 1);
    assert!(
        (out[0].score - 0.622_459_3).abs() < 1e-5,
        "score {} should be sigmoid(logit / temperature)",
        out[0].score
    );
}

// ── the ordered map the stages are built on ─────────────────────────────────

/// Rewriting a key must not move it. Python's `dict` behaves this way, and all
/// three stages depend on it: the first entry under a key fixes the output
/// position, and a later higher-scoring edge replaces the *value* only.
///
/// A `HashMap` here would change the output with nothing failing, which is why
/// this is pinned rather than assumed.
#[test]
fn the_ordered_map_keeps_the_position_a_key_first_claimed() {
    let mut map: OrderedMap<&str, i32> = OrderedMap::new();
    map.insert("a", 1);
    map.insert("b", 2);
    map.insert("c", 3);
    map.insert("a", 99);
    assert_eq!(map.values().copied().collect::<Vec<_>>(), vec![99, 2, 3]);
    assert_eq!(map.len(), 3);
    assert_eq!(map.get(&"a"), Some(&99));
    assert!(map.get(&"zz").is_none());
    if let Some(slot) = map.get_mut(&"b") {
        *slot = 7;
    }
    assert_eq!(map.into_values().collect::<Vec<_>>(), vec![99, 7, 3]);
}
