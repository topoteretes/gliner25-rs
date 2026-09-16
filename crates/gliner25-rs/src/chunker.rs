// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! Extraction over documents longer than the model can see at once.
//!
//! A boundary export declares its length buckets, and the largest one is a hard
//! ceiling: `jugaadsrl/gliner2.5-multi-v1-onnx` stops at 512 words. Ask
//! [`BoundaryEngine::extract`](crate::BoundaryEngine::extract) for more and it
//! does not truncate — it returns [`GlinerError::NoLengthBucket`]. That is the
//! right behaviour for a single call, and useless for a document.
//!
//! This module does what `gliner2.inference.chunking` does on the Python side:
//! splits the text into overlapping word windows, runs each one, shifts the
//! offsets back onto the original document, and merges what the windows have in
//! common.
//!
//! ```no_run
//! use gliner25_rs::{BoundaryConfig, BoundaryEngine, SchemaTask};
//!
//! let document = std::fs::read_to_string("contract.txt")?;
//! let mut engine = BoundaryEngine::new(BoundaryConfig::new("models/g25"))?;
//! let tasks = vec![SchemaTask::Entities(vec!["person".into(), "location".into()])];
//! let out = engine.extract_long(&document, &tasks)?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! ## Why the windows overlap
//!
//! A mention straddling a window edge is seen by neither window whole. The
//! overlap is what gives it a second chance: with 64 words of margin, anything
//! shorter than that appears intact in at least one window. Widening the
//! overlap costs inference time — it is the fraction of the document processed
//! twice — and narrowing it starts losing mentions at the seams.
//!
//! ## What merging can and cannot fix
//!
//! Duplicate mentions from overlapping windows are collapsed by span, keeping
//! the highest score. Classifications are collapsed per label, also by highest
//! score: for a guardrail that is the answer you want — one flagged window
//! means a flagged document — and for a descriptive label it is optimistic, so
//! read a document-level classification as "somewhere in here", not "overall".
//!
//! What no merge can recover is a mention longer than the overlap, or a
//! relation whose two ends fall in different windows. Both are inherent to
//! chunking rather than to this implementation.

use crate::boundary::{BoundaryOutput, Mention, RelationEdge};
use crate::processor::WhitespaceTokenSplitter;
use anyhow::{Result, anyhow};
use std::collections::HashMap;

/// One window over the document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// Byte range `[start, end)` of this window in the original text.
    pub byte_start: usize,
    pub byte_end: usize,
    /// Half-open word range `[start, end)` in the original text.
    pub word_start: usize,
    pub word_end: usize,
}

impl Chunk {
    /// The window's own text.
    pub fn slice<'a>(&self, text: &'a str) -> &'a str {
        &text[self.byte_start..self.byte_end]
    }
}

/// How a document is cut into windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Chunker {
    size: usize,
    overlap: usize,
}

impl Default for Chunker {
    /// 384 words with 64 of overlap — the defaults `gliner2` uses.
    ///
    /// 384 leaves room under the 512-word bucket for the schema markers the
    /// prompt adds, which are counted against the same budget.
    fn default() -> Self {
        Self { size: 384, overlap: 64 }
    }
}

impl Chunker {
    pub fn new(size: usize, overlap: usize) -> Result<Self> {
        if size == 0 {
            return Err(anyhow!("chunk size must be greater than 0"));
        }
        if overlap >= size {
            return Err(anyhow!(
                "chunk overlap ({overlap}) must be smaller than the size ({size}); \
                 equal or larger and the window never advances"
            ));
        }
        Ok(Self { size, overlap })
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn overlap(&self) -> usize {
        self.overlap
    }

    /// Cuts `text` into overlapping word windows.
    ///
    /// Words are counted with the same splitter the engine tokenises with, so a
    /// window of `size` words is a window of `size` words as the model will
    /// count them — not as whitespace would.
    pub fn split(&self, text: &str) -> Result<Vec<Chunk>> {
        let splitter = WhitespaceTokenSplitter::new()?;
        let words = splitter.split_with_offsets(text);
        if words.is_empty() {
            return Ok(vec![Chunk {
                byte_start: 0,
                byte_end: text.len(),
                word_start: 0,
                word_end: 0,
            }]);
        }

        let step = self.size - self.overlap;
        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < words.len() {
            let end = (start + self.size).min(words.len());
            chunks.push(Chunk {
                byte_start: words[start].1,
                byte_end: words[end - 1].2,
                word_start: start,
                word_end: end,
            });
            if end == words.len() {
                break;
            }
            start += step;
        }
        Ok(chunks)
    }
}

/// Shifts a window's output onto the original document.
pub fn remap(output: &mut BoundaryOutput, chunk: &Chunk, text: &str) {
    for m in &mut output.mentions {
        m.char_start += chunk.byte_start;
        m.char_end += chunk.byte_start;
        m.word_start += chunk.word_start;
        m.word_end += chunk.word_start;
        // Re-slice rather than trust the window's copy: identical in practice,
        // but it keeps `text` and the offsets from ever disagreeing.
        if let Some(s) = text.get(m.char_start..m.char_end) {
            m.text = s.to_string();
        }
    }
    // Relation endpoints are window-local in exactly the same two frames, so
    // they shift the same way. Both ends of every edge, not just the head.
    for r in &mut output.relations {
        for e in [&mut r.head, &mut r.tail] {
            e.char_start += chunk.byte_start;
            e.char_end += chunk.byte_start;
            e.word_start += chunk.word_start;
            e.word_end += chunk.word_start;
            if let Some(s) = text.get(e.char_start..e.char_end) {
                e.text = s.to_string();
            }
        }
    }
}

/// Which fields decide that two relation edges are the same prediction.
///
/// `gliner2`'s `_canonical_key` (`chunking.py`) strips `confidence` at every
/// level and keeps whatever else the item carries — so the key depends on the
/// shape the caller asked for. Both shapes are reproduced here rather than one
/// guessed, because picking the wrong one merges too much or too little without
/// failing anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RelationKeyMode {
    /// Relation, spans and surface text; score excluded. Matches
    /// `include_spans=True`, which is the shape the checked-in Python parity
    /// reference was produced with, so it is the default.
    #[default]
    SpanAndText,
    /// Relation and surface text only. Matches the bare `(head, tail)` tuple
    /// `gliner2` emits when neither spans nor confidence are requested.
    ///
    /// In this shape the first item seen **always** survives, whatever the
    /// scores: `_representative_confidence` (`chunking.py:403-414`) reads only a
    /// `dict` or a `list`, and a Python tuple is neither, so it falls through to
    /// `return 0.0` for every item — `0.0 > 0.0` is false, so the replacement
    /// branch at `chunking.py:305` is dead on this path.
    TextOnly,
}

/// `(relation, head text, tail text, spans when the mode keeps them)`.
type RelationKey = (String, String, String, Option<(usize, usize, usize, usize)>);

fn relation_key(edge: &RelationEdge, mode: RelationKeyMode) -> RelationKey {
    let spans = match mode {
        RelationKeyMode::SpanAndText => Some((
            edge.head.char_start,
            edge.head.char_end,
            edge.tail.char_start,
            edge.tail.char_end,
        )),
        RelationKeyMode::TextOnly => None,
    };
    (edge.relation.clone(), edge.head.text.clone(), edge.tail.text.clone(), spans)
}

/// Collapses relations two windows both saw, keeping first-seen order.
///
/// A port of the non-span branch of `gliner2`'s `_dedupe_items`: a map from the
/// score-insensitive canonical key to a position in the output, and a duplicate
/// replaces the incumbent only when its score is **strictly** greater — so the
/// earlier window wins a tie, and the output order is the order in which keys
/// were first seen.
///
/// Python dedupes per relation label (`_merge_relation_maps` loops over labels);
/// the relation name is part of the key here, which has the same effect on a
/// flat list, minus Python's grouping of the result by label.
///
/// The score comparison is itself mode-dependent, because Python's is: see
/// [`RelationKeyMode::TextOnly`], where the incumbent is never replaced.
pub fn merge_relations(parts: Vec<Vec<RelationEdge>>, mode: RelationKeyMode) -> Vec<RelationEdge> {
    let mut seen: HashMap<RelationKey, usize> = HashMap::new();
    let mut merged: Vec<RelationEdge> = Vec::new();
    for part in parts {
        for edge in part {
            let key = relation_key(&edge, mode);
            match seen.get(&key) {
                None => {
                    seen.insert(key, merged.len());
                    merged.push(edge);
                }
                Some(&at) => {
                    // `SpanAndText` items are dicts carrying `confidence`, so
                    // Python compares real numbers; `TextOnly` items are bare
                    // tuples, which `_representative_confidence` scores `0.0`
                    // across the board, so Python never replaces there.
                    let replaces = match mode {
                        RelationKeyMode::SpanAndText => edge.score > merged[at].score,
                        RelationKeyMode::TextOnly => false,
                    };
                    if replaces {
                        merged[at] = edge;
                    }
                }
            }
        }
    }
    merged
}

/// Collapses what overlapping windows saw twice.
///
/// Two passes. Identical spans are keyed by `(range, task, field)` and the
/// highest score wins. Then, within each `(task, field)`, *overlapping* spans
/// are resolved greedily by score — the seam case, where one window saw
/// `Mario` at its edge and the neighbouring window saw `Mario Rossi` whole,
/// and both survived the first pass because their ranges differ. A single
/// window never produces such a pair (the engine's overlap policy removed it),
/// so this pass only ever removes seam artefacts. `gliner2`'s
/// `merge_chunk_results` resolves overlaps at merge for the same reason.
///
/// Fields never interact, exactly as in single-window decoding.
///
/// Classifications are collapsed per `(task, label)` by highest score.
///
/// Relations are folded by [`merge_relations`], and — this is deliberate —
/// **independently of mentions**. The seam pass below *deletes* mentions: if
/// window A saw `Mario` at its edge and window B saw `Mario Rossi` whole, the
/// wider one wins and A's `founded_by :: Mario | Acme` now names a span that no
/// surviving mention covers. That edge is kept anyway. `gliner2` never
/// cross-checks either — `_merge_relation_maps` does not look at `entities` —
/// and a [`RelationEdge`] carries its own spans and surface text, so it stands
/// on its own. Adding a referential-integrity filter here would look like a fix
/// and would in fact be a silent divergence from the reference that no parity
/// test can catch, because parity is measured on the relation set alone. Do not
/// add one.
pub fn merge(parts: Vec<BoundaryOutput>) -> BoundaryOutput {
    let mut mentions: HashMap<(usize, usize, String, String), Mention> = HashMap::new();
    let mut classes: HashMap<(String, String), crate::boundary::Classification> = HashMap::new();
    let mut expected_counts = Vec::new();
    let mut relation_parts: Vec<Vec<RelationEdge>> = Vec::new();

    for part in parts {
        for m in part.mentions {
            let key = (m.char_start, m.char_end, m.task.clone(), m.field.clone());
            match mentions.get(&key) {
                Some(seen) if seen.score >= m.score => {}
                _ => {
                    mentions.insert(key, m);
                }
            }
        }
        for c in part.classifications {
            let key = (c.task.clone(), c.label.clone());
            match classes.get(&key) {
                Some(seen) if seen.score >= c.score => {}
                _ => {
                    classes.insert(key, c);
                }
            }
        }
        expected_counts.extend(part.expected_counts);
        relation_parts.push(part.relations);
    }
    let relations = merge_relations(relation_parts, RelationKeyMode::default());

    // Seam pass: greedy by score within each (task, field), spans half-open.
    let mut mentions: Vec<Mention> = mentions.into_values().collect();
    mentions.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.word_start.cmp(&b.word_start))
            .then(a.word_end.cmp(&b.word_end))
    });
    let mut kept: Vec<Mention> = Vec::new();
    for cand in mentions {
        let clashes = kept.iter().any(|k| {
            k.task == cand.task
                && k.field == cand.field
                && cand.word_start < k.word_end
                && k.word_start < cand.word_end
        });
        if !clashes {
            kept.push(cand);
        }
    }
    let mut mentions = kept;
    mentions.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.char_start.cmp(&b.char_start))
    });
    let mut classifications: Vec<crate::boundary::Classification> =
        classes.into_values().collect();
    classifications.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    BoundaryOutput { mentions, classifications, expected_counts, relations }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary::RelationEndpoint;

    #[test]
    fn windows_advance_by_size_minus_overlap() {
        let text = (0..10).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        let chunks = Chunker::new(4, 1).unwrap().split(&text).unwrap();
        let spans: Vec<(usize, usize)> =
            chunks.iter().map(|c| (c.word_start, c.word_end)).collect();
        assert_eq!(spans, vec![(0, 4), (3, 7), (6, 10)]);
    }

    #[test]
    fn every_word_is_covered() {
        let text = (0..97).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        let chunks = Chunker::new(16, 4).unwrap().split(&text).unwrap();
        let mut covered = [false; 97];
        for c in &chunks {
            covered[c.word_start..c.word_end].fill(true);
        }
        assert!(covered.iter().all(|c| *c), "a window boundary dropped a word");
    }

    #[test]
    fn short_text_is_one_window() {
        let chunks = Chunker::default().split("Mario Rossi lavora a Roma.").unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].word_start, 0);
    }

    #[test]
    fn empty_text_still_yields_a_window() {
        assert_eq!(Chunker::default().split("").unwrap().len(), 1);
    }

    fn men(field: &str, cs: usize, ce: usize, ws: usize, we: usize, score: f32) -> Mention {
        Mention {
            text: String::new(),
            task: "entities".into(),
            field: field.into(),
            score,
            char_start: cs,
            char_end: ce,
            word_start: ws,
            word_end: we,
            query_id: 0,
        }
    }

    #[test]
    fn merge_collapses_seam_truncations_within_a_field() {
        // half-open ranges: the truncation is [5,6), the whole mention [5,7).
        let a = BoundaryOutput {
            mentions: vec![men("person", 30, 35, 5, 6, 0.71)],
            classifications: vec![],
            expected_counts: vec![],
            relations: vec![],
        };
        let b = BoundaryOutput {
            mentions: vec![men("person", 30, 41, 5, 7, 0.97)],
            classifications: vec![],
            expected_counts: vec![],
            relations: vec![],
        };
        let merged = merge(vec![a, b]);
        assert_eq!(merged.mentions.len(), 1);
        assert_eq!(merged.mentions[0].word_end, 7, "the whole mention wins");
    }

    #[test]
    fn merge_adjacent_half_open_spans_do_not_clash() {
        // [5,6) and [6,7) touch but do not overlap under half-open semantics.
        let a = BoundaryOutput {
            mentions: vec![men("person", 30, 35, 5, 6, 0.9)],
            classifications: vec![],
            expected_counts: vec![],
            relations: vec![],
        };
        let b = BoundaryOutput {
            mentions: vec![men("person", 36, 41, 6, 7, 0.9)],
            classifications: vec![],
            expected_counts: vec![],
            relations: vec![],
        };
        assert_eq!(merge(vec![a, b]).mentions.len(), 2);
    }

    fn endpoint(text: &str, cs: usize, ce: usize, ws: usize, we: usize) -> RelationEndpoint {
        RelationEndpoint {
            text: text.into(),
            char_start: cs,
            char_end: ce,
            word_start: ws,
            word_end: we,
        }
    }

    fn edge(
        relation: &str,
        score: f32,
        head: RelationEndpoint,
        tail: RelationEndpoint,
    ) -> RelationEdge {
        RelationEdge { relation: relation.into(), score, head, tail }
    }

    fn out(mentions: Vec<Mention>, relations: Vec<RelationEdge>) -> BoundaryOutput {
        BoundaryOutput { mentions, classifications: vec![], expected_counts: vec![], relations }
    }

    #[test]
    fn remap_shifts_relation_endpoints() {
        // bytes: zero 0..4, one 5..8, Mario 9..14, Rossi 15..20, works 21..26,
        //        for 27..30, Acme 31..35.
        let text = "zero one Mario Rossi works for Acme";
        let chunk = Chunk { byte_start: 9, byte_end: 35, word_start: 2, word_end: 7 };
        // Window-local coordinates, as a scorer running on that window emits
        // them. The tail's surface text is deliberately stale, so the test also
        // proves remap re-slices instead of trusting the window's copy.
        let mut output = out(
            vec![],
            vec![edge(
                "works_for: person works at an organization",
                0.9,
                endpoint("Mario Rossi", 0, 11, 0, 2),
                endpoint("cme", 22, 26, 4, 5),
            )],
        );
        remap(&mut output, &chunk, text);
        let r = &output.relations[0];
        assert_eq!((r.head.char_start, r.head.char_end), (9, 20));
        assert_eq!((r.head.word_start, r.head.word_end), (2, 4));
        assert_eq!(r.head.text, "Mario Rossi");
        assert_eq!((r.tail.char_start, r.tail.char_end), (31, 35), "the tail shifts too");
        assert_eq!((r.tail.word_start, r.tail.word_end), (6, 7), "the tail shifts too");
        assert_eq!(r.tail.text, "Acme", "the surface text is re-sliced, not trusted");
    }

    #[test]
    fn merge_relations_keeps_first_on_tie() {
        // Same canonical key, same score, different word coordinates — which
        // are not part of the key, so they reveal which copy survived.
        let a = edge("works_for", 0.5, endpoint("A", 0, 1, 0, 1), endpoint("B", 2, 3, 1, 2));
        let b = edge("works_for", 0.5, endpoint("A", 0, 1, 40, 41), endpoint("B", 2, 3, 41, 42));
        let merged = merge_relations(vec![vec![a], vec![b]], RelationKeyMode::default());
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].head.word_start, 0, "only a strictly greater score replaces");
    }

    #[test]
    fn merge_relations_prefers_strictly_higher_score() {
        let lo = edge("works_for", 0.5, endpoint("A", 0, 1, 0, 1), endpoint("B", 2, 3, 1, 2));
        let hi = edge("works_for", 0.9, endpoint("A", 0, 1, 0, 1), endpoint("B", 2, 3, 1, 2));
        let up =
            merge_relations(vec![vec![lo.clone()], vec![hi.clone()]], RelationKeyMode::default());
        assert_eq!(up.len(), 1);
        assert_eq!(up[0].score, 0.9, "a later, better duplicate replaces the incumbent");
        let down = merge_relations(vec![vec![hi], vec![lo]], RelationKeyMode::default());
        assert_eq!(down.len(), 1);
        assert_eq!(down[0].score, 0.9, "a later, worse duplicate does not");
    }

    #[test]
    fn merge_relations_preserves_first_seen_order() {
        let x = edge("works_for", 0.5, endpoint("X", 0, 1, 0, 1), endpoint("P", 2, 3, 1, 2));
        let y = edge("works_for", 0.5, endpoint("Y", 4, 5, 2, 3), endpoint("P", 2, 3, 1, 2));
        let x_better = edge("works_for", 0.9, endpoint("X", 0, 1, 0, 1), endpoint("P", 2, 3, 1, 2));
        let merged = merge_relations(vec![vec![x, y], vec![x_better]], RelationKeyMode::default());
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].head.text, "X", "a replaced entry keeps its original position");
        assert_eq!(merged[0].score, 0.9);
        assert_eq!(merged[1].head.text, "Y");
    }

    #[test]
    fn merge_relations_keys_on_the_relation_name() {
        let a = edge("works_for", 0.5, endpoint("A", 0, 1, 0, 1), endpoint("B", 2, 3, 1, 2));
        let mut b = a.clone();
        b.relation = "founded_by".into();
        let merged = merge_relations(vec![vec![a, b]], RelationKeyMode::default());
        assert_eq!(merged.len(), 2, "two relation types over one pair are two edges");
    }

    #[test]
    fn merge_relations_text_only_mode_ignores_spans() {
        // The same pair of surface forms at two different places in the text.
        let first =
            edge("works_for", 0.5, endpoint("Mario", 0, 5, 0, 1), endpoint("Acme", 10, 14, 2, 3));
        let second =
            edge("works_for", 0.4, endpoint("Mario", 20, 25, 4, 5), endpoint("Acme", 30, 34, 6, 7));
        let spanned = merge_relations(
            vec![vec![first.clone(), second.clone()]],
            RelationKeyMode::SpanAndText,
        );
        assert_eq!(spanned.len(), 2, "spans are part of the default key");
        let flat = merge_relations(vec![vec![first, second]], RelationKeyMode::TextOnly);
        assert_eq!(flat.len(), 1, "text-only keys collapse the two occurrences");
        assert_eq!(flat[0].head.char_start, 0);
    }

    #[test]
    fn merge_relations_text_only_never_replaces_the_first_seen() {
        // The bare `(head, tail)` tuple carries no confidence, so Python's
        // `_representative_confidence` returns 0.0 for both sides and the
        // replacement branch is unreachable: first seen wins even when a later
        // duplicate scores far higher. `SpanAndText` is the contrast — there the
        // items are dicts with a `confidence`, so the better one does replace.
        let weak_first =
            edge("works_for", 0.1, endpoint("Mario", 0, 5, 0, 1), endpoint("Acme", 10, 14, 2, 3));
        let strong_later =
            edge("works_for", 0.9, endpoint("Mario", 20, 25, 4, 5), endpoint("Acme", 30, 34, 6, 7));
        let flat = merge_relations(
            vec![vec![weak_first.clone()], vec![strong_later.clone()]],
            RelationKeyMode::TextOnly,
        );
        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].score, 0.1, "a higher score does not replace under TextOnly");
        assert_eq!(flat[0].head.char_start, 0, "the first occurrence is the survivor");

        // Same two edges, same spans, differing only in score, under the
        // default mode: here the strictly greater score does replace.
        let mut same_span = strong_later;
        same_span.head = weak_first.head.clone();
        same_span.tail = weak_first.tail.clone();
        let spanned =
            merge_relations(vec![vec![weak_first], vec![same_span]], RelationKeyMode::SpanAndText);
        assert_eq!(spanned.len(), 1);
        assert_eq!(spanned[0].score, 0.9, "SpanAndText still replaces on a better score");
    }

    #[test]
    fn merge_folds_a_relation_two_windows_both_saw() {
        let e = |score| {
            edge(
                "works_for",
                score,
                endpoint("Mario Rossi", 30, 41, 5, 7),
                endpoint("Acme", 50, 54, 9, 10),
            )
        };
        let merged = merge(vec![out(vec![], vec![e(0.61)]), out(vec![], vec![e(0.93)])]);
        assert_eq!(merged.relations.len(), 1);
        assert_eq!(merged.relations[0].score, 0.93);
    }

    #[test]
    fn relation_outlives_its_endpoint_mention() {
        // Window A saw the truncated `Mario` [5,6) and hung an edge off it;
        // window B saw `Mario Rossi` [5,7) whole. The seam pass deletes A's
        // mention. The edge must survive anyway: gliner2's
        // `_merge_relation_maps` never consults the entity set, and a filter
        // here would be a divergence no parity test could catch.
        let a = out(
            vec![men("person", 30, 35, 5, 6, 0.71)],
            vec![edge(
                "works_for",
                0.8,
                endpoint("Mario", 30, 35, 5, 6),
                endpoint("Acme", 50, 54, 9, 10),
            )],
        );
        let b = out(vec![men("person", 30, 41, 5, 7, 0.97)], vec![]);
        let merged = merge(vec![a, b]);
        assert_eq!(merged.mentions.len(), 1, "the seam pass keeps only the wider mention");
        assert_eq!(merged.mentions[0].word_end, 7);
        assert!(
            !merged.mentions.iter().any(|m| m.char_end == 35),
            "the mention the edge names really is gone"
        );
        assert_eq!(merged.relations.len(), 1, "the edge outlives it");
        assert_eq!(merged.relations[0].head.char_end, 35);
    }

    #[test]
    fn overlap_must_be_smaller_than_size() {
        assert!(Chunker::new(64, 64).is_err());
        assert!(Chunker::new(64, 65).is_err());
        assert!(Chunker::new(0, 0).is_err());
    }
}
