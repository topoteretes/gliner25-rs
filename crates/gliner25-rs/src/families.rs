// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! Schema families: the schema hygiene a boundary model needs in practice.
//!
//! ## Why families
//!
//! Labels within one schema compete. The queries share the encoder context, so
//! a wide schema makes them interfere: on `gliner2.5-multi-v1` this shows up as
//! date-like entities being lost when many unrelated labels are present, a
//! regression against the span models that only appears at width.
//!
//! The remedy is to split the schema into families of related labels, run each
//! separately and merge. [`Family`] holds a group, [`run_families`] does the
//! passes and merges the results.

use std::collections::HashMap;

use crate::boundary::RelationEdge;
use crate::chunker::RelationKeyMode;
use crate::{BoundaryEngine, BoundaryOutput, BoundaryParams, Mention, SchemaTask};

/// A named group of related labels, run as one pass.
#[derive(Debug, Clone)]
pub struct Family {
    pub name: String,
    pub labels: Vec<String>,
}

impl Family {
    pub fn new(name: impl Into<String>, labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            name: name.into(),
            labels: labels.into_iter().map(Into::into).collect(),
        }
    }

    pub fn task(&self) -> SchemaTask {
        SchemaTask::Entities(self.labels.clone())
    }
}

/// Runs one pass per family and merges the mentions.
///
/// Merging is by `(word_start, word_end, field)`: a label belongs to exactly one
/// family, so the only way the same triple appears twice is a genuine duplicate,
/// and the higher-scoring one wins. Mentions from different families are kept
/// side by side even when they cover the same text — labels are independent, and
/// two families disagreeing about a stretch is information, not a conflict.
///
/// Relations are folded by [`crate::chunker::merge_relations`], the same rule
/// the chunker's window merge uses.
///
/// Cost is one encoder pass per family. That is the price of not letting the
/// labels compete; measure before assuming it is too much.
pub fn run_families(
    engine: &mut BoundaryEngine,
    text: &str,
    families: &[Family],
    params: &BoundaryParams,
) -> anyhow::Result<BoundaryOutput> {
    let mut passes = Vec::with_capacity(families.len());
    for family in families {
        passes.push(engine.extract_with(text, &[family.task()], params)?);
    }
    Ok(merge_family_outputs(passes))
}

/// Merges one [`BoundaryOutput`] per family into one.
///
/// Split out of [`run_families`] so the merge rule is exercisable without a
/// model behind it; `run_families` is exactly this plus the encoder passes.
fn merge_family_outputs(passes: Vec<BoundaryOutput>) -> BoundaryOutput {
    let mut merged: HashMap<(usize, usize, String), Mention> = HashMap::new();
    let mut classifications = Vec::new();
    let mut expected_counts = Vec::new();
    let mut relation_parts: Vec<Vec<RelationEdge>> = Vec::new();

    for out in passes {
        for mention in out.mentions {
            let key = (mention.word_start, mention.word_end, mention.field.clone());
            merged
                .entry(key)
                .and_modify(|kept| {
                    if mention.score > kept.score {
                        *kept = mention.clone();
                    }
                })
                .or_insert(mention);
        }
        classifications.extend(out.classifications);
        expected_counts.extend(out.expected_counts);
        relation_parts.push(out.relations);
    }

    let mut mentions: Vec<Mention> = merged.into_values().collect();
    mentions.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.word_start.cmp(&b.word_start))
            .then(a.word_end.cmp(&b.word_end))
            .then(a.field.cmp(&b.field))
    });

    // Every family re-runs the *same* text, so all passes share one coordinate
    // frame and a relation two families both emitted is a genuine duplicate —
    // exactly the case `merge_relations` collapses, and by the same canonical
    // key the chunker uses. Folding rather than dropping: a family pass can
    // carry relations the moment the scorer is wired, and discarding them here
    // would be silent data loss no test could see.
    let relations = crate::chunker::merge_relations(relation_parts, RelationKeyMode::default());

    BoundaryOutput { mentions, classifications, expected_counts, relations }
}

/// Splits a flat label list into families of at most `max_per_family`.
///
/// A fallback for callers who have no semantic grouping to offer. Real families
/// — dates together, identifiers together — work better than arbitrary chunks,
/// because the interference is between *unrelated* labels.
pub fn chunk_into_families(labels: &[String], max_per_family: usize) -> Vec<Family> {
    assert!(max_per_family > 0, "max_per_family must be positive");
    labels
        .chunks(max_per_family)
        .enumerate()
        .map(|(i, chunk)| Family::new(format!("family_{i}"), chunk.to_vec()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_family_becomes_an_entity_task() {
        let f = Family::new("dates", ["date", "date_of_birth"]);
        match f.task() {
            SchemaTask::Entities(labels) => assert_eq!(labels, vec!["date", "date_of_birth"]),
            other => panic!("expected entities, got {other:?}"),
        }
    }

    #[test]
    fn chunking_covers_every_label_exactly_once() {
        let labels: Vec<String> = (0..7).map(|i| format!("l{i}")).collect();
        let families = chunk_into_families(&labels, 3);
        assert_eq!(families.len(), 3);
        let flat: Vec<String> = families.iter().flat_map(|f| f.labels.clone()).collect();
        assert_eq!(flat, labels);
    }

    fn endpoint(
        text: &str,
        cs: usize,
        ce: usize,
        ws: usize,
        we: usize,
    ) -> crate::boundary::RelationEndpoint {
        crate::boundary::RelationEndpoint {
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
        head: &str,
        hs: usize,
        tail: &str,
        ts: usize,
    ) -> RelationEdge {
        RelationEdge {
            relation: relation.into(),
            score,
            head: endpoint(head, hs, hs + head.len(), 0, 1),
            tail: endpoint(tail, ts, ts + tail.len(), 2, 3),
        }
    }

    fn pass(relations: Vec<RelationEdge>) -> BoundaryOutput {
        BoundaryOutput {
            mentions: vec![],
            classifications: vec![],
            expected_counts: vec![],
            relations,
        }
    }

    #[test]
    fn family_relations_are_folded_not_discarded() {
        // Two families over the same text. One edge both passes saw, and one
        // private to each. All three must reach the output, the shared one
        // collapsed to a single entry carrying the better score, in the order
        // the keys were first seen.
        let shared = |score| edge("works_for", score, "Mario", 0, "Acme", 20);
        let only_a = edge("works_for", 0.7, "Mario", 0, "Globex", 40);
        let only_b = edge("founded_by", 0.6, "Mario", 0, "Acme", 20);

        let merged = merge_family_outputs(vec![
            pass(vec![shared(0.5), only_a]),
            pass(vec![shared(0.9), only_b]),
        ]);

        assert_eq!(merged.relations.len(), 3, "family relations must survive the merge");
        assert_eq!(merged.relations[0].tail.text, "Acme");
        assert_eq!(
            merged.relations[0].score, 0.9,
            "the duplicate collapses and the better score wins"
        );
        assert_eq!(merged.relations[1].tail.text, "Globex", "first-seen order is preserved");
        assert_eq!(
            merged.relations[2].relation, "founded_by",
            "a second family's own edge survives"
        );
    }

    #[test]
    fn family_relations_keep_the_first_seen_on_a_tie() {
        // Same canonical key, same score, different word coordinates — which
        // are not in the key, so they say which copy survived. Pins that the
        // fold uses `merge_relations`' strictly-greater rule rather than a
        // last-wins overwrite.
        let mut first = edge("works_for", 0.5, "Mario", 0, "Acme", 20);
        first.head.word_start = 11;
        let second = edge("works_for", 0.5, "Mario", 0, "Acme", 20);

        let merged = merge_family_outputs(vec![pass(vec![first]), pass(vec![second])]);
        assert_eq!(merged.relations.len(), 1);
        assert_eq!(
            merged.relations[0].head.word_start, 11,
            "only a strictly greater score replaces"
        );
    }
}
