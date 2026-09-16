// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! Turning relation-scorer logits into the edges `gliner2` would emit.
//!
//! Port of `BoundaryExtractor._decode_relations` (`boundary/engine.py:794-891`)
//! and `_deduplicate_relation_edges` (`boundary/engine.py:900-1000`).
//!
//! ## Why the de-duplication is split into four named stages
//!
//! Python writes it as one static method. It is reproduced here as four
//! separately callable functions because they fail for different reasons and a
//! single function that happens to agree on four scenarios tells you nothing
//! about *which* rule drifted:
//!
//! | stage | function | what it collapses |
//! |---|---|---|
//! | 1 | [`canonicalise_containment`] | `Apple` -> `Apple Inc.`, then exact duplicates by span |
//! | 2 | [`collapse_semantic_duplicates`] | repeated mentions of the same pair of surfaces |
//! | 3 | [`drop_dominated_by_token_superset`] | a partial argument beside a more complete one |
//! | 4 | [`sort_edges`] | the deterministic output order |
//!
//! [`deduplicate_relation_edges`] composes them, in that order, and is the only
//! thing the engine calls.
//!
//! ## Traps that are reproduced deliberately
//!
//! * **Insertion order is part of the answer.** Python uses plain `dict`s, which
//!   preserve it, and the winner of a tie is whichever entry got there first. No
//!   [`std::collections::HashMap`] appears below: the ordered maps are small
//!   association vectors ([`OrderedMap`]), bounded by `relation_pair_cap` (64),
//!   so the linear scan is cheaper than hashing and the order is not a matter of
//!   luck. That also avoids adding `indexmap` as a direct dependency to a fork
//!   kept byte-close to upstream.
//! * **`max()` returns the FIRST maximal element in Python**, while Rust's
//!   [`Iterator::max_by_key`] returns the last. Stage 1 picks the first.
//! * **`str.casefold()` is not `to_lowercase()`** — see [`casefold`].
//! * **Python `len()` counts characters.** Every length and distance below is a
//!   *character* count taken from the source text, never `str::len()`, which
//!   counts bytes. The span offsets themselves stay byte offsets, as the rest of
//!   this crate uses them: byte and character offsets order identically inside
//!   one string, so every comparison that only orders is unaffected, and the two
//!   places where a *magnitude* is compared convert.
//! * **`x is y`** in stage 3 is object identity, i.e. "the same list position",
//!   not equality. Two structurally equal edges do not cancel each other out.

use crate::boundary::{RelationEdge, RelationEndpoint};
use crate::relations::ProposedPair;
use crate::runtime::sigmoid;

// ── an insertion-ordered map, small on purpose ───────────────────────────────

/// A tiny association list with `dict` semantics: first insertion fixes the
/// position, a later write to the same key replaces the value in place.
///
/// Used instead of [`std::collections::HashMap`] because the iteration order is
/// load-bearing in three places below, and instead of `indexmap` because the
/// collections are bounded by the checkpoint's `relation_pair_cap` of 64.
#[derive(Debug, Clone)]
pub struct OrderedMap<K, V> {
    entries: Vec<(K, V)>,
}

impl<K: PartialEq, V> Default for OrderedMap<K, V> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<K: PartialEq, V> OrderedMap<K, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Inserts, or replaces the value of an existing key **without moving it**.
    pub fn insert(&mut self, key: K, value: V) {
        match self.entries.iter_mut().find(|(k, _)| *k == key) {
            Some(slot) => slot.1 = value,
            None => self.entries.push((key, value)),
        }
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.entries
            .iter_mut()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.entries.iter().map(|(_, v)| v)
    }

    pub fn into_values(self) -> impl Iterator<Item = V> {
        self.entries.into_iter().map(|(_, v)| v)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ── string normalisation ─────────────────────────────────────────────────────

/// Python's `str.casefold()`, not `str.lower()`.
///
/// The two differ wherever Unicode defines a *full* case folding that is not a
/// simple lowercase mapping — most visibly `ß`, which casefolds to `ss` while
/// lowercasing to itself. Using [`str::to_lowercase`] would leave `STRASSE` and
/// `Straße` as different semantic keys, and stage 2 would keep both edges.
///
/// The mappings that differ from `to_lowercase` are listed explicitly; every
/// other character falls through to `to_lowercase`, which already agrees with
/// the fold. The list covers the full-fold entries of `CaseFolding.txt` that can
/// appear in ordinary Latin/Greek/Cyrillic text plus the Latin ligatures; it is
/// **not** the whole of `CaseFolding.txt` (Armenian ligatures and the Greek
/// iota-subscript family are not handled). That is a documented approximation,
/// not an oversight: adding a Unicode table crate to this fork would be a larger
/// change than the gap it closes, and no parity scenario reaches those ranges.
pub fn casefold(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            // Full folds that expand to more than one character.
            '\u{00DF}' => out.push_str("ss"), // ß  LATIN SMALL LETTER SHARP S
            '\u{1E9E}' => out.push_str("ss"), // ẞ  LATIN CAPITAL LETTER SHARP S
            '\u{FB00}' => out.push_str("ff"), // ﬀ
            '\u{FB01}' => out.push_str("fi"), // ﬁ
            '\u{FB02}' => out.push_str("fl"), // ﬂ
            '\u{FB03}' => out.push_str("ffi"), // ﬃ
            '\u{FB04}' => out.push_str("ffl"), // ﬄ
            '\u{FB05}' => out.push_str("st"), // ﬅ
            '\u{FB06}' => out.push_str("st"), // ﬆ
            '\u{0149}' => out.push_str("\u{02BC}n"), // ŉ
            '\u{01F0}' => out.push_str("j\u{030C}"), // ǰ
            '\u{0390}' => out.push_str("\u{03B9}\u{0308}\u{0301}"), // ΐ
            '\u{03B0}' => out.push_str("\u{03C5}\u{0308}\u{0301}"), // ΰ
            '\u{1E96}' => out.push_str("h\u{0331}"),
            '\u{1E97}' => out.push_str("t\u{0308}"),
            '\u{1E98}' => out.push_str("w\u{030A}"),
            '\u{1E99}' => out.push_str("y\u{030A}"),
            '\u{1E9A}' => out.push_str("a\u{02BE}"),
            // Single-character folds `to_lowercase` does not perform.
            '\u{017F}' => out.push('s'), // ſ  LATIN SMALL LETTER LONG S
            '\u{00B5}' => out.push('\u{03BC}'), // µ  MICRO SIGN -> GREEK SMALL MU
            '\u{03C2}' => out.push('\u{03C3}'), // ς  FINAL SIGMA -> SIGMA
            '\u{1E9B}' => out.push('\u{1E61}'), // ẛ
            '\u{03D0}' => out.push('\u{03B2}'), // ϐ
            '\u{03D1}' => out.push('\u{03B8}'), // ϑ
            '\u{03D5}' => out.push('\u{03C6}'), // ϕ
            '\u{03D6}' => out.push('\u{03C0}'), // ϖ
            '\u{03F0}' => out.push('\u{03BA}'), // ϰ
            '\u{03F1}' => out.push('\u{03C1}'), // ϱ
            '\u{03F5}' => out.push('\u{03B5}'), // ϵ
            '\u{1FBE}' => out.push('\u{03B9}'), // ι  GREEK PROSGEGRAMMENI
            '\u{1C80}' => out.push('\u{0432}'),
            '\u{1C81}' => out.push('\u{0434}'),
            '\u{1C82}' => out.push('\u{043E}'),
            '\u{1C83}' => out.push('\u{0441}'),
            '\u{1C84}' | '\u{1C85}' => out.push('\u{0442}'),
            '\u{1C86}' => out.push('\u{044A}'),
            '\u{1C87}' => out.push('\u{0463}'),
            '\u{1C88}' => out.push('\u{A64B}'),
            other => out.extend(other.to_lowercase()),
        }
    }
    out
}

/// `" ".join(value.casefold().split())` — `engine.py:945-946`.
///
/// Whitespace-insensitive and case-insensitive; the key stage 2 groups by.
pub fn semantic_text(value: &str) -> String {
    let folded = casefold(value);
    folded.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The characters of `text` between two **byte** offsets.
///
/// Python compares character distances, so the two call sites that compare a
/// magnitude rather than an order go through this instead of subtracting byte
/// offsets. Returns 0 when the range is empty or does not lie on character
/// boundaries.
fn chars_between(text: &str, from: usize, to: usize) -> usize {
    if to <= from {
        return 0;
    }
    text.get(from..to).map(|s| s.chars().count()).unwrap_or(0)
}

// ── stage 1 — containment canonicalisation, then exact collapse ──────────────

/// Replaces every mention by the longest mention that contains it, then keeps
/// one edge per canonical `(head span, tail span)`.
///
/// This is the stage that turns `Apple` into `Apple Inc.` — the single biggest
/// source of Python's relation precision over a naive cross-product, because
/// once both surfaces are canonical the three edges `Apple|Steve Jobs`,
/// `Apple Inc.|Steve Jobs` and `Apple|Steve` collapse to one.
///
/// Faithful details, each of which changes the output if got wrong:
///
/// * the containment pool is built **per side**, from the distinct spans of that
///   side only (`engine.py:915-919`) — a head is never canonicalised against a
///   tail;
/// * `candidate.start <= start && candidate.end >= end`, so a mention always
///   contains itself and the pool is never empty;
/// * the winner is `max(length, -start)` and Python's `max` keeps the **first**
///   maximal element, where Rust's `max_by_key` keeps the last;
/// * `length` is a character count, not a byte count;
/// * the exact collapse replaces the incumbent only on a **strictly** greater
///   score, so the first edge seen wins a tie.
pub fn canonicalise_containment(edges: Vec<RelationEdge>, text: &str) -> Vec<RelationEdge> {
    let head_canonical = canonical_mentions(edges.iter().map(|e| &e.head), text);
    let tail_canonical = canonical_mentions(edges.iter().map(|e| &e.tail), text);

    let mut exact: OrderedMap<(usize, usize, usize, usize), RelationEdge> = OrderedMap::new();
    for edge in edges {
        let head = head_canonical
            .get(&(edge.head.char_start, edge.head.char_end))
            .cloned()
            .unwrap_or_else(|| edge.head.clone());
        let tail = tail_canonical
            .get(&(edge.tail.char_start, edge.tail.char_end))
            .cloned()
            .unwrap_or_else(|| edge.tail.clone());
        let key = (
            head.char_start,
            head.char_end,
            tail.char_start,
            tail.char_end,
        );
        let normalized = RelationEdge { head, tail, ..edge };
        // `if previous is None or edge["score"] > previous["score"]` — strictly
        // greater, so the first edge seen survives a tie.
        let replaces = match exact.get(&key) {
            Some(previous) => normalized.score > previous.score,
            None => true,
        };
        if replaces {
            exact.insert(key, normalized);
        }
    }
    exact.into_values().collect()
}

/// `canonical_mentions` (`engine.py:913-935`), for one side.
///
/// Public so a test can pin the `Apple` -> `Apple Inc.` mapping directly rather
/// than inferring it from an edge list.
pub fn canonical_mentions<'a>(
    endpoints: impl Iterator<Item = &'a RelationEndpoint>,
    text: &str,
) -> OrderedMap<(usize, usize), RelationEndpoint> {
    let mut mentions: OrderedMap<(usize, usize), RelationEndpoint> = OrderedMap::new();
    for endpoint in endpoints {
        mentions.insert((endpoint.char_start, endpoint.char_end), endpoint.clone());
    }

    let pool: Vec<RelationEndpoint> = mentions.values().cloned().collect();
    let mut canonical: OrderedMap<(usize, usize), RelationEndpoint> = OrderedMap::new();
    for mention in &pool {
        let (start, end) = (mention.char_start, mention.char_end);
        // `max(containing, key=(length, -start))`, first maximal element.
        let mut best: Option<&RelationEndpoint> = None;
        let mut best_key = (0usize, 0isize);
        for candidate in pool
            .iter()
            .filter(|c| c.char_start <= start && c.char_end >= end)
        {
            let key = (
                chars_between(text, candidate.char_start, candidate.char_end),
                -(candidate.char_start as isize),
            );
            if best.is_none() || key > best_key {
                best = Some(candidate);
                best_key = key;
            }
        }
        let winner = best.unwrap_or(mention).clone();
        canonical.insert((start, end), winner);
    }
    canonical
}

// ── stage 2 — one edge per pair of surface forms ─────────────────────────────

/// Collapses edges whose head and tail *read* the same, keeping the closest.
///
/// `engine.py:944-969`. The key is the case- and whitespace-normalised surface
/// pair, so two occurrences of `Microsoft ... Azure` in different sentences give
/// one edge. The survivor is chosen by `rank = (distance, -score, head start,
/// tail start)`, **smallest wins**, and an incumbent is replaced only on a
/// strictly smaller rank — so again the first entry wins a tie.
///
/// `distance` is the gap between the two mentions, zero when they overlap, and
/// it is measured in **characters**: it is compared as a magnitude between two
/// candidates, so byte offsets would order it differently on non-ASCII text.
pub fn collapse_semantic_duplicates(edges: Vec<RelationEdge>, text: &str) -> Vec<RelationEdge> {
    let mut semantic: OrderedMap<(String, String), RelationEdge> = OrderedMap::new();
    for edge in edges {
        let key = (
            semantic_text(&edge.head.text),
            semantic_text(&edge.tail.text),
        );
        let replaces = match semantic.get(&key) {
            Some(previous) => rank(&edge, text) < rank(previous, text),
            None => true,
        };
        if replaces {
            semantic.insert(key, edge);
        }
    }
    semantic.into_values().collect()
}

/// `rank()` (`engine.py:958-963`), as a totally ordered key.
///
/// `-score` is compared as an `f32` in Python; here it is mapped through
/// [`f32::total_cmp`]'s ordering by way of an ordered wrapper, which agrees with
/// Python for every finite value and cannot panic on a tie.
fn rank(edge: &RelationEdge, text: &str) -> (usize, ScoreKey, usize, usize) {
    let (hs, he) = (edge.head.char_start, edge.head.char_end);
    let (ts, te) = (edge.tail.char_start, edge.tail.char_end);
    let distance = chars_between(text, te, hs).max(chars_between(text, he, ts));
    (distance, ScoreKey(-edge.score), hs, ts)
}

/// `f32` with a total order, so a rank tuple can be compared with `<`.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ScoreKey(f32);

impl Eq for ScoreKey {}

impl PartialOrd for ScoreKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoreKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

// ── stage 3 — drop a partial argument beside a complete one ─────────────────

/// Drops an edge whose head (or tail) is a **strict token subset** of another
/// edge's, when the opposite end is identical.
///
/// `engine.py:971-994`. Where stage 1 works on spans, this works on *tokens*, so
/// it also catches the case where the fuller mention was never proposed as a
/// span containing the shorter one — `Apple` beside `Apple Inc.` when the two
/// came from different windows of the pool, for instance.
///
/// `other is edge` in Python is object identity, so an edge is compared against
/// every *other position* in the list; two structurally identical edges would
/// still be compared to each other. Index comparison reproduces that exactly.
pub fn drop_dominated_by_token_superset(edges: Vec<RelationEdge>) -> Vec<RelationEdge> {
    let tokens: Vec<(Vec<String>, Vec<String>)> = edges
        .iter()
        .map(|e| (token_set(&e.head.text), token_set(&e.tail.text)))
        .collect();

    let mut kept = Vec::with_capacity(edges.len());
    for (i, edge) in edges.iter().enumerate() {
        let (head, tail) = &tokens[i];
        let dominated = tokens
            .iter()
            .enumerate()
            .any(|(j, (other_head, other_tail))| {
                j != i
                    && ((is_strict_subset(head, other_head) && tail == other_tail)
                        || (is_strict_subset(tail, other_tail) && head == other_head))
            });
        if !dominated {
            kept.push(edge.clone());
        }
    }
    kept
}

/// The whitespace tokens of a surface form, as a sorted, de-duplicated set.
///
/// Python builds a `set`, so duplicates collapse and order is irrelevant; a
/// sorted `Vec` gives the same equality and subset answers without a hash map.
fn token_set(value: &str) -> Vec<String> {
    let mut tokens: Vec<String> = semantic_text(value)
        .split_whitespace()
        .map(str::to_string)
        .collect();
    tokens.sort();
    tokens.dedup();
    tokens
}

/// `a < b` on Python sets: every element of `a` is in `b`, and `a != b`.
fn is_strict_subset(a: &[String], b: &[String]) -> bool {
    a.len() < b.len() && a.iter().all(|t| b.contains(t))
}

// ── stage 4 — the output order ───────────────────────────────────────────────

/// `sorted(kept, key=(head start, tail start, -score))` — `engine.py:996-1003`.
///
/// Python's `sorted` is stable, and so is [`slice::sort_by`]; edges that agree on
/// all three keys keep the order stage 3 left them in.
pub fn sort_edges(mut edges: Vec<RelationEdge>) -> Vec<RelationEdge> {
    edges.sort_by(|a, b| {
        a.head
            .char_start
            .cmp(&b.head.char_start)
            .then(a.tail.char_start.cmp(&b.tail.char_start))
            .then(ScoreKey(-a.score).cmp(&ScoreKey(-b.score)))
    });
    edges
}

// ── the four stages together ────────────────────────────────────────────────

/// `_deduplicate_relation_edges` (`engine.py:900-1003`), for **one** relation
/// type.
///
/// Python calls it per relation label, from a dict keyed by label, and never
/// across labels — two different relations between the same two entities are
/// both kept. The engine reproduces that by grouping first.
///
/// The `len < 2` early return is Python's and is kept although the four stages
/// are individually no-ops on a one-element list: it is what makes the single
/// edge come back *untouched*, canonicalisation included.
pub fn deduplicate_relation_edges(edges: Vec<RelationEdge>, text: &str) -> Vec<RelationEdge> {
    if edges.len() < 2 {
        return edges;
    }
    let stage1 = canonicalise_containment(edges, text);
    let stage2 = collapse_semantic_duplicates(stage1, text);
    let stage3 = drop_dominated_by_token_superset(stage2);
    sort_edges(stage3)
}

// ── decode ──────────────────────────────────────────────────────────────────

/// Everything `_decode_relations` needs besides the logits.
///
/// Borrowed rather than owned: it is built once per window inside
/// `BoundaryEngine::extract_once` and dropped immediately.
pub struct EdgeDecoder<'a> {
    /// One name per relation type, indexed by [`ProposedPair::relation_index`].
    /// The engine puts the group's `prompt_str` here, matching
    /// [`crate::boundary::Mention::task`].
    pub relation_names: &'a [String],
    /// The window's text, in the window's own coordinate frame.
    pub text: &'a str,
    /// `ProcessedRecord::word_to_char_maps` — byte range per word.
    pub word_to_char: &'a [(usize, usize)],
    /// The window's real word count. Spans reaching past it are dropped, as
    /// `engine.py:860` does with `0 <= hs < he <= text_len`.
    pub num_words: usize,
    /// The **decode** threshold. This is the call's `threshold` (0.5 on cognee's
    /// path), i.e. `_decode_relations`'s `threshold` argument, and it is *not*
    /// `relation_argument_proposal_threshold` (0.2), which selected the pairs
    /// and has already been applied.
    pub threshold: f32,
    /// `boundary_settings.relation_temperature`, 1.0 for this checkpoint.
    pub temperature: f32,
}

impl EdgeDecoder<'_> {
    /// Thresholds the logits, resolves both endpoints to text, groups by
    /// relation type and de-duplicates each group.
    ///
    /// The output is relation-major in first-seen order, which is the order
    /// Python's `edges` dict gives `out`.
    pub fn decode(&self, pairs: &[ProposedPair], logits: &[f32]) -> Vec<RelationEdge> {
        let mut grouped: OrderedMap<usize, Vec<RelationEdge>> = OrderedMap::new();
        for (index, pair) in pairs.iter().enumerate() {
            let Some(&logit) = logits.get(index) else {
                continue;
            };
            // A pair whose relation index is out of range scores exactly 0.0 in
            // the graph (`relations.py:407`), and `sigmoid(0) = 0.5` is not
            // obviously under any threshold. Drop it rather than decode it.
            let Some(name) = self.relation_names.get(pair.relation_index) else {
                continue;
            };
            let score = sigmoid(logit / self.temperature);
            if score < self.threshold {
                continue;
            }
            let Some(head) = self.endpoint(pair.head_start, pair.head_end) else {
                continue;
            };
            let Some(tail) = self.endpoint(pair.tail_start, pair.tail_end) else {
                continue;
            };
            let edge = RelationEdge {
                relation: name.clone(),
                score,
                head,
                tail,
            };
            // `edges.setdefault(relation_type, []).append(...)`: the bucket
            // keeps the position its relation type first claimed.
            match grouped.get_mut(&pair.relation_index) {
                Some(bucket) => bucket.push(edge),
                None => grouped.insert(pair.relation_index, vec![edge]),
            }
        }

        let mut out = Vec::new();
        for bucket in grouped.into_values() {
            out.extend(deduplicate_relation_edges(bucket, self.text));
        }
        out
    }

    /// Word span -> endpoint, or `None` when the span is out of range or the
    /// surface is blank.
    ///
    /// `engine.py:856-868`: the *offsets* are the untrimmed character
    /// boundaries, and only the *text* is stripped. Keeping the untrimmed
    /// offsets matters — they are stage 1's containment key.
    fn endpoint(&self, start: usize, end: usize) -> Option<RelationEndpoint> {
        if end <= start || end > self.num_words || end > self.word_to_char.len() {
            return None;
        }
        let char_start = self.word_to_char[start].0;
        let char_end = self.word_to_char[end - 1].1;
        let slice = self.text.get(char_start..char_end)?;
        let trimmed = slice.trim();
        if trimmed.is_empty() {
            return None;
        }
        Some(RelationEndpoint {
            text: trimmed.to_string(),
            char_start,
            char_end,
            word_start: start,
            word_end: end,
        })
    }
}
