// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! Engine for the **boundary** architecture (GLiNER2.5).
//!
//! It shares nothing with the span architecture: no exhaustive span
//! enumeration, no `count_lstm`. The model proposes a pool of `(start, end)`
//! candidates **shared across all queries**, of constant size (`pool_size`,
//! typically 192), then assigns a logit to each query/candidate pair.
//!
//! ```text
//! encoder(input_ids, attention_mask) -> last_hidden_state [1,S,H]
//!   +- routed_gather(lhs, word_idx,  word_mask)  -> text_states  [1,L,H]
//!   +- routed_gather(lhs, query_idx, ones)       -> query_states [1,Q,H]
//!   +- routed_gather(lhs, cls_idx,   ones)       -> cls_states   [1,K,H]
//!
//! boundary_head_L{bucket}(text_states, text_mask, query_states, query_mask)
//!   -> cand_indices    [1,Q,C,2]   HALF-OPEN (start, end) pairs
//!   -> pair_logits     [1,Q,C]
//!   -> cand_valid      [1,Q,C]
//!   -> null_logits     [1,Q]       per-query abstention
//!   -> count_log_rates [1,Q]       expected mention count per query
//!
//! classifier(cls_states) -> logits [K]
//! ```
//!
//! ## Length buckets
//!
//! The boundary heads have a **static** `num_words`: `torch.export` specialises
//! it because the candidate-pool builder contains a Python loop over a symbolic
//! dimension. The engine therefore picks the smallest bucket that fits the text
//! and pads with `text_mask = 0`.
//!
//! Masked padding is verified to be transparent: for the same real words,
//! padding to a larger bucket yields the same candidate set and probabilities
//! to within ~5e-06, even with random noise in the padded rows.
//!
//! This costs nothing on disk — a head is under 5 MB against 1.1 GB of encoder —
//! and is an advantage at run time: static shapes are what TensorRT, QNN and
//! IOBinding prefer.

use anyhow::{Result, anyhow};
use serde::Deserialize;
use std::path::PathBuf;

use ort::session::Session;

use crate::error::GlinerError;
use crate::overlap::{OverlapPolicy, Spanned, resolve_overlaps};
use crate::processor::{SchemaTask, SchemaTransformer, TaskType};
use crate::chunker::Chunker;
use crate::chain::{Carrier, Chain, ExecutionMode, Feed, Sink};
use crate::runtime::{
    IoDType, Precision, build_session, i64_tensor, resolve_aux, resolve_fragment,
    resolve_tokenizer, sigmoid, softmax,
};

/// `boundary_manifest.json`, written by `export_boundary_v1.py`.
#[derive(Debug, Clone, Deserialize)]
pub struct BoundaryManifest {
    pub architecture: String,
    pub hidden_size: usize,
    pub pool_size: usize,
    pub length_buckets: Vec<usize>,
    pub min_bucket: usize,
    pub enable_abstention: bool,
    pub enable_count_head: bool,
    pub overlap_policy: String,
    pub max_position_embeddings: usize,
    /// `true` when the export carries `relation_scorer_{fp32,fp16,…}.onnx`.
    ///
    /// Defaulted rather than required: every export written before the relation
    /// scorer existed lacks the key, and an engine that only extracts entities
    /// must keep loading those.
    #[serde(default)]
    pub enable_relation_scorer: bool,
}

#[derive(Debug, Clone)]
pub struct BoundaryParams {
    /// Threshold on each query/candidate pair probability.
    pub threshold: f32,
    /// Overlap policy; `None` uses the one recorded in the manifest.
    pub overlap_policy: Option<OverlapPolicy>,
    /// When `true` and the model exposes abstention, a query whose
    /// `null_logit` beats its best candidate logit yields no mention.
    pub use_abstention: bool,
    pub classification_temperature: f32,
    /// Overrides the per-task `multi_label` flag carried by
    /// [`SchemaTask::Classifications`]. Leave it `None` — the schema is the
    /// right place for that decision, since a single request routinely mixes
    /// single-label and multi-label tasks.
    pub multi_label_override: Option<bool>,
    /// COGNEE-EVAL PATCH. Per-group `label -> description` pairs, folded into
    /// that group's single `prompt_str` token (the token at schema index 2) by
    /// [`SchemaTransformer::transform_with_descriptions`], exactly as Python's
    /// `SchemaTransformer._transform_schema` does. Empty by default, which
    /// reproduces the previous behaviour byte for byte.
    ///
    /// Without this the engine called `transform()` and the descriptions the
    /// crate can already build were unreachable from `BoundaryEngine`.
    pub descriptions: Vec<Vec<(String, String)>>,
}

impl Default for BoundaryParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            overlap_policy: None,
            use_abstention: true,
            classification_temperature: 1.0,
            multi_label_override: None,
            descriptions: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mention {
    pub text: String,
    /// Schema group name (e.g. `entities`, or the relation name).
    pub task: String,
    /// Field that produced the query (the label, or the `head`/`tail` role).
    pub field: String,
    pub score: f32,
    /// Byte range `[start, end)` in the original text.
    pub char_start: usize,
    pub char_end: usize,
    /// Half-open word range `[start, end)`.
    pub word_start: usize,
    pub word_end: usize,
    /// Query index, useful when reassembling relations.
    pub query_id: usize,
}

impl Spanned for Mention {
    fn start(&self) -> usize {
        self.word_start
    }
    fn end(&self) -> usize {
        self.word_end
    }
    fn score(&self) -> f32 {
        self.score
    }
}

#[derive(Debug, Clone)]
pub struct Classification {
    pub task: String,
    pub label: String,
    pub score: f32,
}

/// One end of a [`RelationEdge`].
///
/// Deliberately self-contained: the spans and the surface text travel with the
/// edge instead of pointing into [`BoundaryOutput::mentions`], because merging
/// two windows can delete a mention an edge names. See [`crate::chunker::merge`].
#[derive(Debug, Clone, PartialEq)]
pub struct RelationEndpoint {
    pub text: String,
    /// Byte range `[start, end)` in the original text, as [`Mention`] uses.
    pub char_start: usize,
    pub char_end: usize,
    /// Half-open word range `[start, end)`.
    pub word_start: usize,
    pub word_end: usize,
}

/// One decoded relation, in the coordinate frame of the text that produced it.
///
/// Relations are *intra-window* by construction: the model's relation scorer
/// indexes both endpoints against a single padded length, so a pair whose ends
/// live in different windows has no shared frame and is undefined rather than
/// merely hard. `gliner2` does the same — it scores inside the window and then
/// merges the *decoded* edges (`chunking._merge_relation_maps`). That is why
/// this type carries spans and text rather than hidden states.
#[derive(Debug, Clone, PartialEq)]
pub struct RelationEdge {
    /// The relation group's `prompt_str` (`"name: description"`), matching
    /// [`Mention::task`]; the caller maps it back to the bare name.
    pub relation: String,
    pub score: f32,
    pub head: RelationEndpoint,
    pub tail: RelationEndpoint,
}

#[derive(Debug, Clone, Default)]
pub struct BoundaryOutput {
    pub mentions: Vec<Mention>,
    /// Every label of every classification task, with its probability. Use
    /// [`BoundaryOutput::verdict`] to turn one task into the answer gliner2
    /// would give.
    pub classifications: Vec<Classification>,
    /// Expected mention count per query, when the model exposes the count head.
    pub expected_counts: Vec<f32>,
    /// Relations decoded inside the window that produced this output, ready to
    /// be shifted by [`crate::chunker::remap`] and folded by
    /// [`crate::chunker::merge`]. Nothing fills this in yet — the relation
    /// scorer is not wired — so it is empty on every path today.
    pub relations: Vec<RelationEdge>,
}

impl BoundaryOutput {
    /// The labels gliner2 would report for a classification task.
    ///
    /// Reproduces `_extract_classification_result`, including the detail that
    /// is easy to miss: in multi-label mode, **when no label clears the
    /// threshold the top-scoring one is returned anyway**. The list is never
    /// empty. Thresholding the scores yourself and keeping the empty result
    /// will silently disagree with the reference implementation.
    ///
    /// In single-label mode the argmax is returned, which is the same thing.
    pub fn verdict(&self, task: &str, threshold: f32) -> Vec<&Classification> {
        let mut rows: Vec<&Classification> =
            self.classifications.iter().filter(|c| c.task == task).collect();
        if rows.is_empty() {
            return rows;
        }
        rows.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
        });
        let over: Vec<&Classification> =
            rows.iter().copied().filter(|c| c.score >= threshold).collect();
        if over.is_empty() { vec![rows[0]] } else { over }
    }
}

#[derive(Debug, Clone)]
pub struct BoundaryConfig {
    pub models_dir: PathBuf,
    pub precision: Precision,
    pub intra_threads: usize,
    /// How intermediate tensors travel between fragments.
    pub execution: ExecutionMode,
    /// Set when the caller named a precision, so the engine does not override
    /// it with the one the execution mode would prefer.
    precision_pinned: bool,
    /// Where to fetch the export from if `models_dir` does not hold one.
    #[cfg(feature = "hub")]
    pub hub: Option<crate::hub::Model>,
    /// Loads only the heads actually needed instead of every bucket, which
    /// shortens start-up when there are many buckets.
    pub lazy_heads: bool,
}

impl BoundaryConfig {
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        let models_dir = models_dir.into();
        let precision = Precision::autodetect(&models_dir, "encoder");
        Self {
            models_dir,
            precision,
            intra_threads: 4,
            lazy_heads: true,
            execution: ExecutionMode::from_env(),
            // GLINER2_PRECISION is as explicit as calling with_precision.
            precision_pinned: std::env::var("GLINER2_PRECISION").is_ok(),
            #[cfg(feature = "hub")]
            hub: None,
        }
    }

    /// Fetches the export straight from the Hub, into the shared cache.
    ///
    /// Nothing is downloaded until [`BoundaryEngine::new`] runs, so
    /// `with_precision` still applies to what gets fetched.
    #[cfg(feature = "hub")]
    pub fn from_hub(model: crate::hub::Model) -> Self {
        Self::new(PathBuf::new()).or_download(model)
    }

    /// Names the repository to fall back to when `models_dir` holds no export.
    ///
    /// The local directory always wins: a checkout already on disk is used as
    /// it is, and the network is touched only when the export is missing.
    #[cfg(feature = "hub")]
    pub fn or_download(mut self, model: crate::hub::Model) -> Self {
        self.hub = Some(model);
        self
    }

    /// Chooses the transport between fragments.
    ///
    /// Overrides `GLINER2_EXECUTION`. The default is [`ExecutionMode::Auto`]:
    /// bound on a device provider, standard on CPU.
    pub fn with_execution(mut self, mode: ExecutionMode) -> Self {
        self.execution = mode;
        self
    }

    /// Pins the export variant.
    ///
    /// Also stops the engine choosing one from the execution mode when it has
    /// to download: an explicit choice is an instruction, not a hint.
    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self.precision_pinned = true;
        self
    }

    pub fn with_intra_threads(mut self, n: usize) -> Self {
        self.intra_threads = n;
        self
    }

    pub fn eager_heads(mut self) -> Self {
        self.lazy_heads = false;
        self
    }
}

pub struct BoundaryEngine {
    encoder: Session,
    routed_gather: Session,
    classifier: Session,
    /// Heads by bucket, ascending. `None` until first needed.
    heads: Vec<(usize, Option<Session>)>,
    /// The relation scorer. Loaded on first use like a head, and for the same
    /// reason: an entity-only run must not pay for 23 MB it never touches.
    relation_scorer: Option<Session>,

    transformer: SchemaTransformer,
    chain: Chain,
    manifest: BoundaryManifest,
    dtype: IoDType,
    dir: PathBuf,
    precision: Precision,
    intra_threads: usize,
    default_policy: OverlapPolicy,
}

impl BoundaryEngine {
    pub fn new(config: BoundaryConfig) -> Result<Self> {
        #[allow(unused_mut)]
        let mut config = config;

        // A directory that already holds the export is used untouched; only a
        // missing one reaches the network.
        #[cfg(feature = "hub")]
        if let Some(model) = config.hub {
            if resolve_aux(&config.models_dir, "boundary_manifest.json", config.precision)
                .is_none()
            {
                // Nothing on disk, so `autodetect` had no files to inspect and
                // fell back to FP32. Let the transport pick instead: a bound
                // chain wants the FP16 I/O graphs, the standard path does not.
                if !config.precision_pinned {
                    config.precision = config.execution.preferred_precision();
                }
                let (dir, got) = crate::hub::download(model, config.precision)?;
                config.models_dir = dir;
                config.precision = got;
            }
        }

        let dir = config.models_dir.clone();
        let sfx = config.precision.suffix();

        // Both layouts are accepted: flat, as the exporter writes it, and the
        // fp32_v2/ + fp16_v2/ subfolders the published gliner2 exports use.
        let manifest_path = resolve_aux(&dir, "boundary_manifest.json", config.precision)
            .ok_or_else(|| {
                // `span_rep` is the span architecture's signature fragment; if
                // it is here, this is a GLiNER2 export and the right message is
                // "wrong crate", not "missing file".
                let is_span = resolve_fragment(&dir, "span_rep", config.precision).is_some()
                    || resolve_fragment(&dir, "span_rep", Precision::Fp32).is_some();
                if is_span {
                    GlinerError::IncompleteModelDir(format!(
                        "{} holds a GLiNER2 **span** export (span_rep is present, \
                         boundary_manifest.json is not). This engine runs the \
                         boundary architecture only — use the gliner2-rs crate \
                         for this model.",
                        dir.display()
                    ))
                } else {
                    GlinerError::IncompleteModelDir(format!(
                        "boundary_manifest.json not found in {} nor in its variant \
                         subfolders; the directory does not hold a boundary export",
                        dir.display()
                    ))
                }
            })?;
        let manifest: BoundaryManifest =
            serde_json::from_slice(&std::fs::read(&manifest_path)?)?;
        if manifest.architecture != "boundary" {
            return Err(GlinerError::IncompleteModelDir(format!(
                "the manifest declares architecture '{}', expected 'boundary'",
                manifest.architecture
            ))
            .into());
        }

        let default_policy = OverlapPolicy::parse(&manifest.overlap_policy).ok_or_else(|| {
            anyhow!("overlap_policy '{}' not recognised", manifest.overlap_policy)
        })?;

        let tok_path = resolve_tokenizer(&dir, config.precision).ok_or_else(|| {
            GlinerError::IncompleteModelDir(format!(
                "tokenizer.json not found in {} nor in its variant subfolders",
                dir.display()
            ))
        })?;
        let transformer = SchemaTransformer::from_tokenizer_file(&tok_path)?;

        let load = |stem: &str| -> Result<Session> {
            let path = resolve_fragment(&dir, stem, config.precision).ok_or_else(|| {
                GlinerError::IncompleteModelDir(format!(
                    "fragment '{stem}{sfx}' not found in {} (looked in the directory \
                     itself and in {}/)",
                    dir.display(),
                    config.precision.legacy_subdir(),
                ))
            })?;
            build_session(&path, config.intra_threads)
        };

        let mut buckets: Vec<usize> = manifest.length_buckets.clone();
        buckets.sort_unstable();

        let mut heads: Vec<(usize, Option<Session>)> = Vec::with_capacity(buckets.len());
        for b in &buckets {
            let session = if config.lazy_heads {
                None
            } else {
                Some(load(&format!("boundary_head_L{b}"))?)
            };
            heads.push((*b, session));
        }

        Ok(Self {
            chain: Chain::new(config.execution, config.precision.io_dtype())?,
            encoder: load("encoder")?,
            routed_gather: load("routed_gather")?,
            classifier: load("classifier")?,
            heads,
            relation_scorer: None,
            transformer,
            manifest,
            dtype: config.precision.io_dtype(),
            dir,
            precision: config.precision,
            intra_threads: config.intra_threads,
            default_policy,
        })
    }

    pub fn manifest(&self) -> &BoundaryManifest {
        &self.manifest
    }

    /// Extracts over a document longer than the model's largest length bucket.
    ///
    /// [`extract`](Self::extract) refuses text above the ceiling rather than
    /// truncating it, which is right for one call and useless for a document.
    /// This splits the text into overlapping word windows, runs each, shifts
    /// the offsets back onto the original, and merges the duplicates the
    /// overlap produces.
    ///
    /// Text that fits in one window takes the single-call path, so this is safe
    /// to use unconditionally — there is no penalty for short input.
    ///
    /// See [`chunker`](crate::chunker) for what merging can and cannot recover.
    pub fn extract_long(&mut self, text: &str, tasks: &[SchemaTask]) -> Result<BoundaryOutput> {
        self.extract_long_with(text, tasks, &BoundaryParams::default(), Chunker::default())
    }

    /// [`extract_long`](Self::extract_long) with the window geometry spelled out.
    pub fn extract_long_with(
        &mut self,
        text: &str,
        tasks: &[SchemaTask],
        params: &BoundaryParams,
        chunker: Chunker,
    ) -> Result<BoundaryOutput> {
        let chunks = chunker.split(text)?;
        if chunks.len() <= 1 {
            return self.extract_with(text, tasks, params);
        }
        let mut parts = Vec::with_capacity(chunks.len());
        for chunk in &chunks {
            let mut part = self.extract_with(chunk.slice(text), tasks, params)?;
            crate::chunker::remap(&mut part, chunk, text);
            parts.push(part);
        }
        Ok(crate::chunker::merge(parts))
    }

    pub fn extract(&mut self, text: &str, tasks: &[SchemaTask]) -> Result<BoundaryOutput> {
        self.extract_with(text, tasks, &BoundaryParams::default())
    }

    /// Runs the chain, and on a device allocation failure runs it again on the
    /// standard path.
    ///
    /// Binding holds every intermediate on the device at once, so it is the
    /// first thing to give way on a long input. Failing the call there would be
    /// the wrong answer: the standard path releases each tensor as soon as the
    /// next fragment has consumed it and will very often succeed on exactly the
    /// input that broke binding. The engine stays on the standard path
    /// afterwards rather than paying the same failure on every call.
    pub fn extract_with(
        &mut self,
        text: &str,
        tasks: &[SchemaTask],
        params: &BoundaryParams,
    ) -> Result<BoundaryOutput> {
        match self.extract_once(text, tasks, params) {
            Err(e)
                if self.chain.mode() == ExecutionMode::IoBinding
                    && matches!(
                        e.downcast_ref::<GlinerError>(),
                        Some(GlinerError::OomDeviceBinding(_) | GlinerError::BindingNotSupported(_))
                    ) =>
            {
                eprintln!("[gliner25] {e}; continuing on the standard path");
                self.chain.fall_back();
                self.extract_once(text, tasks, params)
            }
            other => other,
        }
    }

    fn extract_once(
        &mut self,
        text: &str,
        tasks: &[SchemaTask],
        params: &BoundaryParams,
    ) -> Result<BoundaryOutput> {
        // COGNEE-EVAL PATCH: was `self.transformer.transform(text, tasks)?`,
        // which hard-coded an empty description list and made
        // `transform_with_descriptions` dead code.
        let record =
            self.transformer
                .transform_with_descriptions(text, tasks, &params.descriptions)?;
        let num_words = record.num_words();
        if num_words == 0 {
            return Ok(BoundaryOutput::default());
        }

        let bucket = self.pick_bucket(num_words)?;
        let hidden_size = self.manifest.hidden_size;
        let seq = record.input_ids.len() as i64;

        // ── 1. encoder ────────────────────────────────────────────────────
        let hidden = {
            self.chain
                .run(
                    &mut self.encoder,
                    &[
                        Feed::Owned(i64_tensor(vec![1, seq], record.input_ids.clone())?),
                        Feed::Owned(i64_tensor(vec![1, seq], record.attention_mask.clone())?),
                    ],
                    // consumed three times by routed_gather: text, queries, choices
                    &[Sink::Device],
                )?
                .remove(0)
        };

        // ── 2. routing: text padded to the bucket, queries, choices ───────
        let mut word_idx = record.word_first_positions();
        let mut word_mask = vec![1i64; num_words];
        word_idx.resize(bucket, 0);
        word_mask.resize(bucket, 0);

        let text_states = self.gather(&hidden, seq, hidden_size, &word_idx, &word_mask)?;

        let (query_idx, query_specs) = record.query_markers();
        let num_queries = query_idx.len();
        if num_queries == 0 {
            // nothing to extract: only classifications remain
            let mut out = BoundaryOutput::default();
            self.run_classifications(&record, &hidden, seq, hidden_size, params, &mut out)?;
            return Ok(out);
        }
        let query_mask = vec![1i64; num_queries];
        let query_states = self.gather(&hidden, seq, hidden_size, &query_idx, &query_mask)?;

        // ── 3. boundary head ──────────────────────────────────────────────
        // `head_for` takes `&mut self`, so fields needed inside the block must
        // be copied out first or they stay locked by the borrow.
        let slot = self.ensure_head(bucket)?;
        let (cand_indices, pair_logits, cand_valid, null_logits, count_log_rates) = {
            // `chain` and the head are separate fields, so they can be borrowed
            // together; a method handing back `&mut Session` would borrow all of
            // `self` and lock the chain out.
            let chain = &self.chain;
            let head = self.heads[slot]
                .1
                .as_mut()
                .expect("ensure_head just loaded it");
            let mut out = chain.run(
                head,
                &[
                    Feed::Carried(&text_states, vec![1, bucket as i64, hidden_size as i64]),
                    Feed::Owned(i64_tensor(vec![1, bucket as i64], word_mask.clone())?),
                    Feed::Carried(
                        &query_states,
                        vec![1, num_queries as i64, hidden_size as i64],
                    ),
                    Feed::Owned(i64_tensor(vec![1, num_queries as i64], query_mask)?),
                ],
                // every head output is decoded on the host
                &[
                    Sink::HostI64,
                    Sink::Host,
                    Sink::HostBool,
                    Sink::Host,
                    Sink::Host,
                ],
            )?;
            let count_log_rates = out.remove(4).host(self.dtype)?;
            let null_logits = out.remove(3).host(self.dtype)?;
            let cand_valid = out.remove(2).host_bool()?;
            let pair_logits = out.remove(1).host(self.dtype)?;
            let cand_indices = out.remove(0).host_i64()?;
            (
                cand_indices,
                pair_logits,
                cand_valid,
                null_logits,
                count_log_rates,
            )
        };

        // ── 4. decoding ───────────────────────────────────────────────────
        let policy = params.overlap_policy.unwrap_or(self.default_policy);
        let c = self.manifest.pool_size;
        let mut output = BoundaryOutput {
            expected_counts: count_log_rates.iter().map(|r| r.exp()).collect(),
            ..Default::default()
        };

        for (q, &(group, role)) in query_specs.iter().enumerate() {
            let task = &record.tasks[group];

            // abstention: if the null logit beats the best one, the query stays silent
            if params.use_abstention && self.manifest.enable_abstention {
                let best = (0..c)
                    .filter(|&i| cand_valid[q * c + i])
                    .map(|i| pair_logits[q * c + i])
                    .fold(f32::NEG_INFINITY, f32::max);
                if null_logits[q] > best {
                    continue;
                }
            }

            let mut candidates: Vec<Mention> = Vec::new();
            for i in 0..c {
                if !cand_valid[q * c + i] {
                    continue;
                }
                let score = sigmoid(pair_logits[q * c + i]);
                if score < params.threshold {
                    continue;
                }
                // indices are half-open: [start, end)
                let start = cand_indices[(q * c + i) * 2] as usize;
                let end = cand_indices[(q * c + i) * 2 + 1] as usize;
                if end <= start || end > num_words {
                    // drop candidates falling inside the padding
                    continue;
                }
                let (cs, _) = record.word_to_char_maps[start];
                let (_, ce) = record.word_to_char_maps[end - 1];
                candidates.push(Mention {
                    text: text[cs..ce].to_string(),
                    task: task.task_name.clone(),
                    field: task.labels[role].clone(),
                    score,
                    char_start: cs,
                    char_end: ce,
                    word_start: start,
                    word_end: end,
                    query_id: q,
                });
            }

            output.mentions.extend(resolve_overlaps(&candidates, policy));
        }

        // stable global ranking: descending confidence, then start, end, field
        output.mentions.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.word_start.cmp(&b.word_start))
                .then(a.word_end.cmp(&b.word_end))
                .then(a.field.cmp(&b.field))
        });

        self.run_classifications(&record, &hidden, seq, hidden_size, params, &mut output)?;
        Ok(output)
    }

    fn run_classifications(
        &mut self,
        record: &crate::processor::ProcessedRecord,
        hidden: &Carrier,
        seq: i64,
        hidden_size: usize,
        params: &BoundaryParams,
        output: &mut BoundaryOutput,
    ) -> Result<()> {
        let (cls_idx, cls_specs) = record.cls_markers();
        if cls_idx.is_empty() {
            return Ok(());
        }
        let mask = vec![1i64; cls_idx.len()];
        let cls_states = self.gather(hidden, seq, hidden_size, &cls_idx, &mask)?;

        let logits = {
            self.chain
                .run(
                    &mut self.classifier,
                    &[Feed::Carried(
                        &cls_states,
                        vec![cls_idx.len() as i64, hidden_size as i64],
                    )],
                    &[Sink::Host],
                )?
                .remove(0)
                .host(self.dtype)?
        };

        // logits must be normalised per group, not across every choice at once
        let mut by_group: std::collections::BTreeMap<usize, Vec<(usize, usize)>> =
            Default::default();
        for (flat, &(group, choice)) in cls_specs.iter().enumerate() {
            by_group.entry(group).or_default().push((flat, choice));
        }

        for (group, entries) in by_group {
            let task = &record.tasks[group];
            let scaled: Vec<f32> = entries
                .iter()
                .map(|&(flat, _)| logits[flat] / params.classification_temperature)
                .collect();
            let multi_label = params.multi_label_override.unwrap_or(task.multi_label);
            let probs = if multi_label {
                scaled.iter().copied().map(sigmoid).collect::<Vec<_>>()
            } else {
                softmax(&scaled)
            };
            for (&(_, choice), score) in entries.iter().zip(probs) {
                output.classifications.push(Classification {
                    task: task.task_name.clone(),
                    label: task.labels[choice].clone(),
                    score,
                });
            }
        }
        Ok(())
    }

    /// Gathers the states at `indices` out of the encoder output.
    ///
    /// Takes and returns a [`Carrier`] rather than `&[f32]`: when the chain is
    /// bound, the encoder output never left the device and this is where that
    /// pays — it is called three times per sentence, on the largest tensor in
    /// the pipeline.
    fn gather(
        &mut self,
        hidden: &Carrier,
        seq: i64,
        hidden_size: usize,
        indices: &[i64],
        mask: &[i64],
    ) -> Result<Carrier> {
        let n = indices.len() as i64;
        let h = Feed::Carried(hidden, vec![1, seq, hidden_size as i64]);
        Ok(self
            .chain
            .run(
                &mut self.routed_gather,
                &[
                    h,
                    Feed::Owned(i64_tensor(vec![1, n], indices.to_vec())?),
                    Feed::Owned(i64_tensor(vec![1, n], mask.to_vec())?),
                ],
                &[Sink::Device],
            )?
            .remove(0))
    }

    /// Smallest bucket that fits `num_words`.
    fn pick_bucket(&self, num_words: usize) -> Result<usize> {
        let max_bucket = self.heads.last().map(|(b, _)| *b).unwrap_or(0);
        self.heads
            .iter()
            .map(|(b, _)| *b)
            .find(|&b| b >= num_words)
            .ok_or_else(|| GlinerError::NoLengthBucket { words: num_words, max_bucket }.into())
    }

    /// Returns the bucket's head, loading it on demand when `lazy_heads`.
    /// Loads the head for `bucket` if it is not loaded yet and returns its slot.
    ///
    /// Returning an index rather than `&mut Session` is what lets the caller
    /// borrow the head and `self.chain` at the same time: as separate fields
    /// that is fine, through a method returning a reference it is not.
    fn ensure_head(&mut self, bucket: usize) -> Result<usize> {
        let slot = self
            .heads
            .iter()
            .position(|(b, _)| *b == bucket)
            .ok_or_else(|| anyhow!("bucket {bucket} not present"))?;
        if self.heads[slot].1.is_none() {
            let stem = format!("boundary_head_L{bucket}");
            let path = resolve_fragment(&self.dir, &stem, self.precision).ok_or_else(|| {
                GlinerError::IncompleteModelDir(format!(
                    "head for bucket {bucket} not found in {} nor in {}/",
                    self.dir.display(),
                    self.precision.legacy_subdir(),
                ))
            })?;
            self.heads[slot].1 = Some(build_session(&path, self.intra_threads)?);
        }
        Ok(slot)
    }

    /// The transport actually in force, after `Auto` was resolved and after any
    /// fallback a device OOM forced.
    pub fn execution(&self) -> ExecutionMode {
        self.chain.mode()
    }
}

/// Inputs to the `relation_scorer` fragment, for a single window.
///
/// Batch is 1, as in every other fragment: `L = padded_words`,
/// `R = relations`, `P = pairs.len()`. The graph declares all three as free
/// symbolic dimensions — the relation scorer has **no length buckets**, unlike
/// the boundary head.
pub struct RelationScoreInputs<'a> {
    /// `[1, padded_words, hidden_size]`, exactly as `routed_gather` produced
    /// it. The padded rows are **zero**: `routed_gather` returns
    /// `states * mask`, so everything past the word count is 0.0.
    pub text_states: &'a Carrier,
    /// Dim 1 of `text_states` — the length bucket the window was padded to
    /// (64/128/256/512), *not* the word count.
    pub padded_words: usize,
    /// `[relations * hidden_size]`, row-major: the **head** role's query state
    /// for each relation type, in the order `pairs`'s `relation_index` uses.
    pub relation_query_head: &'a [f32],
    /// `[relations * hidden_size]` — the **tail** role's half. The graph
    /// concatenates the two halves itself into the
    /// `relation_query_dim = 1536` state (`model.py:1397-1401`), so keep them
    /// separate here rather than pre-concatenating.
    pub relation_query_tail: &'a [f32],
    /// `R`. Pairs whose `relation_index` falls outside `0..R` score exactly
    /// `0.0` — not `-inf` (`relations.py:407`).
    pub relations: usize,
    /// The proposal set, in [`generate_pairs`](crate::relations::generate_pairs)
    /// order, already compacted. **Scoring is a no-op when this is empty** —
    /// see the `P = 0` guard in [`BoundaryEngine::score_relation_pairs`].
    pub pairs: &'a [crate::relations::ProposedPair],
    /// The denominator of `dist = |tail_start - head_start| / text_len`
    /// (`relations.py:374`), lifted out of the graph because `float()` on a
    /// `SymInt` would have baked the tracing length in.
    ///
    /// **Feed the window's own word count, not [`Self::padded_words`].** The
    /// reasoning, and the Python measurement behind it, are at the call site in
    /// [`BoundaryEngine::score_relation_pairs`]; feeding the bucket instead
    /// shifts every relation logit by ~1e-01 while every existing test stays
    /// green.
    pub text_len: usize,
}

impl BoundaryEngine {
    /// Scores proposed relation pairs with the `relation_scorer` fragment.
    ///
    /// Returns one **raw logit** per pair, in `inputs.pairs` order. Decoding is
    /// deliberately not done here: `sigmoid(logit / relation_temperature)`, the
    /// 0.5 abstention threshold — which is a different knob from the 0.2
    /// argument-proposal threshold the pairs were selected with — and edge
    /// de-duplication all belong to the caller.
    ///
    /// ## The `P = 0` guard
    ///
    /// An empty proposal set returns `Ok(vec![])` **without touching the
    /// session**. That is not an optimisation. `num_pairs >= 1` was baked into
    /// the graph at tracing time, and onnxruntime fails inside `node_add_163`
    /// ("Can broadcast 0 by 0 or 1. 768 is invalid") when fed `P = 0`; Python
    /// never reaches the model either, early-returning at
    /// `boundary/engine.py:818-819` and again at `relations.py:335-336`.
    ///
    /// The guard is on the **compacted pair list**, i.e. after `same_span` and
    /// after compaction, and it has to be: a guard on the *argument pool*
    /// answers the same way on a pool that was empty to begin with, and the
    /// wrong way when a head and a tail both survive selection and `same_span`
    /// kills the only pair they could form. Both routes to `P = 0` are in the
    /// fixture, and only the second one distinguishes the placements.
    ///
    /// ## Precision
    ///
    /// The fragment follows the engine's precision like every other one. The
    /// fp16 margin against the 2e-2 probability tolerance **narrows with the
    /// padded length** — 20.6× at `L = 64`, 18.2× at `L = 128`, 8.6× at
    /// `L = 512` — so an export run at the 512 bucket has the least headroom.
    /// (Span depth does not order the error; only length does.)
    pub fn score_relation_pairs(&mut self, inputs: RelationScoreInputs<'_>) -> Result<Vec<f32>> {
        // ── the P = 0 guard, before anything else ─────────────────────────
        if inputs.pairs.is_empty() {
            return Ok(Vec::new());
        }

        // `relations.py:341-343`: with no relation queries every pair scores
        // 0.0 rather than running the model. The graph cannot run `R = 0`
        // either — `safe_relation_indices` is inlined as `clamp(0, R - 1)`.
        if inputs.relations == 0 {
            return Ok(vec![0.0; inputs.pairs.len()]);
        }

        let hidden = self.manifest.hidden_size;
        let want = inputs.relations * hidden;
        if inputs.relation_query_head.len() != want || inputs.relation_query_tail.len() != want {
            return Err(anyhow!(
                "relation query states must hold {want} floats each ({} relations x {hidden} \
                 hidden), got head {} and tail {}",
                inputs.relations,
                inputs.relation_query_head.len(),
                inputs.relation_query_tail.len(),
            ));
        }
        if inputs.text_len == 0 || inputs.text_len > inputs.padded_words {
            return Err(anyhow!(
                "text_len {} is not a word count inside the padded length {} — it is the \
                 window's own word count, not the bucket and not an arbitrary number",
                inputs.text_len,
                inputs.padded_words,
            ));
        }

        let p = inputs.pairs.len();
        let mut relation_index = Vec::with_capacity(p);
        let mut head_start = Vec::with_capacity(p);
        let mut head_end = Vec::with_capacity(p);
        let mut tail_start = Vec::with_capacity(p);
        let mut tail_end = Vec::with_capacity(p);
        for pair in inputs.pairs {
            relation_index.push(pair.relation_index as i64);
            head_start.push(pair.head_start as i64);
            head_end.push(pair.head_end as i64);
            tail_start.push(pair.tail_start as i64);
            tail_end.push(pair.tail_end as i64);
        }

        // ── why `text_len` is the WORD COUNT and not `padded_words` ───────
        //
        // `text_len` is the one channel through which padding reaches a
        // relation score. Everything else in the fragment is padding-blind:
        // `routed_gather` zeroes the padded rows, and every span is bounded by
        // the word count, so no `prefix[end] - prefix[start]` in the biaffine
        // branch ever straddles padding. Measured, not assumed — states padded
        // to 97 rows with `text_len = 41` reproduce PyTorch-at-41 to 1.2e-06,
        // while `text_len = 97` on the same states moves the logits by 7.8e-02.
        //
        // Python takes the denominator from `boundary_states.shape[1]`
        // (`relations.py:355`), which is `pad_sequence(..., batch_first=True)`
        // over the collated batch (`model.py:1433` -> `_pad_states`) — the
        // batch maximum, not a bucket. Measured on cognee's own call
        // (`batch_extract_long(..., batch_size=16, chunk_size=384,
        // chunk_overlap=64)`) over the four parity scenarios, by spying on
        // `_decode_relations`:
        //
        //     scenario    batch  padded L  this window's words
        //     short         1       18            18
        //     medium        1       81            81
        //     long          1      384           384
        //     very_long     3      384      384 / 384 / 235
        //
        // So Python's denominator equals the window's own word count in five of
        // the six windows, and differs only for the short tail window of a
        // multi-chunk document, where it inherits 384 from its batch-mates.
        //
        // Rust pads to a 64/128/256/512 bucket, which matches Python in **none**
        // of those six windows (64/128/512/512/512/256). Feeding the bucket
        // would therefore be wrong everywhere; feeding the word count is right
        // wherever Python is effectively at batch 1, which is every
        // single-window document and every full window of a long one.
        //
        // The remaining alternative — reproducing Python's batch maximum — was
        // rejected: it makes a window's relation scores depend on which other
        // windows happen to share its batch, which is an artifact of Python's
        // collation rather than a property of the model, and this engine runs
        // one window per session call.
        let text_len = inputs.text_len as f32;

        self.ensure_relation_scorer()?;
        let pair_dim = vec![p as i64];
        let query_dim = vec![1, inputs.relations as i64, hidden as i64];
        let logits = {
            // `chain` and the session are separate fields, so they can be
            // borrowed together — the same reason `extract_once` copies the
            // head out by slot instead of through a method.
            let chain = &self.chain;
            let dtype = self.dtype;
            let session = self
                .relation_scorer
                .as_mut()
                .expect("ensure_relation_scorer just loaded it");
            let mut out = chain.run(
                session,
                &[
                    Feed::Carried(
                        inputs.text_states,
                        vec![1, inputs.padded_words as i64, hidden as i64],
                    ),
                    Feed::Owned(crate::runtime::float_tensor(
                        dtype,
                        query_dim.clone(),
                        inputs.relation_query_head.to_vec(),
                    )?),
                    Feed::Owned(crate::runtime::float_tensor(
                        dtype,
                        query_dim,
                        inputs.relation_query_tail.to_vec(),
                    )?),
                    Feed::Owned(i64_tensor(pair_dim.clone(), relation_index)?),
                    Feed::Owned(i64_tensor(pair_dim.clone(), head_start)?),
                    Feed::Owned(i64_tensor(pair_dim.clone(), head_end)?),
                    Feed::Owned(i64_tensor(pair_dim.clone(), tail_start)?),
                    Feed::Owned(i64_tensor(pair_dim, tail_end)?),
                    Feed::Owned(crate::runtime::float_tensor(
                        dtype,
                        vec![1],
                        vec![text_len],
                    )?),
                ],
                &[Sink::Host],
            )?;
            out.remove(0).host(dtype)?
        };

        if logits.len() != p {
            return Err(anyhow!(
                "relation_scorer returned {} logits for {p} pairs",
                logits.len()
            ));
        }
        Ok(logits)
    }

    /// Loads the relation scorer if it is not loaded yet.
    ///
    /// Separate from [`Self::score_relation_pairs`] so the `P = 0` guard can
    /// run before it: an empty proposal set must cost nothing, including on an
    /// export that never shipped the fragment.
    fn ensure_relation_scorer(&mut self) -> Result<()> {
        if self.relation_scorer.is_some() {
            return Ok(());
        }
        if !self.manifest.enable_relation_scorer {
            return Err(GlinerError::IncompleteModelDir(format!(
                "the manifest in {} does not declare enable_relation_scorer, so this export \
                 predates the relation scorer and cannot score relation pairs",
                self.dir.display(),
            ))
            .into());
        }
        let path =
            resolve_fragment(&self.dir, "relation_scorer", self.precision).ok_or_else(|| {
                GlinerError::IncompleteModelDir(format!(
                    "fragment 'relation_scorer{}' not found in {} nor in {}/",
                    self.precision.suffix(),
                    self.dir.display(),
                    self.precision.legacy_subdir(),
                ))
            })?;
        self.relation_scorer = Some(build_session(&path, self.intra_threads)?);
        Ok(())
    }
}

/// Relations are reassembled by grouping mentions per schema group: the first
/// role is the head, the second the tail.
pub fn pair_relations(mentions: &[Mention], tasks: &[SchemaTask]) -> Vec<(Mention, Mention, String)> {
    let mut pairs = Vec::new();
    for task in tasks {
        let SchemaTask::Relations(name, roles) = task else { continue };
        if roles.len() < 2 {
            continue;
        }
        let heads: Vec<&Mention> = mentions
            .iter()
            .filter(|m| m.task == *name && m.field == roles[0])
            .collect();
        let tails: Vec<&Mention> = mentions
            .iter()
            .filter(|m| m.task == *name && m.field == roles[1])
            .collect();
        for h in &heads {
            for t in &tails {
                if h.word_start == t.word_start && h.word_end == t.word_end {
                    continue;
                }
                pairs.push(((*h).clone(), (*t).clone(), name.clone()));
            }
        }
    }
    pairs
}

#[allow(unused)]
fn _assert_task_type_used(t: TaskType) -> bool {
    matches!(t, TaskType::Relations)
}
