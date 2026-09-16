"""
GLiNER2.5 *boundary* ONNX Exporter v1
=====================================
Exports a checkpoint with `"architecture": "boundary"` (BoundaryExtractor,
e.g. `gliner2.5-multi-v1`) into ONNX fragments runnable by the `gliner25` crate.

The boundary architecture has nothing in common with the span one: no
`span_rep`, no `count_lstm`, no exhaustive span enumeration. Instead:

    encoder (mDeBERTa-v3)
        -> token_embeddings [1, seq_len, H]
        |
        +- routed_gather(indices, mask)   (token_pooling="first")
        |     -> text_states  [1, L, H]     one row per word of the text
        |     -> query_states [1, Q, H]     one row per [E]/[C]/[R] marker
        |     -> cls_states   [1, K, H]     one row per classification choice
        |
        +- boundary_head(text_states, text_mask, query_states, query_mask)
        |     -> cand_indices    [1, Q, C, 2]  HALF-OPEN (start, end) pairs
        |     -> pair_logits     [1, Q, C]     query x candidate logits
        |     -> cand_valid      [1, Q, C]
        |     -> null_logits     [1, Q]        per-query abstention
        |     -> count_log_rates [1, Q]        expected count per query
        |
        +- classifier(cls_states) -> logits [K]

C is constant (`pool_size`, typically 192): the candidate pool is shared across
queries and has a fixed size. Decoding - sigmoid, per-query threshold, overlap
policy, ranking - is left to the caller.


Why the length buckets
----------------------
`torch.export` specialises `num_words` to a constant: the candidate-pool
builder contains a Python loop over a symbolic dimension, and no `export_mode`
removes it on this branch. It is not something a caller can work around.

The boundary head weighs under 1 MB, though (the encoder is 99.9% of the
parameters), so one copy per length bucket is exported and the runtime picks
the smallest bucket that fits the text, padding with `text_mask=0`.

Masked padding is verified to be equivalent: for the same real words, padding
to a larger bucket - even with noise in the padded rows - yields the **exact
same candidate set** and logits to within ~5e-06. See `verify_parity.py`.

A welcome side effect: static shapes in L and C make the graph ideal for
IOBinding, TensorRT and QNN, all of which degrade with dynamic shapes.

The smallest bucket is 32 because `select_top_boundaries` computes
`k = min(pool_boundary_top_k, n_boundaries)`: below that threshold the graph
would be traced with a reduced `k` and would no longer hold for longer texts.


Stable sort
-----------
The pool uses `torch.sort(..., stable=True)`; ONNX has no stable Sort and the
`aten.sort.stable` operator has no translation. `stable=True` is stripped
during export. Exact score ties happen in practice only between positions
masked to `MASK_LOGIT`, which `cand_valid` discards anyway; measured parity
against PyTorch is ~1e-05.


The relation scorer has no buckets
----------------------------------
`SparseRelationScorer.forward` (`gliner2/models/boundary/relations.py:327`)
contains no Python loop over a symbolic dimension, so nothing forces
specialisation: one graph serves every `L` (words), `R` (relation types) and
`P` (proposed pairs).

The one trap is `dist = |delta| / float(max(length, 1))` (`relations.py:374`),
where `length` is `boundary_states.shape[1]`. `float()` on a `SymInt` freezes
the tracing length into the graph while `L` still *looks* dynamic, so the
denominator is lifted to an explicit `text_len` input and `_assert_no_traced_length`
fails the export if any constant equal to the tracing length survives.

`text_len` is the **padded** length of `text_states`, not the word count, and
nothing masks it — see `notes.relation_text_len` in the manifest before wiring
a runtime to this fragment.


Usage:
    python export_boundary_v1.py \
        --model_path fastino/gliner2.5-multi-v1 \
        --out_dir models/gliner2.5-multi-v1-onnx

    # custom buckets
    python export_boundary_v1.py --model_path ... --out_dir ... \
        --buckets 64,128,256,512

    # re-export one fragment into an existing directory
    python export_boundary_v1.py --model_path ... --out_dir ... \
        --only relation_scorer
"""

from __future__ import annotations

import argparse
import contextlib
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from gliner2 import AutoExtractor

DEFAULT_BUCKETS = (64, 128, 256, 512)
MIN_BUCKET = 32
OPSET = 18  # the dynamo path cannot downgrade to 17 (Squeeze axes-as-input)


# ─────────────────────────────────────────────────────────────────────────────
# Export-time patches
# ─────────────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _unstable_sort():
    """Strips `stable=True` from torch.sort/argsort for the duration of the export."""
    _sort, _argsort = torch.sort, torch.argsort

    def sort(*args, **kwargs):
        kwargs.pop("stable", None)
        return _sort(*args, **kwargs)

    def argsort(*args, **kwargs):
        kwargs.pop("stable", None)
        return _argsort(*args, **kwargs)

    torch.sort, torch.argsort = sort, argsort
    try:
        yield
    finally:
        torch.sort, torch.argsort = _sort, _argsort


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 1 – Encoder
# ─────────────────────────────────────────────────────────────────────────────
class EncoderWrapper(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        input_ids: torch.Tensor,        # [1, seq_len]  int64
        attention_mask: torch.Tensor,   # [1, seq_len]  int64
    ) -> torch.Tensor:                  # [1, seq_len, H]
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 2 – RoutedGather
# Mirrors `gather_routed` in BoundaryExtractorModel._encode_core: takes the
# first sub-token of each slot (token_pooling="first") and zeroes the masked
# rows. A single graph serves text / query / cls.
# ─────────────────────────────────────────────────────────────────────────────
class RoutedGatherWrapper(nn.Module):
    def forward(
        self,
        hidden_state: torch.Tensor,  # [1, seq_len, H]
        indices: torch.Tensor,       # [1, S]  int64
        mask: torch.Tensor,          # [1, S]  int64 (0/1)
    ) -> torch.Tensor:               # [1, S, H]
        h = hidden_state.shape[-1]
        safe = indices.clamp(0, hidden_state.shape[1] - 1)
        states = hidden_state.gather(1, safe.unsqueeze(-1).expand(-1, -1, h))
        return states * mask.unsqueeze(-1).to(states.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 3 – BoundaryHead
# Returns flat tensors instead of ExtractorOutput/CandidateTensorBatch.
# ─────────────────────────────────────────────────────────────────────────────
class BoundaryHeadWrapper(nn.Module):
    def __init__(self, head: nn.Module):
        super().__init__()
        self.head = head

    def forward(
        self,
        text_states: torch.Tensor,   # [1, L, H]
        text_mask: torch.Tensor,     # [1, L]  int64
        query_states: torch.Tensor,  # [1, Q, H]
        query_mask: torch.Tensor,    # [1, Q]  int64
    ):
        out = self.head(
            text_states, text_mask.bool(),
            query_states, query_mask.bool(),
            targets=None,
            return_candidates=True,
            collect_diagnostics=False,
        )
        c = out.candidates
        return (
            c.indices,        # [1, Q, C, 2]  int64
            c.pair_logits,    # [1, Q, C]
            c.valid_mask,     # [1, Q, C]  bool
            out.null_logits,       # [1, Q]
            out.count_log_rates,   # [1, Q]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 4 – Classifier
# ─────────────────────────────────────────────────────────────────────────────
class ClassifierWrapper(nn.Module):
    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.classifier = classifier

    def forward(self, choice_states: torch.Tensor) -> torch.Tensor:
        # [K, H] -> [K]
        return self.classifier(choice_states).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper 5 – SparseRelationScorer
#
# Mirrors `SparseRelationScorer.forward` (gliner2/models/boundary/relations.py
# :327-407) with the batch dimension fixed to 1, which is what the decode path
# feeds anyway (`boundary/engine.py:826` slices `core["text_states"][i:i+1]`).
# With B = 1 `batch_index` is all-zero, so `batch_valid` is vacuously true and
# the `b` advanced index collapses to a plain `index_select` on dim 0.
#
# Three deliberate deviations from the reference forward, all load-bearing:
#
#   1. `relation_query_states` arrives SPLIT into its head and tail halves and
#      is concatenated in-graph. The engine builds that tensor as
#      `cat(query_states[head_id], query_states[tail_id])` when
#      `directional_relation_states=true` (`boundary/model.py:1397-1401`).
#      Both halves are rows of `routed_gather`'s output, i.e. already device
#      tensors under IOBinding; concatenating them on the host would force a
#      device -> host -> device round trip per call.
#
#   2. The distance denominator is an EXPLICIT INPUT (`text_len`). The
#      reference writes
#
#          dist = (delta.abs() / float(max(length, 1))).unsqueeze(-1)
#                                                            relations.py:374
#
#      with `length = boundary_states.shape[1]` (relations.py:355). Under
#      `torch.export` that `length` is a `SymInt`, and `float()` on a `SymInt`
#      materialises a plain Python constant: the exported graph would divide by
#      whichever `L` the tracing example happened to use, while still declaring
#      `L` dynamic. Lifting it to an input keeps `L` genuinely free — and makes
#      the value the runtime must feed an explicit, documented decision. See
#      `notes.relation_text_len` in the manifest: it is the PADDED length, not
#      the word count.
#
#   3. `safe_relation_indices` (boundary/validation.py:23-34) is inlined. Its
#      `relation_count <= 0` early return cannot be expressed in a graph, so R
#      is required to be >= 1; out-of-range rows are clamped for the gather and
#      then zeroed by the final `masked_fill`, exactly as in the reference.
#
# Everything else is kept byte-for-byte equivalent, in particular the two
# position clamps (relations.py:358 and relations.py:390-392). Those stay
# derived from `text_states.shape[1]`: they are genuine shape bounds, not
# semantic knobs, and they export to `Shape` -> `Sub` -> `Clip`, which stays
# dynamic in `L`. Routing them through `text_len` would silently turn a safety
# clamp into a knob.
# ─────────────────────────────────────────────────────────────────────────────
class RelationScorerWrapper(nn.Module):
    def __init__(self, scorer: nn.Module):
        super().__init__()
        self.scorer = scorer

    def forward(
        self,
        text_states: torch.Tensor,           # [1, L, H]   f32
        relation_query_head: torch.Tensor,   # [1, R, H]   f32
        relation_query_tail: torch.Tensor,   # [1, R, H]   f32
        relation_index: torch.Tensor,        # [P]         i64
        head_start: torch.Tensor,            # [P]         i64
        head_end: torch.Tensor,              # [P]         i64  half-open
        tail_start: torch.Tensor,            # [P]         i64
        tail_end: torch.Tensor,              # [P]         i64  half-open
        text_len: torch.Tensor,              # [1]         f32  PADDED length
    ) -> torch.Tensor:                       # [P]         f32  pair_logits
        s = self.scorer

        states = text_states.squeeze(0)                       # [L, H]
        length = states.shape[0]                              # SymInt, stays free

        # model.py:1397-1401 — directional_relation_states=true.
        rel_states = torch.cat(
            (relation_query_head, relation_query_tail), dim=-1
        ).squeeze(0)                                          # [R, 2H]
        rel_count = rel_states.shape[0]                       # SymInt

        # validation.py:33-34
        relation_valid = (relation_index >= 0) & (relation_index < rel_count)
        safe_index = relation_index.clamp(0, rel_count - 1)

        # relations.py:356-364 — gather(pos) with b == 0 everywhere.
        def gather(pos: torch.Tensor) -> torch.Tensor:
            return states.index_select(0, pos.clamp(0, length - 1))

        h_start = gather(head_start)
        h_end = gather(head_end - 1)
        t_start = gather(tail_start)
        t_end = gather(tail_end - 1)
        rel = rel_states.index_select(0, safe_index)           # [P, 2H]

        # relations.py:369-374
        delta = (tail_start - head_start).to(states.dtype)
        order = torch.sign(delta).unsqueeze(-1)
        dist = delta.abs().unsqueeze(-1) / text_len.clamp_min(1.0).to(states.dtype)

        feats = torch.cat([h_start, h_end, t_start, t_end, rel, order, dist], dim=-1)
        score = s.mlp(feats).squeeze(-1)

        if s.use_biaffine_content:
            # relations.py:378-407. The cumsum branch IS padding-transparent:
            # `routed_gather` zeroes the padded rows, and every span is bounded
            # by the word count, so no `prefix[end] - prefix[start]` difference
            # ever straddles padding. Only `dist` above is not.
            prefix = torch.cat(
                (
                    states.new_zeros(1, s.hidden_size),
                    states.float().cumsum(0).to(states.dtype),
                ),
                dim=0,
            )                                                  # [L + 1, H]

            def pool(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
                span_sum = prefix.index_select(0, end.clamp(0, length)) - \
                    prefix.index_select(0, start.clamp(0, length))
                width = (end - start).clamp_min(1).unsqueeze(-1).to(span_sum.dtype)
                return span_sum / width

            head_content = s.head_content_projection(pool(head_start, head_end))
            tail_content = s.tail_content_projection(pool(tail_start, tail_end))
            gate = torch.sigmoid(s.relation_content_gate(rel))
            biaffine = (head_content * gate * tail_content).sum(-1) / (
                s.hidden_size ** 0.5
            )
            linear = s.content_linear(
                torch.cat((head_content, tail_content, rel), dim=-1)
            ).squeeze(-1)
            score = score + biaffine + linear

        # relations.py:407 — invalid pairs score exactly 0.0, not -inf.
        return score.masked_fill(~relation_valid, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Export helpers
# ─────────────────────────────────────────────────────────────────────────────
def _export(
    module: nn.Module,
    args: tuple,
    out_path: Path,
    input_names: list,
    output_names: list,
    *,
    dynamic_axes: dict | None = None,
    dynamic_shapes: dict | None = None,
    dynamo: bool = False,
) -> None:
    kwargs = dict(
        input_names=input_names,
        output_names=output_names,
        opset_version=OPSET,
        dynamo=dynamo,
    )
    if dynamo:
        kwargs["dynamic_shapes"] = dynamic_shapes
    else:
        kwargs["dynamic_axes"] = dynamic_axes
    with torch.no_grad():
        torch.onnx.export(module, args, str(out_path), **kwargs)
    print(f"    FP32 -> {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def _fix_constant_of_shape(model) -> int:
    """
    Repairs `ConstantOfShape` nodes after FP16 conversion.

    `ConstantOfShape` derives its output type from the `value` attribute; when
    the attribute is absent the ONNX spec mandates `float32`. The FP16 converter
    rewrites the output's `value_info` declaring it `float16` but leaves the
    attribute alone, and the graph becomes inconsistent::

        Type Error: Type (tensor(float16)) of output arg (val_684) of node
        (node_ConstantOfShape_676) does not match expected type (tensor(float))

    Here the attribute is materialised in the type actually declared.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    declared = {
        vi.name: vi.type.tensor_type.elem_type
        for vi in list(model.graph.value_info) + list(model.graph.output)
    }
    fixed = 0
    for node in model.graph.node:
        if node.op_type != "ConstantOfShape":
            continue
        if declared.get(node.output[0]) != TensorProto.FLOAT16:
            continue
        current = None
        for attr in list(node.attribute):
            if attr.name == "value":
                current = numpy_helper.to_array(attr.t)
                node.attribute.remove(attr)
        scalar = np.float16(0.0) if current is None else current.astype(np.float16).reshape(-1)[0]
        node.attribute.append(
            helper.make_attribute(
                "value", numpy_helper.from_array(np.array([scalar], dtype=np.float16), "value")
            )
        )
        fixed += 1
    return fixed


def _topological_sort(graph) -> bool:
    """Restores topological node order after FP16 conversion.

    The converter appends its `*_cast_to_fp16` / `*_cast_to_fp32` nodes at the
    end of the node list, so a cast feeding the very first node ends up behind
    it. ONNX requires topological order and `onnx.checker` rejects the result::

        Nodes in a graph must be topologically sorted, however input
        'graph_input_cast_0' of node: name: node_Shape_0 OpType: Shape
        is not output of any previous nodes.

    onnxruntime loads such a graph anyway, which is why every fragment exported
    before this helper existed is checker-invalid and still runs. Reordering is
    a pure permutation: no node, input, output or attribute changes.

    Returns True when the order was changed. A graph that cannot be sorted (a
    genuine cycle, or an input no node produces) is left exactly as it was — the
    checker should report that, not this function.
    """
    produced = {""}
    produced |= {i.name for i in graph.initializer}
    produced |= {i.name for i in graph.input}
    original = list(graph.node)
    ordered: list = []
    pending = original
    while pending:
        progressed = False
        blocked = []
        for node in pending:
            if all(name in produced for name in node.input):
                ordered.append(node)
                produced.update(node.output)
                progressed = True
            else:
                blocked.append(node)
        if not progressed:
            return False
        pending = blocked
    if all(a is b for a, b in zip(ordered, original)):
        return False
    del graph.node[:]
    graph.node.extend(ordered)
    return True


def _convert_fp16(fp32_path: Path, keep_io_types: bool, out_path: Path) -> Path:
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = onnx.load(str(fp32_path))
    model = convert_float_to_float16(model, keep_io_types=keep_io_types)
    fixed = _fix_constant_of_shape(model)
    sorted_ = _topological_sort(model.graph)
    onnx.save(model, str(out_path))
    label = "fp16 (keep_io=FP32)" if keep_io_types else "fp16 (full FP16 IO)"
    note = f", {fixed} ConstantOfShape repaired" if fixed else ""
    note += ", nodes re-sorted" if sorted_ else ""
    print(f"    {label} -> {out_path.name}  ({out_path.stat().st_size / 1e6:.1f} MB){note}")
    return out_path


def _both_fp16(fp32_path: Path, out_dir: Path, stem: str) -> None:
    _convert_fp16(fp32_path, True, out_dir / f"{stem}_fp16.onnx")
    _convert_fp16(fp32_path, False, out_dir / f"{stem}_fp16_iobinding.onnx")


def _assert_no_traced_length(path: Path, traced_len: int) -> None:
    """Regression guard for the `float(max(L, 1))` hazard (relations.py:374).

    If the distance denominator were still computed from `boundary_states.shape[1]`
    through a Python `float()`, the tracing length would appear in the graph as a
    scalar constant. The graph would keep declaring `L` dynamic and would keep
    running at any `L` — it would simply return the wrong numbers everywhere but
    the traced length. So the check is on the constants, not on the shapes.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    onnx.checker.check_model(model)

    suspects = []

    def _scan(name: str, arr) -> None:
        arr = np.asarray(arr)
        if arr.size == 0 or arr.size > 8 or not np.issubdtype(arr.dtype, np.number):
            return
        flat = arr.reshape(-1)
        for v in flat:
            if float(v) in (float(traced_len), 1.0 / float(traced_len)):
                suspects.append((name, arr.tolist()))
                break

    for init in model.graph.initializer:
        _scan(init.name, numpy_helper.to_array(init))
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                _scan(node.name or node.output[0], numpy_helper.to_array(attr.t))

    if suspects:
        raise SystemExit(
            f"ERROR: {path.name} contains a constant equal to the tracing length "
            f"{traced_len} (or its reciprocal): {suspects}.\n"
            "  That is the `float(max(length, 1))` hazard — the distance "
            "denominator has been burned into the graph and `num_words` is a lie.\n"
            "  Feed the denominator through the `text_len` input instead."
        )

    ops = sorted({n.op_type for n in model.graph.node})
    banned = {"Loop", "If", "Scan", "SequenceAt"}
    hit = banned.intersection(ops)
    if hit:
        raise SystemExit(f"ERROR: {path.name} contains control-flow ops {sorted(hit)}.")
    print(f"    checked: {len(model.graph.node)} nodes, {len(ops)} distinct opset-"
          f"{OPSET} ops, no constant == traced L ({traced_len})")


def _dump_relation_settings(
    out_dir: Path, settings, scorer, hidden_size: int, params: int
) -> None:
    """Audit trail: every knob the relation path reads, straight off the checkpoint.

    The values here decide the shape of the exported graph. They come from the
    checkpoint's own `boundary_head` config block and several of them contradict
    the library defaults in `relations.py` (`pair_cap` 64 vs 128,
    `directional_relation_states` / `relation_biaffine_content` true vs false).
    Never substitute a library default for one of these.
    """
    dump = {
        "source": "model.boundary_settings (checkpoint config, not library defaults)",
        "hidden_size": hidden_size,
        "relation_scorer_parameters": int(params),
        "enable_relations": bool(settings.enable_relations),
        "relation_temperature": float(settings.relation_temperature),
        "relation_heads_per_type": int(settings.relation_heads_per_type),
        "relation_tails_per_type": int(settings.relation_tails_per_type),
        "relation_pair_cap": int(settings.relation_pair_cap),
        "relation_argument_proposal_threshold": float(
            settings.relation_argument_proposal_threshold
        ),
        "abstention_threshold": float(settings.abstention_threshold),
        "directional_relation_states": bool(settings.directional_relation_states),
        "relation_biaffine_content": bool(settings.relation_biaffine_content),
        "relation_query_dim": int(scorer.relation_query_dim),
        "library_defaults_overridden": {
            "relation_pair_cap": "library default is 128; the checkpoint says 64",
            "relation_argument_proposal_threshold":
                "library default is 0.0; the checkpoint says 0.2",
            "directional_relation_states":
                "library default is False; the checkpoint says True",
            "relation_biaffine_content":
                "library default is False; the checkpoint says True",
        },
    }
    (out_dir / "relation_settings.json").write_text(json.dumps(dump, indent=2))
    print("    relation_settings.json written")


# ─────────────────────────────────────────────────────────────────────────────
# Main export
# ─────────────────────────────────────────────────────────────────────────────
SECTIONS = ("encoder", "routed_gather", "boundary_head", "classifier", "relation_scorer")


def export_boundary(
    model_path: str,
    out_dir: Path,
    buckets: tuple[int, ...],
    sections: tuple[str, ...] = SECTIONS,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("GLiNER2.5 boundary ONNX Exporter v1")
    print("=" * 64)
    print(f"Model   : {model_path}")
    print(f"Output  : {out_dir}")
    print(f"Buckets : {list(buckets)}")
    print(f"Sections: {list(sections)}")
    print()

    bad = [b for b in buckets if b < MIN_BUCKET]
    if bad:
        raise SystemExit(
            f"ERROR: buckets {bad} are below the minimum of {MIN_BUCKET}.\n"
            f"  select_top_boundaries uses k=min(pool_boundary_top_k, n_boundaries):\n"
            f"  below {MIN_BUCKET} words the graph would be traced with a reduced k."
        )

    print("Loading AutoExtractor...")
    model = AutoExtractor.from_pretrained(model_path)
    model.eval()

    arch = getattr(model, "architecture", None) or type(model).__name__
    if getattr(model, "boundary_head", None) is None:
        raise SystemExit(
            f"ERROR: {model_path} is not a boundary model (architecture={arch}).\n"
            f"  Span checkpoints belong to the gliner2-rs crate instead."
        )

    head = model.boundary_head
    head.eval()
    head.collect_diagnostics = False

    settings = model.boundary_settings
    H = model.encoder.config.hidden_size
    pool_size = int(settings.pool_size)

    print(f"hidden_size      = {H}")
    print(f"candidate_pool   = {settings.candidate_pool}")
    print(f"pool_size (C)    = {pool_size}")
    print(f"pool_top_k       = {settings.pool_boundary_top_k}")
    print(f"abstention       = {settings.enable_abstention}")
    print(f"count_head       = {settings.enable_count_head}")
    print(f"overlap_policy   = {settings.overlap_policy}")
    print()

    if settings.candidate_pool != "shared":
        raise SystemExit(
            f"ERROR: candidate_pool='{settings.candidate_pool}' is not supported.\n"
            f"  Only the 'shared' branch produces a constant-size pool."
        )

    SEQ, Q, K = 64, 4, 5

    # ═════════════════════════════════════════════════════════════════════════
    # 1. ENCODER  (dynamic in seq_len)
    # ═════════════════════════════════════════════════════════════════════════
    if "encoder" in sections:
        print("--- 1. encoder ---")
        enc32 = out_dir / "encoder_fp32.onnx"
        _export(
            EncoderWrapper(model.encoder),
            (torch.randint(0, 1000, (1, SEQ)), torch.ones((1, SEQ), dtype=torch.long)),
            enc32,
            ["input_ids", "attention_mask"],
            ["last_hidden_state"],
            dynamic_axes={
                "input_ids":         {0: "batch", 1: "seq_len"},
                "attention_mask":    {0: "batch", 1: "seq_len"},
                "last_hidden_state": {0: "batch", 1: "seq_len"},
            },
        )
        _both_fp16(enc32, out_dir, "encoder")
        print()

    # ═════════════════════════════════════════════════════════════════════════
    # 2. ROUTED GATHER  (dynamic; serves text / query / cls)
    # ═════════════════════════════════════════════════════════════════════════
    if "routed_gather" in sections:
        print("--- 2. routed_gather ---")
        rg32 = out_dir / "routed_gather_fp32.onnx"
        _export(
            RoutedGatherWrapper(),
            (
                torch.randn(1, SEQ, H),
                torch.randint(0, SEQ, (1, 20)),
                torch.ones(1, 20, dtype=torch.long),
            ),
            rg32,
            ["last_hidden_state", "indices", "mask"],
            ["states"],
            dynamic_axes={
                "last_hidden_state": {1: "seq_len"},
                "indices":           {1: "num_slots"},
                "mask":              {1: "num_slots"},
                "states":            {1: "num_slots"},
            },
        )
        _both_fp16(rg32, out_dir, "routed_gather")
        print()

    # ═════════════════════════════════════════════════════════════════════════
    # 3. BOUNDARY HEAD  (one graph per bucket; num_queries dynamic)
    # ═════════════════════════════════════════════════════════════════════════
    if "boundary_head" in sections:
        print("--- 3. boundary_head (per bucket) ---")
        nq = torch.export.Dim("num_queries", min=1, max=256)
        for L in buckets:
            stem = f"boundary_head_L{L}"
            print(f"  bucket L={L}")
            fp32 = out_dir / f"{stem}_fp32.onnx"
            with _unstable_sort():
                _export(
                    BoundaryHeadWrapper(head),
                    (
                        torch.randn(1, L, H),
                        torch.ones(1, L, dtype=torch.long),
                        torch.randn(1, Q, H),
                        torch.ones(1, Q, dtype=torch.long),
                    ),
                    fp32,
                    ["text_states", "text_mask", "query_states", "query_mask"],
                    ["cand_indices", "pair_logits", "cand_valid",
                     "null_logits", "count_log_rates"],
                    dynamic_shapes={
                        "text_states":  {1: L},
                        "text_mask":    {1: L},
                        "query_states": {1: nq},
                        "query_mask":   {1: nq},
                    },
                    dynamo=True,
                )
            _both_fp16(fp32, out_dir, stem)
        print()

    # ═════════════════════════════════════════════════════════════════════════
    # 4. CLASSIFIER
    # ═════════════════════════════════════════════════════════════════════════
    if "classifier" in sections:
        print("--- 4. classifier ---")
        cls32 = out_dir / "classifier_fp32.onnx"
        _export(
            ClassifierWrapper(model.classifier),
            (torch.randn(K, H),),
            cls32,
            ["choice_states"],
            ["logits"],
            dynamic_axes={"choice_states": {0: "num_choices"},
                          "logits": {0: "num_choices"}},
        )
        _both_fp16(cls32, out_dir, "classifier")
        print()

    # ═════════════════════════════════════════════════════════════════════════
    # 5. RELATION SCORER  (no length buckets: L, R and P are all dynamic)
    #
    # `SparseRelationScorer.forward` has no Python loop over a symbolic
    # dimension, so - unlike the boundary head - nothing forces specialisation
    # and one graph serves every length. See `RelationScorerWrapper` for the
    # `float(max(L, 1))` hazard that made `text_len` an explicit input.
    # ═════════════════════════════════════════════════════════════════════════
    scorer = getattr(model, "relation_scorer", None)
    if "relation_scorer" in sections and scorer is not None:
        scorer.eval()
        print("--- 5. relation_scorer ---")
        params = sum(p.numel() for p in scorer.parameters())
        print(f"  relation_query_dim   = {scorer.relation_query_dim}")
        print(f"  use_biaffine_content = {scorer.use_biaffine_content}")
        print(f"  parameters           = {params:,}")
        if scorer.relation_query_dim != 2 * H and settings.directional_relation_states:
            raise SystemExit(
                "ERROR: directional_relation_states=True but relation_query_dim "
                f"is {scorer.relation_query_dim}, not {2 * H}. The wrapper "
                "concatenates the head and tail query states in-graph and would "
                "feed the MLP the wrong width."
            )

        L0, R0, P0 = 37, 3, 11  # tracing example; no dimension may survive it
        nl = torch.export.Dim("num_words", min=2, max=4096)
        nr = torch.export.Dim("num_relations", min=1, max=256)
        npair = torch.export.Dim("num_pairs", min=1, max=16384)
        hs = torch.randint(0, L0 - 1, (P0,))
        ts = torch.randint(0, L0 - 1, (P0,))
        rel32 = out_dir / "relation_scorer_fp32.onnx"
        with _unstable_sort():
            _export(
                RelationScorerWrapper(scorer),
                (
                    torch.randn(1, L0, H),
                    torch.randn(1, R0, H),
                    torch.randn(1, R0, H),
                    torch.randint(0, R0, (P0,)),
                    hs, hs + 1, ts, ts + 1,
                    torch.tensor([float(L0)]),
                ),
                rel32,
                ["text_states", "relation_query_head", "relation_query_tail",
                 "relation_index", "head_start", "head_end",
                 "tail_start", "tail_end", "text_len"],
                ["pair_logits"],
                dynamic_shapes={
                    "text_states":         {1: nl},
                    "relation_query_head": {1: nr},
                    "relation_query_tail": {1: nr},
                    "relation_index":      {0: npair},
                    "head_start":          {0: npair},
                    "head_end":            {0: npair},
                    "tail_start":          {0: npair},
                    "tail_end":            {0: npair},
                    "text_len":            {},
                },
                dynamo=True,
            )
        _assert_no_traced_length(rel32, L0)
        _both_fp16(rel32, out_dir, "relation_scorer")
        _dump_relation_settings(out_dir, settings, scorer, H, params)
        print()
    elif "relation_scorer" in sections:
        print("--- 5. relation_scorer ---")
        print("  checkpoint has no relation_scorer (enable_relations=False); skipped")
        print()

    if not (out_dir / "tokenizer.json").exists():
        _copy_tokenizer(model_path, out_dir)
    _write_manifest(out_dir, model, settings, H, buckets, pool_size, scorer)
    _print_summary(out_dir, buckets)


# ─────────────────────────────────────────────────────────────────────────────
# Manifest consumed by the Rust runtime
# ─────────────────────────────────────────────────────────────────────────────
def _write_manifest(
    out_dir: Path,
    model,
    settings,
    hidden_size: int,
    buckets: tuple[int, ...],
    pool_size: int,
    relation_scorer=None,
) -> None:
    manifest = {
        "architecture": "boundary",
        "exporter": "export_boundary_v1.py",
        "opset": OPSET,
        "hidden_size": hidden_size,
        "pool_size": pool_size,
        "pool_boundary_top_k": int(settings.pool_boundary_top_k),
        "length_buckets": list(buckets),
        "min_bucket": MIN_BUCKET,
        "enable_abstention": bool(settings.enable_abstention),
        "enable_count_head": bool(settings.enable_count_head),
        "enable_relations": bool(settings.enable_relations),
        "enable_records": bool(settings.enable_records),
        "overlap_policy": str(settings.overlap_policy),
        "token_pooling": str(getattr(model.processor, "token_pooling", "first")),
        "max_position_embeddings": int(model.encoder.config.max_position_embeddings),
        "notes": {
            "padding": "pad the text to the smallest bucket that fits it, "
                       "with text_mask=0 on the padded rows",
            "stable_sort": "stable=True stripped at export; measured parity ~1e-05",
            "decoding": "sigmoid(pair_logits) -> per-query threshold -> overlap_policy -> stable ranking",
        },
    }

    # ── relation scorer ──────────────────────────────────────────────────────
    # Every value below is read off `model.boundary_settings`, i.e. off the
    # checkpoint, never off a library default — see relation_settings.json.
    manifest["enable_relation_scorer"] = bool(
        relation_scorer is not None and settings.enable_relations
    )
    manifest["relation_temperature"] = float(settings.relation_temperature)
    manifest["relation_heads_per_type"] = int(settings.relation_heads_per_type)
    manifest["relation_tails_per_type"] = int(settings.relation_tails_per_type)
    manifest["relation_pair_cap"] = int(settings.relation_pair_cap)
    manifest["relation_argument_proposal_threshold"] = float(
        settings.relation_argument_proposal_threshold
    )
    manifest["relation_query_dim"] = int(
        relation_scorer.relation_query_dim
        if relation_scorer is not None
        else (2 * hidden_size if settings.directional_relation_states else hidden_size)
    )
    manifest["relation_biaffine_content"] = bool(settings.relation_biaffine_content)
    manifest["directional_relation_states"] = bool(
        settings.directional_relation_states
    )
    manifest["notes"]["relation_scorer"] = (
        "relation_scorer_{fp32,fp16,fp16_iobinding}.onnx. No length buckets: L "
        "(num_words), R (num_relations) and P (num_pairs) are all dynamic. "
        "Inputs: text_states[1,L,H], relation_query_head[1,R,H], "
        "relation_query_tail[1,R,H] (concatenated in-graph, so keep both halves "
        "on the device), relation_index/head_start/head_end/tail_start/tail_end "
        "[P] int64 with half-open ends, text_len[1] float32. "
        "Output: pair_logits[P] float32; apply "
        "sigmoid(pair_logits / relation_temperature). Pairs whose relation_index "
        "is out of [0, R) score exactly 0.0."
    )
    manifest["notes"]["relation_text_len"] = (
        "text_len is the PADDED length of text_states (its dim 1) — NOT the word "
        "count, and NOT a masked length. Python takes it from "
        "boundary_states.shape[1] (relations.py:355) and divides the raw word "
        "distance by it (relations.py:374) with no mask anywhere, so padding is "
        "NOT transparent for this fragment, unlike the boundary head. Python pads "
        "text_states to the batch maximum; Rust pads to a 64/128/256/512 length "
        "bucket. Feeding different numbers shifts every relation score while "
        "fragment parity, entity parity and every existing test stay green. "
        "Whatever value you feed, feed the same one both sides, and take it from "
        "the fixture rather than from a local convention."
    )
    manifest["notes"]["relation_precision"] = (
        "measured against PyTorch on random states: fp32 max |dlogit| 3.3e-06 "
        "over L in {41,97,128,512}; fp16 max |dprob| 2.8e-04 at L<=128 and "
        "5.5e-03 at L=512 (tolerance 2e-02). The fp16 error grows with the "
        "padded length and with how far into the sequence a span sits, because "
        "the biaffine branch pools through an fp16 cumsum prefix over all L "
        "rows; prefer fp32 for the 512 bucket if a tighter bound is needed."
    )
    manifest["notes"]["relation_thresholds"] = (
        "relation_argument_proposal_threshold (0.2) and abstention_threshold "
        "(0.5) are distinct keys that coexist. 0.2 filters the RAW candidate "
        "pool when proposing relation arguments (relations.py:165-167); 0.5 "
        "remains the decode threshold. Proposing at 0.2 does not lower the "
        "decode threshold."
    )

    (out_dir / "boundary_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"boundary_manifest.json written ({len(buckets)} buckets)")


def _copy_tokenizer(model_path: str, out_dir: Path) -> None:
    src = Path(model_path) / "tokenizer.json"
    if src.exists():
        shutil.copy(src, out_dir / "tokenizer.json")
        print(f"tokenizer.json copied from {model_path}")
        return
    try:
        from huggingface_hub import hf_hub_download

        shutil.copy(hf_hub_download(model_path, "tokenizer.json"), out_dir / "tokenizer.json")
        print(f"tokenizer.json downloaded from the HuggingFace Hub: {model_path}")
    except Exception as e:
        print(f"WARN: could not copy tokenizer.json: {e}")


def _print_summary(out_dir: Path, buckets: tuple[int, ...]) -> None:
    print("=" * 64)
    print("Boundary export v1 complete")
    print()
    total = sum(f.stat().st_size for f in out_dir.glob("*.onnx"))
    print(f"{len(list(out_dir.glob('*.onnx')))} ONNX files, {total / 1e9:.2f} GB total")
    print()
    print("Execution chain:")
    print("  encoder(input_ids, attention_mask) -> last_hidden_state")
    print("  routed_gather(last_hidden_state, idx, mask) -> text/query/cls states")
    print(f"  boundary_head_L<{'|'.join(map(str, buckets))}>(text, text_mask, query, query_mask)")
    print("      -> cand_indices, pair_logits, cand_valid, null_logits, count_log_rates")
    print("  classifier(choice_states) -> logits")
    if (out_dir / "relation_scorer_fp32.onnx").exists():
        print("  relation_scorer(text_states, relation_query_head, relation_query_tail,")
        print("                  relation_index, head_start, head_end,")
        print("                  tail_start, tail_end, text_len) -> pair_logits")
        print("      text_len is the PADDED length of text_states — see the manifest.")
    print("=" * 64)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GLiNER2.5 boundary ONNX Exporter v1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model_path", required=True,
                   help="Local path or HuggingFace repo id of the boundary checkpoint")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--buckets", default=",".join(map(str, DEFAULT_BUCKETS)),
                   help=f"Comma-separated length buckets (default {DEFAULT_BUCKETS}, min {MIN_BUCKET})")
    p.add_argument("--only", default=",".join(SECTIONS),
                   help="Comma-separated subset of "
                        f"{','.join(SECTIONS)} to (re-)export. The manifest and "
                        "the summary are always rewritten; files belonging to "
                        "sections not listed are left untouched.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    requested = tuple(x.strip() for x in args.only.split(",") if x.strip())
    unknown = [x for x in requested if x not in SECTIONS]
    if unknown:
        raise SystemExit(f"ERROR: unknown --only section(s) {unknown}; pick from {list(SECTIONS)}")
    export_boundary(
        model_path=args.model_path,
        out_dir=Path(args.out_dir),
        buckets=tuple(sorted(int(x) for x in args.buckets.split(","))),
        sections=requested,
    )
