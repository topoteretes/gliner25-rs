"""
PyTorch vs ONNX parity check for the GLiNER2.5 boundary export.

Compares every ONNX fragment against the matching PyTorch module on random
inputs, and additionally checks the assumption the bucketing rests on: padding
the text to a larger bucket, with `text_mask = 0` on the padded rows, must not
change the result.

Five fragments: `encoder`, `routed_gather`, `classifier`, `boundary_head_L*`
(one per length bucket) and `relation_scorer`. The relation scorer is the only
one that is not bucketed — one graph serves every length — and the only one for
which padding is NOT transparent, so it carries its own `text_len` contract;
see the note above `RELATION_CASES`.

Usage:
    python verify_parity.py \
        --model_path fastino/gliner2.5-multi-v1 \
        --onnx_dir models/gliner2.5-multi-v1-onnx

Exits with status 1 if any check exceeds its tolerance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import onnxruntime as ort
import torch

# **Relative** tolerances, scaled by the reference's largest magnitude. An
# absolute threshold is unusable here because the fragments operate at very
# different scales.
TOL_FP32 = 1e-5
TOL_FP16 = 5e-3

# Fragments that emit probabilities live in [0,1], so relative and absolute
# error coincide. With random inputs the dot products over 768 dimensions have
# magnitude ~sqrt(768), the logits saturate, and an FP16 perturbation around the
# sigmoid transition is worth a few thousandths of a probability. That is the
# cost of half precision on a probability, not a defect of the export.
TOL_FP16_PROB = 2e-2

# Boundary candidate pool: FP32 must match exactly; in FP16 rounding can swap a
# near-tied candidate right at the `pool_size` cut. Up to 1% is a precision
# effect, not a defect of the export.
TOL_POOL_FP32 = 0.0
TOL_POOL_FP16 = 1e-2


def _session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def _run(sess, feeds: dict) -> list[np.ndarray]:
    typed = {}
    for i in sess.get_inputs():
        v = feeds[i.name]
        want = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64,
            "tensor(bool)": np.bool_,
        }[i.type]
        typed[i.name] = np.asarray(v).astype(want)
    return sess.run(None, typed)


class Report:
    def __init__(self) -> None:
        # (fragment, variant, value, bound, ok, sense) — `sense` is "<=" for a
        # tolerance and ">=" for a floor (a check that must *not* be small; see
        # the `text_len` liveness probe).
        self.rows: list[tuple[str, str, float, float, bool, str]] = []
        self.skips: list[tuple[str, str]] = []

    def add(self, fragment: str, variant: str, delta: float, tol: float) -> None:
        self.rows.append((fragment, variant, delta, tol, delta <= tol, "<="))

    def add_at_least(self, fragment: str, variant: str, value: float, floor: float) -> None:
        """A check that fails when the measured quantity is too *small*."""
        self.rows.append((fragment, variant, value, floor, value >= floor, ">="))

    def skip(self, fragment: str, reason: str) -> None:
        """Record a check that did NOT run. Never silent: `show()` shouts it."""
        self.skips.append((fragment, reason))

    @staticmethod
    def relative(ref: np.ndarray, got: np.ndarray) -> float:
        """Largest error, scaled by the reference's magnitude."""
        ref = ref.astype(np.float64)
        got = got.astype(np.float64)
        scale = max(float(np.abs(ref).max()), 1e-12)
        return float(np.abs(ref - got).max() / scale)

    def failed(self) -> bool:
        return any(not row[4] for row in self.rows)

    def show(self) -> None:
        print()
        print("error column: boundary rows are RELATIVE (scaled by the reference's")
        print("largest magnitude); relation_scorer rows are ABSOLUTE — |dlogit| in")
        print("fp32, |dprob| after the sigmoid in fp16.")
        print()
        print(f"{'fragment':<44} {'variant':<18} {'error':>12} {'bound':>14}  result")
        print("-" * 104)
        for frag, var, d, bound, ok, sense in self.rows:
            print(f"{frag:<44} {var:<18} {d:>12.3e} {sense + f' {bound:.3e}':>14}  "
                  f"{'OK' if ok else 'FAILED'}")
        if self.skips:
            print()
            for frag, reason in self.skips:
                print(f"!! SKIPPED  {frag}: {reason}")
        print()
        if self.failed():
            print("FAILED")
        elif self.skips:
            print(f"all checks that RAN passed, but {len(self.skips)} check(s) were SKIPPED "
                  "— this is not a pass")
        else:
            print("all checks passed")


def _compare_candidate_pool(ref_idx, ref_log, ref_val, got_idx, got_log, got_val, num_queries):
    """
    Compares the candidate pool as a **set**, not positionally.

    Pool order carries no meaning: it comes from an `argsort` over frequently
    near-tied scores, and sort stability is exactly what the export removes
    (ONNX has no stable Sort). Under FP16 rounding permutes the ties. Comparing
    `cand_indices` element by element would fail a perfectly correct graph.

    What matters is that the set of proposed `(start, end)` pairs matches and
    that the logit attached to each pair is the same.

    Logits are compared **after the sigmoid**: that is the quantity the decoder
    thresholds on, and therefore the only one whose deviation is interpretable.
    A 1.6e-2 gap on a raw logit, which looks large, is worth less than half a
    percentage point of probability.

    Returns `(fraction of missing candidates, max |probability delta|)`.
    """
    def _sigmoid(x: float) -> float:
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        z = np.exp(x)
        return float(z / (1.0 + z))

    missing_total = 0
    expected_total = 0
    worst = 0.0
    for q in range(num_queries):
        ref_pairs = {
            tuple(int(v) for v in pair): float(logit)
            for pair, logit, ok in zip(ref_idx[0, q], ref_log[0, q], ref_val[0, q])
            if ok
        }
        got_pairs = {
            tuple(int(v) for v in pair): float(logit)
            for pair, logit, ok in zip(got_idx[0, q], got_log[0, q], got_val[0, q])
            if ok
        }
        common = set(ref_pairs) & set(got_pairs)
        expected_total += len(ref_pairs)
        missing_total += len(ref_pairs) - len(common)
        for key in common:
            worst = max(worst, abs(_sigmoid(ref_pairs[key]) - _sigmoid(got_pairs[key])))
    fraction_missing = missing_total / max(expected_total, 1)
    return fraction_missing, worst


def _variants(onnx_dir: Path, stem: str) -> list[tuple[str, Path, float]]:
    out = []
    for suffix, tol in (("_fp32", TOL_FP32), ("_fp16", TOL_FP16), ("_fp16_iobinding", TOL_FP16)):
        p = onnx_dir / f"{stem}{suffix}.onnx"
        if p.exists():
            out.append((suffix.lstrip("_"), p, tol))
    return out


def _compare(
    report: Report,
    stem: str,
    onnx_dir: Path,
    feeds: dict,
    ref: np.ndarray,
    out_index: int = 0,
    tol_fp16: float = TOL_FP16,
) -> None:
    for name, path, tol in _variants(onnx_dir, stem):
        if name != "fp32":
            tol = tol_fp16
        got = _run(_session(path), feeds)[out_index]
        report.add(stem, name, Report.relative(ref, got), tol)


# ─────────────────────────────────────────────────────────────────────────────
# boundary
# ─────────────────────────────────────────────────────────────────────────────
_MODEL_CACHE: dict[str, object] = {}


def _load_model(model_path: str):
    """The checkpoint is loaded once and shared by both verification sections."""
    if model_path not in _MODEL_CACHE:
        from gliner2 import AutoExtractor

        model = AutoExtractor.from_pretrained(model_path)
        model.eval()
        _MODEL_CACHE[model_path] = model
    return _MODEL_CACHE[model_path]


def verify_boundary(model_path: str, onnx_dir: Path) -> Report:
    import json

    manifest = json.loads((onnx_dir / "boundary_manifest.json").read_text())
    model = _load_model(model_path)
    head = model.boundary_head
    head.eval()
    head.collect_diagnostics = False
    H = model.encoder.config.hidden_size

    torch.manual_seed(0)
    report = Report()
    SEQ, Q, K = 40, 4, 5

    ids = torch.randint(5, 1000, (1, SEQ))
    mask = torch.ones(1, SEQ, dtype=torch.long)
    with torch.no_grad():
        hidden = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
    _compare(report, "encoder", onnx_dir,
             {"input_ids": ids.numpy(), "attention_mask": mask.numpy()}, hidden.numpy())

    idx = torch.randint(0, SEQ, (1, 20))
    rmask = torch.ones(1, 20, dtype=torch.long)
    ref = hidden.gather(1, idx.clamp(0, SEQ - 1).unsqueeze(-1).expand(-1, -1, H)) * rmask.unsqueeze(-1)
    _compare(report, "routed_gather", onnx_dir,
             {"last_hidden_state": hidden.numpy(), "indices": idx.numpy(), "mask": rmask.numpy()},
             ref.numpy())

    choice = torch.randn(K, H)
    with torch.no_grad():
        ref = model.classifier(choice).squeeze(-1)
    _compare(report, "classifier", onnx_dir, {"choice_states": choice.numpy()}, ref.numpy())

    # ── heads, one per bucket ─────────────────────────────────────────────
    for L in manifest["length_buckets"]:
        ts = torch.randn(1, L, H)
        tm = torch.ones(1, L, dtype=torch.bool)
        qs = torch.randn(1, Q, H)
        qm = torch.ones(1, Q, dtype=torch.bool)
        with torch.no_grad():
            out = head(ts, tm, qs, qm, targets=None,
                       return_candidates=True, collect_diagnostics=False)
        ref_idx = out.candidates.indices.numpy()
        ref_log = out.candidates.pair_logits.numpy()

        feeds = {
            "text_states": ts.numpy(), "text_mask": tm.numpy(),
            "query_states": qs.numpy(), "query_mask": qm.numpy(),
        }
        ref_val = out.candidates.valid_mask.numpy()
        for name, path, tol in _variants(onnx_dir, f"boundary_head_L{L}"):
            got = _run(_session(path), feeds)
            g_idx, g_log, g_val = got[0], got[1], got[2].astype(bool)
            miss, delta = _compare_candidate_pool(
                ref_idx, ref_log, ref_val, g_idx, g_log, g_val, Q
            )
            pool_tol = TOL_POOL_FP32 if name == "fp32" else TOL_POOL_FP16
            prob_tol = TOL_FP32 if name == "fp32" else TOL_FP16_PROB
            report.add(f"boundary_head_L{L}[pool]", name, miss, pool_tol)
            report.add(f"boundary_head_L{L}[prob]", name, delta, prob_tol)

    # ── the bucketing assumption: masked padding is transparent ───────────
    buckets = sorted(manifest["length_buckets"])
    for real in (buckets[0] - 8, buckets[0]):
        if real < manifest["min_bucket"]:
            continue
        target = next(b for b in buckets if b >= real)
        ts = torch.randn(1, real, H)
        qs = torch.randn(1, Q, H)
        qm = torch.ones(1, Q, dtype=torch.bool)
        with torch.no_grad():
            a = head(ts, torch.ones(1, real, dtype=torch.bool), qs, qm,
                     targets=None, return_candidates=True, collect_diagnostics=False)
            pad = target - real
            # noise in the padding: if masking works it must not matter
            ts_p = torch.cat([ts, torch.randn(1, pad, H)], 1)
            tm_p = torch.cat([torch.ones(1, real, dtype=torch.bool),
                              torch.zeros(1, pad, dtype=torch.bool)], 1)
            b = head(ts_p, tm_p, qs, qm, targets=None,
                     return_candidates=True, collect_diagnostics=False)
        miss, delta = _compare_candidate_pool(
            a.candidates.indices.numpy(), a.candidates.pair_logits.numpy(),
            a.candidates.valid_mask.numpy(),
            b.candidates.indices.numpy(), b.candidates.pair_logits.numpy(),
            b.candidates.valid_mask.numpy(), Q,
        )
        report.add(f"padding {real}->{target}[pool]", "pytorch", miss, TOL_POOL_FP32)
        report.add(f"padding {real}->{target}[prob]", "pytorch", delta, TOL_FP32)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# relation scorer
# ─────────────────────────────────────────────────────────────────────────────
#
# THE `text_len` CONTRACT — read this before changing anything below.
#
# `text_len` is the PADDED length of `text_states` (its dim 1). It is NOT the
# word count and NOT a masked length. Python takes it from
# `boundary_states.shape[1]` (`relations.py:355`) and divides the raw word
# distance by it (`relations.py:374`) with **no mask anywhere**, so unlike the
# boundary head, padding is NOT transparent for this fragment. Python pads to
# the batch maximum; Rust pads to a 64/128/256/512 bucket. Feed different
# numbers on the two sides and every relation score shifts while fragment
# parity, entity parity and every existing test stay green.
#
# Therefore: `text_len` is always taken from the CASE (the stand-in for the
# 7a-3 fixture) and never recomputed from a tensor shape, never defaulted, and
# never inferred from `num_words`. `_relation_text_len` raises rather than
# guessing, and `_relation_feeds` takes it keyword-only with no default, so a
# caller that forgets it gets a TypeError instead of a plausible number.
#
# `num_words` is how many leading rows of `text_states` are real; the rest are
# zeroed padding and every span stays inside `num_words`. Cases where
# `text_len != num_words` are the exact Risk-2 condition and are mandatory — a
# suite that only ever ran `text_len == num_words` would pass with the trap
# still live.
class RelationCase(NamedTuple):
    """
    One parity case. `text_len` has NO default on purpose — a case cannot be
    written without stating the padded length.

    `span_lo` pins the spans into `[span_lo, num_words)` instead of drawing
    them across the whole sequence. The fp16 error of the biaffine branch grows
    with how deep into the sequence a span sits (the `cumsum` prefix is larger
    there and loses more mantissa), so the deep cases are the fp16 worst case
    and must not be dropped in favour of "random spans" alone.
    """

    name: str
    text_len: int      # PADDED length == text_states.shape[1]; from the fixture
    num_words: int     # real rows; the rest of text_states is zeroed padding
    relations: int
    pairs: int
    span_lo: int = 0


RELATION_CASES = (
    RelationCase("L64 exact",        text_len=64,  num_words=64,  relations=1, pairs=1),
    RelationCase("L64 padded",       text_len=64,  num_words=40,  relations=5, pairs=64),
    RelationCase("L97 padded",       text_len=97,  num_words=41,  relations=3, pairs=64),
    RelationCase("L128 exact",       text_len=128, num_words=128, relations=2, pairs=64),
    RelationCase("L512 padded",      text_len=512, num_words=300, relations=4, pairs=128),
    RelationCase("L512 deep spans",  text_len=512, num_words=512, relations=1, pairs=64,
                 span_lo=470),
    RelationCase("L512 early spans", text_len=512, num_words=512, relations=2, pairs=64,
                 span_lo=0),
    RelationCase("L64 deep spans",   text_len=64,  num_words=64,  relations=1, pairs=64,
                 span_lo=40),
)

# The `text_len` liveness probe: feeding two different denominators to the same
# graph, with everything else held fixed, must move the logits by at least this
# much. If `float(max(length, 1))` (`relations.py:374`) were ever baked back
# into the graph as a constant, this collapses to 0 and the row fails.
TOL_TEXT_LEN_LIVE = 1e-2


def _relation_text_len(case) -> int:
    """
    Pull the PADDED length out of the case. Never derive it, never default it.

    This exists as a function purely so the failure mode is loud: a case that
    does not carry an explicit `text_len` aborts the run instead of silently
    falling back to `num_words` or to `text_states.shape[1]`. When 7a-3's
    fixture lands, the fixture record — not this file — is what fills the field.
    """
    text_len = getattr(case, "text_len", None)
    if text_len is None:
        raise ValueError(
            f"relation case {case!r} carries no `text_len`. It is the PADDED "
            "length (Python: `boundary_states.shape[1]`, relations.py:355) and "
            "must be stated explicitly — see the contract note above. There is "
            "deliberately no default."
        )
    if not isinstance(text_len, int) or text_len < 1:
        raise ValueError(
            f"relation case {case.name!r} has text_len={text_len!r}; it must be "
            "a positive int taken from the fixture, not a convention."
        )
    if case.num_words > text_len:
        raise ValueError(
            f"relation case {case.name!r}: num_words {case.num_words} exceeds "
            f"text_len {text_len} — padding cannot be shorter than the words."
        )
    return text_len


def _relation_pairs(num_pairs: int, num_relations: int, num_words: int, rng,
                    span_lo: int = 0):
    """
    A `RelationPairBatch` whose first rows pin the branches that only fire on
    specially-shaped input:

      row 0  head span == tail span  -> `delta == 0`, the `sign(0) == 0` branch
                                        of `relations.py:373`
      row 1  relation_index == -1    -> out of range below
      row 2  relation_index == R     -> out of range above

    Out-of-range rows must come back as exactly `0.0`, not `-inf`
    (`relations.py:407`, `score.masked_fill(~pair_valid, 0.0)`).
    """
    from gliner2.models.boundary.relations import RelationPairBatch

    hi = max(num_words - 3, 1)
    lo = min(max(span_lo, 0), hi - 1)
    head_start = rng.integers(lo, hi, num_pairs).astype(np.int64)
    head_end = head_start + rng.integers(1, 4, num_pairs).astype(np.int64)
    tail_start = rng.integers(lo, hi, num_pairs).astype(np.int64)
    tail_end = tail_start + rng.integers(1, 4, num_pairs).astype(np.int64)
    relation_index = rng.integers(-1, num_relations + 1, num_pairs).astype(np.int64)

    if num_pairs >= 1:
        tail_start[0], tail_end[0] = head_start[0], head_end[0]
        relation_index[0] = 0
    if num_pairs >= 2:
        relation_index[1] = -1
    if num_pairs >= 3:
        relation_index[2] = num_relations

    t = torch.from_numpy
    pairs = RelationPairBatch(
        batch_index=torch.zeros(num_pairs, dtype=torch.long),
        relation_index=t(relation_index),
        head_start=t(head_start), head_end=t(head_end),
        tail_start=t(tail_start), tail_end=t(tail_end),
        head_prob=torch.rand(num_pairs), tail_prob=torch.rand(num_pairs),
        pair_mask=None,
    )
    invalid = (relation_index < 0) | (relation_index >= num_relations)
    return pairs, invalid


def _relation_feeds(states, query_head, query_tail, pairs, *, text_len: int) -> dict:
    """
    Build the ONNX feed. `text_len` is keyword-only and has NO default: this
    fragment cannot be run without someone stating the padded length out loud.
    """
    if text_len is None:
        raise ValueError("text_len is required; see the contract note above")
    return {
        "text_states": states.numpy(),
        "relation_query_head": query_head.numpy(),
        "relation_query_tail": query_tail.numpy(),
        "relation_index": pairs.relation_index.numpy(),
        "head_start": pairs.head_start.numpy(),
        "head_end": pairs.head_end.numpy(),
        "tail_start": pairs.tail_start.numpy(),
        "tail_end": pairs.tail_end.numpy(),
        "text_len": np.array([float(text_len)], dtype=np.float32),
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float64)))


def _relation_reference(scorer, states, query_head, query_tail, pairs) -> np.ndarray:
    """
    `SparseRelationScorer.forward` with `directional_relation_states=True`:
    the two query halves are concatenated before the call (`model.py:1397-1401`),
    which is what `RelationScorerWrapper` does in-graph.

    NOTE the reference reads its own denominator from `states.shape[1]`, so the
    caller MUST pass states already padded to the case's `text_len` — that is
    the only way the two sides agree, and it is why the padded cases exist.
    """
    with torch.no_grad():
        return scorer(states, torch.cat((query_head, query_tail), dim=-1), None, pairs).numpy()


def verify_relation_scorer(model_path: str, onnx_dir: Path) -> Report:
    report = Report()
    model = _load_model(model_path)
    scorer = getattr(model, "relation_scorer", None)
    H = model.encoder.config.hidden_size

    if scorer is None:
        report.skip("relation_scorer",
                    f"the checkpoint {model_path} has no `relation_scorer` module")
        return report
    scorer.eval()

    if not (onnx_dir / "relation_scorer_fp32.onnx").exists():
        # The checkpoint HAS the module, so a missing export is a failure, not a
        # skip: this is the exact shape of "a skipped check printed success".
        report.add("relation_scorer[missing]", "fp32", float("inf"), 0.0)
        print("!! relation_scorer_fp32.onnx is missing but the checkpoint has the "
              "module — recorded as FAILED, not skipped")
        return report

    torch.manual_seed(0)
    rng = np.random.default_rng(11)
    fp16_by_length: dict[tuple[int, int], float] = {}

    for case in RELATION_CASES:
        name, num_words = case.name, case.num_words
        num_relations, num_pairs = case.relations, case.pairs
        text_len = _relation_text_len(case)          # from the case, never derived

        states = torch.randn(1, text_len, H)
        states[:, num_words:, :] = 0.0               # padded rows, as Rust pads them
        query_head = torch.randn(1, num_relations, H)
        query_tail = torch.randn(1, num_relations, H)
        pairs, invalid = _relation_pairs(num_pairs, num_relations, num_words, rng,
                                         span_lo=case.span_lo)

        ref = _relation_reference(scorer, states, query_head, query_tail, pairs)
        feeds = _relation_feeds(states, query_head, query_tail, pairs, text_len=text_len)
        label = f"relation_scorer[{name}]"

        for variant, path, _ in _variants(onnx_dir, "relation_scorer"):
            got = _run(_session(path), feeds)[0]
            if variant == "fp32":
                delta = float(np.abs(ref.astype(np.float64) - got.astype(np.float64)).max())
                report.add(label, variant, delta, TOL_FP32)
                # relations.py:407 — invalid pairs are zeroed, NOT set to -inf.
                if invalid.any():
                    worst = float(np.abs(got.astype(np.float64)[invalid]).max())
                    report.add(f"{label}[invalid=0]", variant, worst, 0.0)
            else:
                delta = float(np.abs(_sigmoid(ref) - _sigmoid(got)).max())
                report.add(label, variant, delta, TOL_FP16_PROB)
                key = (text_len, case.span_lo)
                fp16_by_length[key] = max(fp16_by_length.get(key, 0.0), delta)

    # ── the `float(max(L, 1))` guard: two different L, one set of weights ────
    # Same states, same pairs, same graph — only the denominator moves. If the
    # denominator were baked in at export time, ONNX would agree with PyTorch at
    # exactly one length and the "live" row would collapse to zero.
    big_len, small_len, num_relations, num_pairs = 97, 41, 2, 24
    big = torch.randn(1, big_len, H)
    big[:, small_len:, :] = 0.0                      # zero-padded past the words
    small = big[:, :small_len, :].contiguous()
    query_head = torch.randn(1, num_relations, H)
    query_tail = torch.randn(1, num_relations, H)
    pairs, _ = _relation_pairs(num_pairs, num_relations, small_len, rng)

    session32 = _session(onnx_dir / "relation_scorer_fp32.onnx")
    ref_big = _relation_reference(scorer, big, query_head, query_tail, pairs)
    ref_small = _relation_reference(scorer, small, query_head, query_tail, pairs)
    got_big = _run(session32, _relation_feeds(
        big, query_head, query_tail, pairs, text_len=big_len))[0]
    got_small = _run(session32, _relation_feeds(
        small, query_head, query_tail, pairs, text_len=small_len))[0]
    # `text_states` padded to 97, denominator stated as 41: the ONLY channel
    # through which padding reaches the score is `text_len`, so this must
    # reproduce PyTorch-at-41 exactly.
    got_cross = _run(session32, _relation_feeds(
        big, query_head, query_tail, pairs, text_len=small_len))[0]

    report.add(f"relation_scorer[L={big_len} dyn]", "fp32",
               float(np.abs(ref_big - got_big).max()), TOL_FP32)
    report.add(f"relation_scorer[L={small_len} dyn]", "fp32",
               float(np.abs(ref_small - got_small).max()), TOL_FP32)
    report.add("relation text_len override", "fp32",
               float(np.abs(ref_small - got_cross).max()), TOL_FP32)
    report.add_at_least("relation text_len live", "fp32",
                        float(np.abs(got_big - got_small).max()), TOL_TEXT_LEN_LIVE)
    report.add_at_least("relation text_len live", "pytorch",
                        float(np.abs(ref_big - ref_small).max()), TOL_TEXT_LEN_LIVE)

    # ── P == 0: Python early-returns before touching the model ──────────────
    # `relations.py:335-336`. The exported graph was traced with num_pairs >= 1
    # and CANNOT run at P == 0, so the Rust port must early-return too; the
    # literal outcome is printed rather than assumed.
    empty, _ = _relation_pairs(0, 1, 8, rng)
    ref_empty = _relation_reference(scorer, torch.randn(1, 64, H),
                                    torch.randn(1, 1, H), torch.randn(1, 1, H), empty)
    report.add("relation P=0 early return", "pytorch",
               float(abs(ref_empty.shape[0])), 0.0)

    ort.set_default_logger_severity(4)   # the probe below is expected to raise
    try:
        got_empty = _run(session32, _relation_feeds(
            torch.randn(1, 64, H), torch.randn(1, 1, H), torch.randn(1, 1, H),
            empty, text_len=64))[0]
        # Accepting P=0 is fine only if the answer is an empty tensor; anything
        # else would be silent garbage.
        p0_ok = got_empty.shape == (0,)
        onnx_p0 = f"the ONNX graph ACCEPTED P=0 and returned {tuple(got_empty.shape)}"
    except Exception as exc:                                   # noqa: BLE001
        p0_ok = True
        onnx_p0 = f"the ONNX graph REJECTED P=0 ({type(exc).__name__})"
    finally:
        ort.set_default_logger_severity(2)
    report.add("relation P=0 guard", "fp32", 0.0 if p0_ok else 1.0, 0.0)
    print(f"\n  P=0: PyTorch returns shape {tuple(ref_empty.shape)} without running the "
          f"model; {onnx_p0}.")
    print("       => the Rust port MUST early-return on an empty proposal set "
          "(relations.py:335-336).")

    # ── fp16 error vs padded length: the biaffine cumsum prefix ─────────────
    # `relations.py:378-392` runs the prefix sum over all `L` rows in fp16, so a
    # longer padded sequence carries larger prefix values and loses more
    # mantissa. Printed per (length, span depth) so a future regression in the
    # margin is visible rather than averaged away. Measured here: the length
    # effect is clear (20.6x margin at L=64 down to 8.6x at L=512); span depth
    # alone does NOT order the error, so the printed note says so.
    print(f"\n  fp16 |dprob| by padded length and span depth (tolerance "
          f"{TOL_FP16_PROB:.0e}):")
    print(f"    {'padded L':>8} {'spans from':>11} {'|dprob|':>11} {'margin':>9}")
    for length, span_lo in sorted(fp16_by_length):
        delta = fp16_by_length[(length, span_lo)]
        print(f"    {length:>8d} {span_lo:>11d} {delta:>11.3e} "
              f"{TOL_FP16_PROB / max(delta, 1e-12):>8.1f}x")
    print("    The margin narrows as the PADDED length grows: the biaffine branch's")
    print("    fp16 cumsum prefix (relations.py:378-392) runs over all L rows, so a")
    print("    longer sequence carries larger prefix values and loses more mantissa.")
    print("    Span depth on its own does NOT order the error in this sampling — the")
    print("    post-sigmoid delta also depends on where each logit sits relative to")
    print("    the sigmoid transition. Read the L column, not the depth column. A")
    print("    change that eats into these margins shows up here before it breaks")
    print("    the tolerance.")

    return report


def main() -> int:
    p = argparse.ArgumentParser(description="PyTorch vs ONNX parity (boundary)")
    p.add_argument("--model_path", required=True,
                   help="Local path or HuggingFace repo id of the boundary checkpoint")
    p.add_argument("--onnx_dir", required=True, help="Directory holding the ONNX export")
    args = p.parse_args()

    onnx_dir = Path(args.onnx_dir)
    print(f"checkpoint : {args.model_path}")
    print(f"onnx       : {onnx_dir}")

    report = verify_boundary(args.model_path, onnx_dir)
    relations = verify_relation_scorer(args.model_path, onnx_dir)
    report.rows.extend(relations.rows)
    report.skips.extend(relations.skips)
    report.show()
    return 1 if report.failed() else 0


if __name__ == "__main__":
    sys.exit(main())
