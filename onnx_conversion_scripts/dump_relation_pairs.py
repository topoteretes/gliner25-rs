#!/usr/bin/env python3
"""Dump Python `RelationPairBatch` fixtures for the Rust proposal-identity gate.

Stage 7, sub-stage 7a-3.  The fixture this writes is consumed by 7b-2, which
asserts **index-for-index** identity between

    gliner2.models.boundary.relations.TypedRelationPairGenerator.generate(...)

and the Rust port's `generate_pairs` / `generate_pairs_detailed`
(`crates/gliner25-rs/src/relations.rs`).  That gate is the blocking one for the
whole relation port: if the two proposal sets differ, the scorer cannot help
and the failure is *silent* — fragment parity still passes, entity parity stays
1.000, every test stays green, and the relation numbers simply never improve.

So the fixture is built to **localise** a mismatch, not merely detect one:

* `inputs`      — raw i64 candidate spans and raw f32 `pair_logits`, so Rust can
                  be driven with nothing decoded and no ONNX session in the loop.
* `proposals[]` — the intermediate top-32 `heads` and top-32 `tails` argument
                  lists, one record per slot, carrying `flat_index`, `query`,
                  `cand_slot`, the raw i64 span, the prob and the valid flag.
                  This is the shape `generate_pairs_detailed` exposes, so a
                  mismatch lands on argument *selection* rather than only
                  showing up as a different span three stages later.
* `pair_batch`  — Python's compacted `RelationPairBatch`, every field, in order.

### The padded-length trap (stage-7 BRIEF, Risk 2)

`SparseRelationScorer` divides the raw word distance by
`boundary_states.shape[1]` (`relations.py:374`) — the **padded** length, with no
mask.  Python pads to the batch max; Rust pads to a 64/128/256/512 bucket.  Two
different numbers there move every score while every existing test stays green.

Each case therefore carries **`boundary_states_padded_len`** and **`num_words`**
as two separately named fields.  There is deliberately no field called
`length`, `len` or `L` that a reader could take for either.

`TypedRelationPairGenerator` itself never reads the padded length — pair
*proposal* is pure index arithmetic over the candidate tensors.  The field is
carried here anyway because 7b-3 feeds the same fixture into the scorer, and
that is the stage where getting it wrong is invisible.

### Intermediates: how they are obtained

`select()` is a closure inside `generate_batched` and does not return the flat
index it ranked, so the intermediates cannot be read off the public API.  This
file carries `_instrumented_generate`, a copy of upstream `generate_batched`
whose `select()` additionally returns `ranked`.  Every case then asserts that
the mirror's compacted output is **tensor-for-tensor equal** to the real
`TypedRelationPairGenerator.generate(...)` (see `_assert_mirror_matches`), so
the intermediates are only trusted for cases where the mirror provably agrees
with the library on the observable output.  A divergence aborts the dump.

Run:

    /home/dmytro/dev/cognee/gliner2-candle/.venv/bin/python \
        onnx_conversion_scripts/dump_relation_pairs.py

Needs only `torch` + `gliner2` (no onnx, no onnxruntime, no checkpoint, no
network).  Output is deterministic: regenerating twice is byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from typing import Dict, List, NamedTuple, Sequence, Tuple

import torch
import torch.nn.functional as F

from gliner2.models.boundary.relations import (
    RelationProposalSettings,
    RelationTypeSpec,
    TypedRelationPairGenerator,
)
from gliner2.models.outputs import CandidateTensorBatch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(
    REPO_ROOT, "onnx_conversion_scripts", "fixtures", "relation_pairs.json"
)
RELATION_SETTINGS = os.path.join(
    REPO_ROOT, "models", "gliner2.5-base-v1-onnx", "relation_settings.json"
)

SCHEMA = "gliner25.relation_pairs.v1"


# --------------------------------------------------------------------------
# settings — read from the exporter's dump of the checkpoint, never defaults
# --------------------------------------------------------------------------


def load_settings() -> Tuple[RelationProposalSettings, dict]:
    """Read the four proposal knobs from `relation_settings.json`.

    Deliberately not `RelationProposalSettings()`: the library defaults are
    `pair_cap=128` / `argument_threshold=0.0`, and those are exactly the two
    values the checkpoint overrides.  A missing key is a hard error.
    """
    with open(RELATION_SETTINGS, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    required = (
        "relation_heads_per_type",
        "relation_tails_per_type",
        "relation_pair_cap",
        "relation_argument_proposal_threshold",
    )
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"{RELATION_SETTINGS} is missing {missing}")
    settings = RelationProposalSettings(
        heads_per_relation=int(raw["relation_heads_per_type"]),
        tails_per_relation=int(raw["relation_tails_per_type"]),
        pair_cap=int(raw["relation_pair_cap"]),
        argument_threshold=float(raw["relation_argument_proposal_threshold"]),
    )
    meta = {
        "heads_per_relation": settings.heads_per_relation,
        "tails_per_relation": settings.tails_per_relation,
        "pair_cap": settings.pair_cap,
        "argument_threshold": settings.argument_threshold,
        "abstention_threshold_not_used_here": raw.get("abstention_threshold"),
        "source": "models/gliner2.5-base-v1-onnx/relation_settings.json",
    }
    return settings, meta


# --------------------------------------------------------------------------
# the instrumented mirror of `generate_batched`
# --------------------------------------------------------------------------


class SelectTrace(NamedTuple):
    """One `select()` call's output, per relation, for a single-sample batch."""

    flat_index: List[int]  # `ranked`, after F.pad — length == requested
    query: List[int]  # qslot
    cand_slot: List[int]  # cslot
    start: List[int]  # raw i64
    end: List[int]  # raw i64
    prob: List[float]
    valid: List[bool]


def _instrumented_generate(
    generator: TypedRelationPairGenerator,
    candidates: CandidateTensorBatch,
    relation_schema: Sequence[RelationTypeSpec],
):
    """Verbatim `generate_batched` (gliner2 2.0.0) + `ranked` escaping `select`.

    Only two edits versus upstream `relations.py:104-290`:
      1. `select()` also returns `ranked`, `qslot` and `cslot`.
      2. the Python-key materialisation loop is dropped (it needs a QueryLayout
         and produces only presentation metadata); the caller rebuilds the keys
         from the returned `hq_out` / `tq_out`.
    `_assert_mirror_matches` pins the mirror to the library output per case.
    """
    s = generator.settings
    device = candidates.indices.device
    bsz, queries, cand_count = candidates.valid_mask.shape
    relation_schemas = [relation_schema for _ in range(bsz)]
    rel_count = max((len(x) for x in relation_schemas), default=0)
    if rel_count == 0:
        raise ValueError("the fixture never dumps a zero-relation case")

    # decode-only routing fallback, relations.py:134-153
    head_member = torch.zeros(bsz, rel_count, queries, dtype=torch.bool, device=device)
    tail_member = torch.zeros_like(head_member)
    relation_valid = torch.zeros(bsz, rel_count, dtype=torch.bool, device=device)
    allow_self = torch.zeros_like(relation_valid)
    for batch_index, schemas in enumerate(relation_schemas):
        for relation_index, spec in enumerate(schemas):
            relation_valid[batch_index, relation_index] = True
            allow_self[batch_index, relation_index] = spec.allow_self
            valid_h = [q for q in spec.head_query_ids if 0 <= q < queries]
            valid_t = [q for q in spec.tail_query_ids if 0 <= q < queries]
            if valid_h:
                head_member[batch_index, relation_index, valid_h] = True
            if valid_t:
                tail_member[batch_index, relation_index, valid_t] = True

    probs = torch.sigmoid(candidates.pair_logits)
    base_valid = candidates.valid_mask & candidates.query_mask.unsqueeze(-1)
    flat_prob = probs.reshape(bsz, 1, queries * cand_count).expand(-1, rel_count, -1)
    flat_valid = base_valid.reshape(bsz, 1, -1).expand(-1, rel_count, -1)
    head_valid = flat_valid & head_member.unsqueeze(-1).expand(
        -1, -1, -1, cand_count
    ).reshape(bsz, rel_count, -1)
    tail_valid = flat_valid & tail_member.unsqueeze(-1).expand(
        -1, -1, -1, cand_count
    ).reshape(bsz, rel_count, -1)
    threshold = flat_prob >= s.argument_threshold
    head_valid = head_valid & threshold
    tail_valid = tail_valid & threshold
    floor = torch.finfo(flat_prob.dtype).min
    flat_spans = candidates.indices.reshape(bsz, queries * cand_count, 2)

    def select(valid, requested: int):
        take = min(requested, queries * cand_count)
        secondary = (
            torch.arange(queries * cand_count, device=device)
            .view(1, 1, -1)
            .expand(bsz, rel_count, -1)
        )
        end_key = flat_spans[..., 1].unsqueeze(1).expand(-1, rel_count, -1)
        end_order = torch.argsort(end_key.gather(-1, secondary), dim=-1, stable=True)
        secondary = secondary.gather(-1, end_order)
        start_key = flat_spans[..., 0].unsqueeze(1).expand(-1, rel_count, -1)
        start_order = torch.argsort(
            start_key.gather(-1, secondary), dim=-1, stable=True
        )
        secondary = secondary.gather(-1, start_order)
        ordered_score = flat_prob.gather(-1, secondary)
        ordered_valid = valid.gather(-1, secondary)
        rank_in_secondary = torch.argsort(
            ordered_score.masked_fill(~ordered_valid, floor),
            dim=-1,
            descending=True,
            stable=True,
        )[..., :take]
        ranked = secondary.gather(-1, rank_in_secondary)
        selected_valid = valid.gather(-1, ranked)
        selected_prob = flat_prob.gather(-1, ranked)
        if take < requested:
            pad = requested - take
            # NB: F.pad fills the *flat index* with 0, so a padded slot carries
            # candidate 0's span, not (0, 0).  7b-1 found the plan had this
            # wrong; it is visible to any fixture test comparing heads/tails.
            ranked = F.pad(ranked, (0, pad))
            selected_valid = F.pad(selected_valid, (0, pad), value=False)
            selected_prob = F.pad(selected_prob, (0, pad))
        qslot = torch.div(ranked, cand_count, rounding_mode="floor")
        cslot = ranked - qslot * cand_count
        qslot = qslot.clamp(0, queries - 1)
        cslot = cslot.clamp(0, cand_count - 1)
        batch = torch.arange(bsz, device=device)[:, None, None]
        spans = candidates.indices[batch, qslot, cslot]
        return selected_prob, qslot, spans, selected_valid, ranked, cslot

    hp, hq, hspan, hvalid, h_ranked, h_cslot = select(head_valid, s.heads_per_relation)
    tp, tq, tspan, tvalid, t_ranked, t_cslot = select(tail_valid, s.tails_per_relation)
    pair_score = hp.unsqueeze(-1) * tp.unsqueeze(-2)
    pair_valid = hvalid.unsqueeze(-1) & tvalid.unsqueeze(-2)
    same_span = (hspan.unsqueeze(-2) == tspan.unsqueeze(-3)).all(-1)
    pair_valid = pair_valid & (allow_self[..., None, None] | ~same_span)
    pair_valid = pair_valid & relation_valid[..., None, None]
    flat_pair_score = pair_score.flatten(2)
    flat_pair_valid = pair_valid.flatten(2)
    take = min(s.pair_cap, flat_pair_score.shape[-1])
    keep = torch.argsort(
        flat_pair_score.masked_fill(~flat_pair_valid, floor),
        dim=-1,
        descending=True,
        stable=True,
    )[..., :take]
    kept_valid = flat_pair_valid.gather(-1, keep)
    if take < s.pair_cap:
        keep = F.pad(keep, (0, s.pair_cap - take))
        kept_valid = F.pad(kept_valid, (0, s.pair_cap - take), value=False)
    hi = torch.div(keep, s.tails_per_relation, rounding_mode="floor")
    ti = keep - hi * s.tails_per_relation
    hi = hi.clamp(0, s.heads_per_relation - 1)
    ti = ti.clamp(0, s.tails_per_relation - 1)

    def gather_selected(values, index):
        return values.gather(
            2,
            index.clamp(0, values.shape[2] - 1)
            .unsqueeze(-1)
            .expand(*index.shape, values.shape[-1]),
        )

    hs = gather_selected(hspan, hi)
    ts = gather_selected(tspan, ti)
    hp_out = hp.gather(2, hi.clamp(0, hp.shape[2] - 1))
    tp_out = tp.gather(2, ti.clamp(0, tp.shape[2] - 1))
    hq_out = hq.gather(2, hi.clamp(0, hq.shape[2] - 1))
    tq_out = tq.gather(2, ti.clamp(0, tq.shape[2] - 1))
    bi = torch.arange(bsz, device=device)[:, None, None].expand_as(keep)
    ri = torch.arange(rel_count, device=device)[None, :, None].expand_as(keep)

    flat_mask = kept_valid.reshape(-1)
    tensors = [
        bi.reshape(-1),
        ri.reshape(-1),
        hs[..., 0].reshape(-1),
        hs[..., 1].reshape(-1),
        ts[..., 0].reshape(-1),
        ts[..., 1].reshape(-1),
        hp_out.reshape(-1),
        tp_out.reshape(-1),
        hq_out.reshape(-1),
        tq_out.reshape(-1),
    ]
    compact = [value[flat_mask] for value in tensors]

    return {
        "compact": compact,
        # per-relation traces are rebuilt by the caller (`_trace`) from these.
        "raw": {
            "h_ranked": h_ranked,
            "h_q": hq,
            "h_c": h_cslot,
            "h_span": hspan,
            "h_prob": hp,
            "h_valid": hvalid,
            "t_ranked": t_ranked,
            "t_q": tq,
            "t_c": t_cslot,
            "t_span": tspan,
            "t_prob": tp,
            "t_valid": tvalid,
            "keep_hi": hi,
            "keep_ti": ti,
            "keep_valid": kept_valid,
            "rel_count": rel_count,
        },
    }


def _trace(raw, relation_index: int, prefix: str) -> SelectTrace:
    ranked = raw[f"{prefix}_ranked"][0, relation_index]
    qslot = raw[f"{prefix}_q"][0, relation_index]
    cslot = raw[f"{prefix}_c"][0, relation_index]
    span = raw[f"{prefix}_span"][0, relation_index]
    prob = raw[f"{prefix}_prob"][0, relation_index]
    valid = raw[f"{prefix}_valid"][0, relation_index]
    return SelectTrace(
        flat_index=[int(x) for x in ranked],
        query=[int(x) for x in qslot],
        cand_slot=[int(x) for x in cslot],
        start=[int(x) for x in span[:, 0]],
        end=[int(x) for x in span[:, 1]],
        prob=[float(x) for x in prob],
        valid=[bool(x) for x in valid],
    )


def _assert_mirror_matches(mirror_compact, reference) -> None:
    """The mirror must agree with the library on every observable field."""
    fields = [
        ("batch_index", reference.batch_index),
        ("relation_index", reference.relation_index),
        ("head_start", reference.head_start),
        ("head_end", reference.head_end),
        ("tail_start", reference.tail_start),
        ("tail_end", reference.tail_end),
        ("head_prob", reference.head_prob),
        ("tail_prob", reference.tail_prob),
    ]
    for position, (name, expected) in enumerate(fields):
        got = mirror_compact[position]
        if got.shape != expected.shape or not torch.equal(got, expected):
            raise AssertionError(
                f"instrumented mirror diverged from TypedRelationPairGenerator "
                f"on `{name}`: mirror={got.tolist()} library={expected.tolist()}"
            )


# --------------------------------------------------------------------------
# case definitions
# --------------------------------------------------------------------------


class Case(NamedTuple):
    name: str
    intent: str
    # `num_words` and `boundary_states_padded_len` are two different numbers and
    # neither is named `length`.  See the module docstring (Risk 2).
    num_words: int
    boundary_states_padded_len: int
    queries: int
    cand_count: int
    spans: List[Tuple[int, int]]  # flat j order, length queries*cand_count
    logits: List[float]  # flat j order, raw f32 pair_logits
    valid_mask: List[bool]  # flat j order
    query_mask: List[bool]  # length queries
    specs: List[RelationTypeSpec]
    expect_zero_pairs: bool = False
    expect_ties: bool = False


def f32(value: float) -> float:
    """Round a Python float to the nearest f32, so the dump is exact."""
    return float(torch.tensor(value, dtype=torch.float32).item())


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _random_spans(rng: random.Random, count: int, num_words: int) -> List[Tuple[int, int]]:
    spans = []
    for _ in range(count):
        start = rng.randrange(0, max(num_words - 1, 1))
        width = rng.randrange(1, min(4, num_words - start) + 1)
        spans.append((start, start + width))
    return spans


def build_cases() -> List[Case]:
    cases: List[Case] = []

    # ---- 1. small pool: Q*C = 18 < 32, so `select` takes the F.pad branch ----
    rng = random.Random(20260917)
    q, c, num_words = 3, 6, 40
    n = q * c
    spans = _random_spans(rng, n, num_words)
    logits = [f32(rng.uniform(-3.0, 3.0)) for _ in range(n)]
    valid = [True] * n
    for j in (4, 11, 17):
        valid[j] = False
    cases.append(
        Case(
            name="small_pool_padded_l64",
            intent=(
                "Q*C = 18 < heads_per_relation, so select() pads: the tail of "
                "heads/tails carries candidate 0's span (F.pad fills the flat "
                "index with 0) at prob 0.0 and valid=false. Query 2 is masked "
                "off; three candidates are invalid. text_len != num_words."
            ),
            num_words=num_words,
            boundary_states_padded_len=64,
            queries=q,
            cand_count=c,
            spans=spans,
            logits=logits,
            valid_mask=valid,
            query_mask=[True, True, False],
            specs=[
                RelationTypeSpec("works_for", (0,), (1,), False),
                RelationTypeSpec("headquartered_in", (1,), (0, 2), False),
            ],
        )
    )

    # ---- 2. wide pool at the deepest bucket: Q*C = 64, take == 32 ----
    rng = random.Random(512512)
    q, c, num_words = 4, 16, 300
    n = q * c
    spans = _random_spans(rng, n, num_words)
    logits = [f32(rng.uniform(-1.0, 4.0)) for _ in range(n)]
    valid = [True] * n
    for j in (7, 23, 40, 61):
        valid[j] = False
    cases.append(
        Case(
            name="wide_pool_l512",
            intent=(
                "Q*C = 64 >= heads_per_relation, so take == 32 and no padding "
                "happens. 32x32 = 1024 candidate pairs collapse to pair_cap=64, "
                "exercising the keep-sort truncation. Padded L is the deepest "
                "bucket (512) and text_len != num_words (300)."
            ),
            num_words=num_words,
            boundary_states_padded_len=512,
            queries=q,
            cand_count=c,
            spans=spans,
            logits=logits,
            valid_mask=valid,
            query_mask=[True, True, True, True],
            specs=[
                RelationTypeSpec("works_for", (0,), (1,), False),
                RelationTypeSpec("acquired", (1,), (1,), False),
                # query id 9 is out of range and must be DROPPED, not clamped
                # onto query 0 (relations.py:143-144).
                RelationTypeSpec("produces", (1, 9), (2, 3), False),
            ],
        )
    )

    # ---- 3. text_len == num_words, the contrast case ----
    rng = random.Random(128128)
    q, c, num_words = 2, 20, 128
    n = q * c
    spans = _random_spans(rng, n, num_words)
    logits = [f32(rng.uniform(-2.5, 2.5)) for _ in range(n)]
    valid = [True] * n
    valid[0] = False  # candidate 0 invalid: padded slots still borrow its span
    cases.append(
        Case(
            name="exact_len_l128",
            intent=(
                "boundary_states_padded_len == num_words == 128, the contrast "
                "against the three cases where they differ. Candidate 0 is "
                "invalid, so a padded slot borrows an invalid candidate's span."
            ),
            num_words=num_words,
            boundary_states_padded_len=128,
            queries=q,
            cand_count=c,
            spans=spans,
            logits=logits,
            valid_mask=valid,
            query_mask=[True, True],
            specs=[RelationTypeSpec("founded_by", (0,), (1,), False)],
        )
    )

    # ---- 4. TOTAL tie: every prob is exactly 0.5 ----
    # sigmoid(0.0) == 0.5 exactly in f32 and 0.5*0.5 == 0.25 exactly, so every
    # argument score and every pair score is bit-identical.  The entire top-32
    # of a 50-candidate pool, and the entire top-64 of 1024 pairs, is therefore
    # decided by the tie-break chain alone:
    #   prob DESC -> start ASC -> end ASC -> flat index ASC
    # and the pair chain: pair prob DESC -> head rank ASC -> tail rank ASC.
    q, c, num_words = 5, 10, 40
    n = q * c
    spans: List[Tuple[int, int]] = []
    for j in range(n):
        # Deliberate collisions, three levels deep:
        #   - many candidates share a start
        #   - some share (start, end) exactly, differing only in flat index
        start = (j // 5) * 3 % 24
        end = start + 1 + (j % 3)
        spans.append((start, end))
    logits = [0.0] * n
    valid = [True] * n
    for j in (13, 37):
        valid[j] = False
    cases.append(
        Case(
            name="tied_scores_total_l64",
            intent=(
                "EVERY pair_logit is 0.0, so every prob is exactly 0.5 and every "
                "pair score is exactly 0.25 in f32 - no float tolerance is "
                "involved. Spans collide at three levels (shared start, shared "
                "(start,end), distinct flat index only), so which 32 of the 50 "
                "candidates survive, and which 64 of the 1024 pairs survive, is "
                "decided ENTIRELY by the tie-break chain. This is the case that "
                "catches a naive `sort_by(score desc)` port."
            ),
            num_words=num_words,
            boundary_states_padded_len=64,
            queries=q,
            cand_count=c,
            spans=spans,
            logits=logits,
            valid_mask=valid,
            query_mask=[True] * q,
            specs=[
                RelationTypeSpec("works_for", (0, 1), (2, 3), False),
                RelationTypeSpec("acquired", (2,), (2, 4), False),
            ],
            expect_ties=True,
        )
    )

    # ---- 5. banded ties: three exact score levels ----
    # Every candidate in a band shares one logit, so it shares one f32 prob
    # bit-for-bit.  Ties are exact *within* a band and the bands order the
    # argument list, so the chain is exercised without the whole list degenerate.
    q, c, num_words = 4, 12, 64
    n = q * c
    bands = [f32(_logit(0.9)), f32(_logit(0.6)), f32(_logit(0.25))]
    spans = []
    logits = []
    for j in range(n):
        start = (j * 7) % 30
        end = start + 1 + ((j // 6) % 3)
        spans.append((start, end))
        logits.append(bands[j % 3])
    valid = [True] * n
    for j in (2, 19, 44):
        valid[j] = False
    cases.append(
        Case(
            name="tied_scores_banded_l128",
            intent=(
                "Three exact probability bands (0.9 / 0.6 / 0.25 as f32). Every "
                "candidate in a band shares one logit, so it shares one f32 prob "
                "bit-for-bit: ties are exact within a band while the bands "
                "themselves order the list. Band 0.25 sits just above the 0.2 "
                "argument threshold and stays eligible."
            ),
            num_words=num_words,
            boundary_states_padded_len=128,
            queries=q,
            cand_count=c,
            spans=spans,
            logits=logits,
            valid_mask=valid,
            query_mask=[True, True, True, True],
            specs=[
                RelationTypeSpec("works_for", (0,), (1, 2), False),
                RelationTypeSpec("produces", (1,), (3,), False),
            ],
            expect_ties=True,
        )
    )

    # ---- 6. P = 0 via the argument threshold ----
    # sigmoid(-5.0) ~= 0.0067 < 0.2, so nothing is eligible, every head/tail
    # slot is floored+invalid and `compact` drops all 64 kept pairs.
    q, c, num_words = 3, 8, 200
    n = q * c
    rng = random.Random(7070)
    spans = _random_spans(rng, n, num_words)
    cases.append(
        Case(
            name="empty_below_threshold_l512",
            intent=(
                "Every pair_logit is -5.0 (prob ~ 0.0067 < the 0.2 argument "
                "threshold), so no argument is eligible, every kept pair is "
                "invalid and the compacted batch has ZERO rows. 7a-2 found the "
                "exported ONNX graph cannot run at P=0 (num_pairs>=1 baked in at "
                "tracing; ORT fails in node_add_163) while Python early-returns "
                "at relations.py:335-336 - so Rust must early-return here too. "
                "Padded L is 512 and text_len != num_words (200)."
            ),
            num_words=num_words,
            boundary_states_padded_len=512,
            queries=q,
            cand_count=c,
            spans=spans,
            logits=[f32(-5.0)] * n,
            valid_mask=[True] * n,
            query_mask=[True, True, True],
            specs=[
                RelationTypeSpec("works_for", (0,), (1,), False),
                RelationTypeSpec("acquired", (1,), (2,), False),
            ],
            expect_zero_pairs=True,
        )
    )

    # ---- 7. P = 0 via same_span, a different route to the empty batch ----
    q, c, num_words = 1, 4, 12
    cases.append(
        Case(
            name="empty_self_span_only_l64",
            intent=(
                "A second, structurally different P=0: exactly one eligible "
                "candidate, in both the head and the tail query set, with "
                "allow_self=false. The only scoreable pair is the span with "
                "itself, which same_span kills (relations.py:216 compares BOTH "
                "endpoints), so the compacted batch is empty even though the "
                "argument pool is not."
            ),
            num_words=num_words,
            boundary_states_padded_len=64,
            queries=q,
            cand_count=c,
            spans=[(2, 4), (0, 1), (5, 7), (9, 11)],
            logits=[f32(3.0), f32(-9.0), f32(-9.0), f32(-9.0)],
            valid_mask=[True, True, True, True],
            query_mask=[True],
            specs=[RelationTypeSpec("acquired", (0,), (0,), False)],
            expect_zero_pairs=True,
        )
    )

    return cases


# --------------------------------------------------------------------------
# emit
# --------------------------------------------------------------------------


class Inline:
    """Wrapper marking a dict/list that must be emitted on a single line."""

    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value


def _emit(node, indent: int, out: List[str]) -> None:
    pad = " " * indent
    if isinstance(node, Inline):
        out.append(json.dumps(node.value, ensure_ascii=True, allow_nan=False))
        return
    if isinstance(node, dict):
        if not node:
            out.append("{}")
            return
        out.append("{\n")
        items = list(node.items())
        for position, (key, value) in enumerate(items):
            out.append(pad + "  " + json.dumps(key) + ": ")
            _emit(value, indent + 2, out)
            out.append(",\n" if position + 1 < len(items) else "\n")
        out.append(pad + "}")
        return
    if isinstance(node, list):
        if not node:
            out.append("[]")
            return
        out.append("[\n")
        for position, value in enumerate(node):
            out.append(pad + "  ")
            _emit(value, indent + 2, out)
            out.append(",\n" if position + 1 < len(node) else "\n")
        out.append(pad + "]")
        return
    out.append(json.dumps(node, ensure_ascii=True, allow_nan=False))


def render(document: dict) -> str:
    out: List[str] = []
    _emit(document, 0, out)
    out.append("\n")
    return "".join(out)


# --------------------------------------------------------------------------
# drive one case
# --------------------------------------------------------------------------


def run_case(case: Case, settings: RelationProposalSettings, generator) -> dict:
    n = case.queries * case.cand_count
    for name, seq, want in (
        ("spans", case.spans, n),
        ("logits", case.logits, n),
        ("valid_mask", case.valid_mask, n),
        ("query_mask", case.query_mask, case.queries),
    ):
        if len(seq) != want:
            raise ValueError(f"{case.name}: {name} has {len(seq)}, expected {want}")
    for start, end in case.spans:
        if not (0 <= start < end <= case.num_words):
            raise ValueError(f"{case.name}: span ({start},{end}) escapes num_words")
    if case.num_words > case.boundary_states_padded_len:
        raise ValueError(
            f"{case.name}: num_words {case.num_words} exceeds "
            f"boundary_states_padded_len {case.boundary_states_padded_len}"
        )

    indices = torch.tensor(case.spans, dtype=torch.long).view(
        1, case.queries, case.cand_count, 2
    )
    pair_logits = torch.tensor(case.logits, dtype=torch.float32).view(
        1, case.queries, case.cand_count
    )
    valid_mask = torch.tensor(case.valid_mask, dtype=torch.bool).view(
        1, case.queries, case.cand_count
    )
    query_mask = torch.tensor(case.query_mask, dtype=torch.bool).view(1, case.queries)
    candidates = CandidateTensorBatch(
        indices=indices,
        proposal_logits=None,
        pair_logits=pair_logits,
        valid_mask=valid_mask,
        query_mask=query_mask,
    )

    # `query_layouts=[]` -> `_query_type` is never reached and the Python keys
    # fall back to `str(query_id)` (relations.py:272-289).  Deterministic, and
    # it keeps the fixture independent of any checkpoint's label vocabulary.
    reference = generator.generate(candidates, [], list(case.specs), compact=True)
    mirror = _instrumented_generate(generator, candidates, list(case.specs))
    _assert_mirror_matches(mirror["compact"], reference)

    raw = mirror["raw"]
    proposals = []
    for relation_index, spec in enumerate(case.specs):
        heads = _trace(raw, relation_index, "h")
        tails = _trace(raw, relation_index, "t")
        # `requested`, not `take`: the list length never shrinks with the pool.
        if len(heads.flat_index) != settings.heads_per_relation:
            raise AssertionError(f"{case.name}: heads list is not `requested` long")
        if len(tails.flat_index) != settings.tails_per_relation:
            raise AssertionError(f"{case.name}: tails list is not `requested` long")
        hi = raw["keep_hi"][0, relation_index]
        ti = raw["keep_ti"][0, relation_index]
        keep_valid = raw["keep_valid"][0, relation_index]

        def slots(trace: SelectTrace) -> List[Inline]:
            return [
                Inline(
                    {
                        "slot": slot,
                        "flat_index": trace.flat_index[slot],
                        "query": trace.query[slot],
                        "cand_slot": trace.cand_slot[slot],
                        "start": trace.start[slot],
                        "end": trace.end[slot],
                        "prob": trace.prob[slot],
                        "valid": trace.valid[slot],
                    }
                )
                for slot in range(len(trace.flat_index))
            ]

        pairs = []
        for position in range(len(keep_valid)):
            if not bool(keep_valid[position]):
                continue  # compact=True drops invalid survivors (relations.py:249)
            head_slot = int(hi[position])
            tail_slot = int(ti[position])
            pairs.append(
                Inline(
                    {
                        "pair_index": len(pairs),
                        "keep_position": position,
                        "head_slot": head_slot,
                        "tail_slot": tail_slot,
                        "head_start": heads.start[head_slot],
                        "head_end": heads.end[head_slot],
                        "tail_start": tails.start[tail_slot],
                        "tail_end": tails.end[tail_slot],
                        "head_prob": heads.prob[head_slot],
                        "tail_prob": tails.prob[tail_slot],
                        "head_query": heads.query[head_slot],
                        "tail_query": tails.query[tail_slot],
                    }
                )
            )

        proposals.append(
            {
                "relation_index": relation_index,
                "relation_type": spec.relation_type,
                "allow_self": spec.allow_self,
                "head_query_ids": Inline(list(spec.head_query_ids)),
                "tail_query_ids": Inline(list(spec.tail_query_ids)),
                "heads": slots(heads),
                "tails": slots(tails),
                "num_pairs": len(pairs),
                "pairs": pairs,
            }
        )

    # Python's compacted RelationPairBatch, every field, in order.
    rows = []
    for index in range(len(reference)):
        rows.append(
            Inline(
                {
                    "index": index,
                    "batch_index": int(reference.batch_index[index]),
                    "relation_index": int(reference.relation_index[index]),
                    "head_start": int(reference.head_start[index]),
                    "head_end": int(reference.head_end[index]),
                    "tail_start": int(reference.tail_start[index]),
                    "tail_end": int(reference.tail_end[index]),
                    "head_prob": float(reference.head_prob[index]),
                    "tail_prob": float(reference.tail_prob[index]),
                    "pair_mask": bool(reference.pair_mask[index]),
                    "relation_type": reference.relation_types[index],
                    "head_key": Inline(list(reference.head_keys[index])).value,
                    "tail_key": Inline(list(reference.tail_keys[index])).value,
                }
            )
        )

    total_pairs = sum(proposal["num_pairs"] for proposal in proposals)
    if total_pairs != len(rows):
        raise AssertionError(
            f"{case.name}: per-relation pairs {total_pairs} != batch rows {len(rows)}"
        )
    if case.expect_zero_pairs and rows:
        raise AssertionError(f"{case.name}: expected P=0, got {len(rows)}")
    if not case.expect_zero_pairs and not rows:
        raise AssertionError(f"{case.name}: expected P>0, got an empty batch")

    tie_stats = _tie_stats(proposals) if case.expect_ties else None
    if case.expect_ties and tie_stats["max_pair_score_multiplicity"] < 2:
        raise AssertionError(f"{case.name}: claimed ties but found none")

    record = {
        "name": case.name,
        "intent": case.intent,
        # --- the padded-length trap: two fields, neither called "length" ---
        "num_words": case.num_words,
        "boundary_states_padded_len": case.boundary_states_padded_len,
        "padded_len_equals_num_words": case.boundary_states_padded_len
        == case.num_words,
        "queries": case.queries,
        "cand_count": case.cand_count,
        "flat_pool_size": n,
        "inputs": {
            "cand_indices": [Inline(list(span)) for span in case.spans],
            "pair_logits": Inline(case.logits),
            "cand_valid_mask": Inline(case.valid_mask),
            "query_mask": Inline(case.query_mask),
        },
        "relation_specs": [
            Inline(
                {
                    "relation_index": relation_index,
                    "relation_type": spec.relation_type,
                    "head_query_ids": list(spec.head_query_ids),
                    "tail_query_ids": list(spec.tail_query_ids),
                    "allow_self": spec.allow_self,
                }
            )
            for relation_index, spec in enumerate(case.specs)
        ],
        "proposals": proposals,
        "pair_batch": {"num_pairs": len(rows), "rows": rows},
    }
    if tie_stats is not None:
        record["tie_evidence"] = Inline(tie_stats)
    return record


def _tie_stats(proposals) -> dict:
    """Count exact float collisions, so "this case has ties" is measured."""
    arg_multiplicity = 0
    pair_multiplicity = 0
    tied_pair_rows = 0
    for proposal in proposals:
        for key in ("heads", "tails"):
            counts: Dict[float, int] = {}
            for slot in proposal[key]:
                record = slot.value
                if record["valid"]:
                    counts[record["prob"]] = counts.get(record["prob"], 0) + 1
            if counts:
                arg_multiplicity = max(arg_multiplicity, max(counts.values()))
        counts = {}
        for pair in proposal["pairs"]:
            record = pair.value
            score = record["head_prob"] * record["tail_prob"]
            counts[score] = counts.get(score, 0) + 1
        if counts:
            pair_multiplicity = max(pair_multiplicity, max(counts.values()))
            tied_pair_rows += sum(value for value in counts.values() if value > 1)
    return {
        "max_argument_prob_multiplicity": arg_multiplicity,
        "max_pair_score_multiplicity": pair_multiplicity,
        "pairs_sharing_a_score_with_another_pair": tied_pair_rows,
    }


# --------------------------------------------------------------------------
# --self-check: an independent re-derivation, and a can-it-fail probe
# --------------------------------------------------------------------------

_FLOOR = float(torch.finfo(torch.float32).min)


def _sigmoid_f32(x: float) -> float:
    """f32-native sigmoid — the arithmetic a Rust port does, not torch's."""
    value = torch.tensor([x], dtype=torch.float32)
    return float((1.0 / (1.0 + torch.exp(-value)))[0])


def _ulp_distance(a: float, b: float) -> int:
    pack = torch.tensor([a, b], dtype=torch.float32).view(torch.int32)
    return abs(int(pack[0]) - int(pack[1]))


def _plain_select(case_view: dict, member: set, requested: int, *, rekey: bool, flat_tie: bool):
    """The documented chain, reimplemented with no torch tensor ops at all."""
    n, cand_count = case_view["n"], case_view["cand_count"]
    spans, probs, eligible = case_view["spans"], case_view["probs"], case_view["eligible"]
    if rekey:
        secondary = sorted(range(n), key=lambda j: (spans[j][0], spans[j][1], j))
    else:
        secondary = list(range(n))
    valid = [eligible[j] and (j // cand_count) in member for j in range(n)]
    if flat_tie:
        order = sorted(
            range(n),
            key=lambda i: (
                -(probs[secondary[i]] if valid[secondary[i]] else _FLOOR),
                secondary[i],
            ),
        )
    else:
        # CPython's sorted is stable and reverse=True preserves the order of equals.
        order = sorted(
            range(n),
            key=lambda i: probs[secondary[i]] if valid[secondary[i]] else _FLOOR,
            reverse=True,
        )
    take = min(requested, n)
    ranked = [secondary[i] for i in order[:take]]
    sel_valid = [valid[j] for j in ranked]
    sel_prob = [probs[j] for j in ranked]
    if take < requested:
        pad = requested - take
        ranked += [0] * pad  # F.pad fills the FLAT INDEX with 0
        sel_valid += [False] * pad
        sel_prob += [0.0] * pad
    return ranked, sel_prob, sel_valid


def _case_view(record: dict) -> dict:
    queries, cand_count = record["queries"], record["cand_count"]
    n = queries * cand_count
    spans = [tuple(pair.value) for pair in record["inputs"]["cand_indices"]]
    logits = record["inputs"]["pair_logits"].value
    cand_valid = record["inputs"]["cand_valid_mask"].value
    query_mask = record["inputs"]["query_mask"].value
    probs = [_sigmoid_f32(x) for x in logits]
    threshold = float(load_settings()[0].argument_threshold)
    eligible = [
        cand_valid[j] and query_mask[j // cand_count] and probs[j] >= threshold
        for j in range(n)
    ]
    return {
        "n": n,
        "queries": queries,
        "cand_count": cand_count,
        "spans": spans,
        "probs": probs,
        "eligible": eligible,
    }


def _pairs_of(view, spec, heads, tails, settings, *, transposed: bool):
    rh, rt = settings.heads_per_relation, settings.tails_per_relation
    h_ranked, h_prob, h_valid = heads
    t_ranked, t_prob, t_valid = tails
    spans = view["spans"]
    score, valid = [], []
    for h in range(rh):
        for t in range(rt):
            same = spans[h_ranked[h]] == spans[t_ranked[t]]
            score.append(h_prob[h] * t_prob[t])
            valid.append(h_valid[h] and t_valid[t] and (spec["allow_self"] or not same))
    keep = sorted(
        range(rh * rt),
        key=lambda i: score[i] if valid[i] else _FLOOR,
        reverse=True,
    )[: min(settings.pair_cap, rh * rt)]
    out = []
    for k in keep:
        if not valid[k]:
            continue
        h, t = (k % rh, k // rh) if transposed else (k // rt, k % rt)
        out.append((h, t, spans[h_ranked[h]], spans[t_ranked[t]]))
    return out


def self_check(records, settings) -> int:
    """Re-derive every case from its own raw inputs, then probe for blind spots."""
    max_ulp, failures = 0, 0
    print("== independent re-derivation (no torch tensor ops, no gliner2) ==")
    for record in records:
        view = _case_view(record)
        cand_count = view["cand_count"]
        for proposal, spec_inline in zip(record["proposals"], record["relation_specs"]):
            spec = spec_inline.value
            heads = _plain_select(
                view,
                {q for q in spec["head_query_ids"] if 0 <= q < view["queries"]},
                settings.heads_per_relation,
                rekey=True,
                flat_tie=False,
            )
            tails = _plain_select(
                view,
                {q for q in spec["tail_query_ids"] if 0 <= q < view["queries"]},
                settings.tails_per_relation,
                rekey=True,
                flat_tie=False,
            )
            for label, (ranked, prob, valid) in (("heads", heads), ("tails", tails)):
                for slot, entry in enumerate(proposal[label]):
                    got = entry.value
                    j = ranked[slot]
                    want = {
                        "slot": slot,
                        "flat_index": j,
                        "query": j // cand_count,
                        "cand_slot": j % cand_count,
                        "start": view["spans"][j][0],
                        "end": view["spans"][j][1],
                        "valid": valid[slot],
                    }
                    for key, value in want.items():
                        if got[key] != value:
                            failures += 1
                            print(
                                f"  FAIL {record['name']}/{spec['relation_type']}/"
                                f"{label}[{slot}].{key}: {got[key]} != {value}"
                            )
                    max_ulp = max(max_ulp, _ulp_distance(got["prob"], prob[slot]))
            want_pairs = _pairs_of(view, spec, heads, tails, settings, transposed=False)
            got_pairs = [
                (
                    p.value["head_slot"],
                    p.value["tail_slot"],
                    (p.value["head_start"], p.value["head_end"]),
                    (p.value["tail_start"], p.value["tail_end"]),
                )
                for p in proposal["pairs"]
            ]
            if got_pairs != want_pairs:
                failures += 1
                print(
                    f"  FAIL {record['name']}/{spec['relation_type']}: pairs differ "
                    f"({len(got_pairs)} vs {len(want_pairs)})"
                )
    print(
        f"  integer fields, valid flags and slot order: "
        f"{'ALL MATCH' if failures == 0 else f'{failures} FAILURES'}"
    )
    print(
        f"  prob: max {max_ulp} ulp from an f32-native sigmoid "
        f"(torch's f32 sigmoid is not bit-identical to 1/(1+exp(-x)); "
        f"7b-2 must compare probs with a tolerance, not ==)"
    )

    print()
    print("== can-it-fail probe: three wrong ports vs this fixture ==")
    print(f"  {'case':<28} {'port':<14} {'slots differ':>12} {'pairs differ':>13}")
    blind = []
    for record in records:
        view = _case_view(record)
        for port in ("naive_no_rekey", "flat_index_tiebreak", "transposed_pair_index"):
            slot_diff = pair_diff = 0
            for proposal, spec_inline in zip(record["proposals"], record["relation_specs"]):
                spec = spec_inline.value
                kwargs = {
                    "rekey": port != "naive_no_rekey",
                    "flat_tie": port == "flat_index_tiebreak",
                }
                heads = _plain_select(
                    view,
                    {q for q in spec["head_query_ids"] if 0 <= q < view["queries"]},
                    settings.heads_per_relation,
                    **kwargs,
                )
                tails = _plain_select(
                    view,
                    {q for q in spec["tail_query_ids"] if 0 <= q < view["queries"]},
                    settings.tails_per_relation,
                    **kwargs,
                )
                for label, (ranked, _, _) in (("heads", heads), ("tails", tails)):
                    for slot, entry in enumerate(proposal[label]):
                        if entry.value["flat_index"] != ranked[slot]:
                            slot_diff += 1
                got = _pairs_of(
                    view,
                    spec,
                    heads,
                    tails,
                    settings,
                    transposed=port == "transposed_pair_index",
                )
                want = [
                    (
                        (p.value["head_start"], p.value["head_end"]),
                        (p.value["tail_start"], p.value["tail_end"]),
                    )
                    for p in proposal["pairs"]
                ]
                got_spans = [(g[2], g[3]) for g in got]
                pair_diff += abs(len(got_spans) - len(want)) + sum(
                    1 for a, b in zip(got_spans, want) if a != b
                )
            print(f"  {record['name']:<28} {port:<14} {slot_diff:>12} {pair_diff:>13}")
            if slot_diff == 0 and pair_diff == 0:
                blind.append((record["name"], port))
    print()
    print(f"  case/port combinations this fixture cannot distinguish: {len(blind)}")
    for entry in blind:
        print(f"    {entry[0]} / {entry[1]}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="re-derive every case independently and probe three wrong ports",
    )
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True, warn_only=True)
    settings, settings_meta = load_settings()
    generator = TypedRelationPairGenerator(settings)

    cases = build_cases()
    records = [run_case(case, settings, generator) for case in cases]

    document = {
        "schema": SCHEMA,
        "generator": "onnx_conversion_scripts/dump_relation_pairs.py",
        "reference": (
            "gliner2 2.0.0 "
            "gliner2.models.boundary.relations.TypedRelationPairGenerator"
            ".generate(..., compact=True)"
        ),
        "torch": torch.__version__,
        "purpose": (
            "7b-2 asserts index-for-index identity between this file and the "
            "Rust port's generate_pairs / generate_pairs_detailed. A difference "
            "in the proposal set is otherwise SILENT: fragment parity passes, "
            "entity parity stays 1.000, every test stays green, and the relation "
            "numbers never improve."
        ),
        "field_notes": [
            "num_words and boundary_states_padded_len are DIFFERENT numbers. "
            "boundary_states_padded_len is boundary_states.shape[1], the value "
            "SparseRelationScorer divides the word distance by "
            "(relations.py:374). Python pads to the batch max, Rust to a "
            "64/128/256/512 bucket. No field here is named `length` or `L`.",
            "TypedRelationPairGenerator does not read the padded length at all "
            "- proposal is pure index arithmetic. The field is carried for 7b-3, "
            "where feeding the wrong one shifts every score invisibly.",
            "inputs.pair_logits are raw f32 LOGITS; sigmoid is applied inside "
            "the generator. inputs.cand_indices are raw i64 half-open spans. "
            "Nothing here is decoded, thresholded at 0.5, or overlap-resolved.",
            "proposals[].heads / .tails are always exactly "
            "heads_per_relation / tails_per_relation slots long, because "
            "take = min(requested, Q*C) counts the WHOLE flat pool, valid and "
            "invalid alike. When take < requested, F.pad fills the FLAT INDEX "
            "with 0, so padded slots carry candidate 0's span - not (0, 0).",
            "proposals[].pairs mirror the Rust ProposedPair fields including "
            "head_slot / tail_slot (ranks into heads/tails), which have no "
            "counterpart in Python's RelationPairBatch. keep_position is the "
            "index into the 64 kept pairs BEFORE compaction dropped invalid "
            "survivors; truncation to pair_cap happens before that filter.",
            "pair_batch.rows are Python's compacted RelationPairBatch, flat and "
            "in order: relation-major, then keep order. head_key / tail_key are "
            "(str(query_id), start, end) because the dump passes "
            "query_layouts=[], so _query_type is never consulted.",
            "abstention_threshold = 0.5 governs decoding and is NOT used here. "
            "The only threshold in play is "
            "relation_argument_proposal_threshold = 0.2, applied with `>=`.",
        ],
        "settings": Inline(settings_meta),
        "case_index": [
            Inline(
                {
                    "name": record["name"],
                    "num_words": record["num_words"],
                    "boundary_states_padded_len": record[
                        "boundary_states_padded_len"
                    ],
                    "flat_pool_size": record["flat_pool_size"],
                    "relations": len(record["relation_specs"]),
                    "num_pairs": record["pair_batch"]["num_pairs"],
                }
            )
            for record in records
        ],
        "cases": records,
    }

    text = render(document)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"wrote {args.out}")
    print(f"sha256  {digest}")
    print(f"bytes   {len(text.encode('utf-8'))}")
    print()
    print(f"{'case':<30} {'num_words':>9} {'padded_len':>10} {'pool':>5} {'P':>4}")
    for record in records:
        print(
            f"{record['name']:<30} {record['num_words']:>9} "
            f"{record['boundary_states_padded_len']:>10} "
            f"{record['flat_pool_size']:>5} "
            f"{record['pair_batch']['num_pairs']:>4}"
        )
    if args.self_check:
        print()
        return 1 if self_check(records, settings) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
