# Relation-proposal fixtures (stage 7, sub-stage 7a-3)

| file | generator | sha256 |
|---|---|---|
| `relation_pairs.json` | `onnx_conversion_scripts/dump_relation_pairs.py` | `da0d74879fc2011c068b3305514aa8282a776109925a66592742963d019f0d51` |

Verify with `sha256sum relation_pairs.json`. **Do not reformat the file** — the
hash is over the exact bytes, and 7b-2 pins them so later stages can prove they
are testing against the same fixture. Regenerate with:

```bash
/home/dmytro/dev/cognee/gliner2-candle/.venv/bin/python \
  onnx_conversion_scripts/dump_relation_pairs.py [--self-check]
```

Needs only `torch` + `gliner2` — no ONNX, no onnxruntime, no checkpoint, no
network. Regenerating is byte-identical (verified across repeated runs and a
changed `PYTHONHASHSEED`).

## What it is

Seven cases of Python `TypedRelationPairGenerator.generate(..., compact=True)`
output (gliner2 2.0.0), each carrying:

* `inputs` — raw i64 candidate spans and raw f32 `pair_logits` (logits, not
  probabilities; nothing is decoded, thresholded at 0.5 or overlap-resolved),
  so the Rust side can be driven with no engine in the loop;
* `proposals[]` — the intermediate top-32 `heads` and top-32 `tails` argument
  lists, one record per slot with `flat_index` / `query` / `cand_slot` /
  `start` / `end` / `prob` / `valid`, plus per-pair `head_slot` / `tail_slot`.
  This is exactly the shape `generate_pairs_detailed` exposes;
* `pair_batch` — the compacted `RelationPairBatch`, every field, in order.

## Two fields that are not the same number

`num_words` and `boundary_states_padded_len` are separate, deliberately
unambiguous names. The second is `boundary_states.shape[1]`, the padded length
`SparseRelationScorer` divides the word distance by (`relations.py:374`, no
mask). **Python pads to the batch max; Rust pads to a 64/128/256/512 bucket.**
Nothing in this file is called `length`, `len` or `L`.

The pair *generator* never reads it — proposal is pure index arithmetic. The
field is here for 7b-3, where feeding the wrong one shifts every score while
every existing test stays green.

## Cases

| case | num_words | padded len | Q·C | P | what it is for |
|---|---:|---:|---:|---:|---|
| `small_pool_padded_l64` | 40 | 64 | 18 | 16 | Q·C < 32 → `F.pad` branch: padded slots carry **candidate 0's span**, not `(0,0)` |
| `wide_pool_l512` | 300 | 512 | 64 | 192 | `take == 32`, no padding; 1024 pairs truncated to `pair_cap = 64`; an out-of-range query id that must be dropped, not clamped |
| `exact_len_l128` | 128 | 128 | 40 | 64 | the contrast case: padded len **equals** `num_words` |
| `tied_scores_total_l64` | 40 | 64 | 50 | 128 | every prob exactly `0.5`, every pair score exactly `0.25` — selection is decided **entirely** by the tie-break chain |
| `tied_scores_banded_l128` | 64 | 128 | 48 | 128 | three exact probability bands; ties exact within a band, bands order the list |
| `empty_below_threshold_l512` | 200 | 512 | 24 | **0** | nothing clears the 0.2 argument threshold → empty argument pool |
| `empty_self_span_only_l64` | 12 | 64 | 4 | **0** | pool is *not* empty, but the only pair is a span with itself and `same_span` kills it |

Two distinct P=0 routes, because 7a-2 found the exported ONNX graph **cannot
run** at `P = 0` (`num_pairs >= 1` baked in at tracing; ORT fails in
`node_add_163`) while Python early-returns at `relations.py:335-336`.

## For 7b-2

Include it from `crates/gliner25-rs/tests/`:

```rust
const RELATION_PAIRS: &str =
    include_str!("../../../onnx_conversion_scripts/fixtures/relation_pairs.json");
```

or copy it into `crates/gliner25-rs/tests/fixtures/` and record the same hash
there. It was not placed under `crates/` by 7a-3 because that lane was live.

**Compare `prob` with a tolerance, not `==`.** Torch's f32 sigmoid is not
bit-identical to a Rust `1.0 / (1.0 + (-x).exp())`: measured ≤ 2 ulp. Span
indices, `flat_index`, `query`, `cand_slot`, `valid` and slot **order** are
exact and must be asserted exactly.

**The tied cases are the ones that matter.** `--self-check` probes three wrong
ports against this fixture and prints what each case can distinguish. On the
non-tied cases a naive port (no `(start, end, flat)` re-key) changes the
heads/tails lists but produces the **same final pairs** — so a test that only
compares `generate_pairs` passes. Only `tied_scores_total_l64` and
`tied_scores_banded_l128` make the *pairs* themselves diverge.
