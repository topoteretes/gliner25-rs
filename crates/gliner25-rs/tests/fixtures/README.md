# `cognee_contract.rs` fixtures

Copied byte-for-byte (`cp -p`) on 2026-09-16 from the read-only candidate
evaluation directory `~/dev/cognee/gliner-candidates/_shared/`. They are
committed here so `cargo test -p gliner25-rs` depends on nothing outside this
fork; `tests/cognee_contract.rs` pulls both in with `include_str!`, so a
missing or renamed fixture is a compile error rather than a silent skip.

| file | source | sha256 |
|---|---|---|
| `cognee_parity_scenarios.json` | `_shared/cognee_parity_scenarios.json` | `fdd81d6a31a033fd2ce34c03e566b06e19438dae078106d7880414070c3a853e` |
| `py_reference.json` | `_shared/py_reference.json` | `9b0e95e90c5e4e4dfebf259b7ae8010977a51b7d81b426cedad407cbb1e1a6c4` |

Verify with `sha256sum *.json`. Do not reformat either file — the hashes above
are over the exact bytes, and the scenario text feeds the model directly.

## `cognee_parity_scenarios.json`

The input spec: model id, `threshold` 0.5, `chunk_size` 384, `chunk_overlap`
64, `overlap_policy` `longest`, six `entity_types` and five `relation_types`
(each `name -> description`), and four `scenarios` — `short` (14 words),
`medium` (69), `long` (348), `very_long` (810). **Key order is significant**:
the label order becomes the prompt order and therefore changes model output,
which is why the test deserializes these objects into an insertion-ordered
`Vec` rather than `serde_json`'s `Map`.

## `py_reference.json`

Output of the **Python `gliner2` 2.0.0** pipeline (the version cognee pins as
`gliner2[local]>=2.0.0,<3`) over those same scenarios, in the shape
`_shared/CONTRACT.md` specifies.

It is ground truth for **agreement, not for facts.** The entity sets are
set-identical to this Rust engine's and are asserted exactly. The relation
sets are not a quality reference: on the `long` scenario the two
implementations agree on 18 edges of which 7 are factually false (e.g.
`works_for::Tim Cook|Microsoft`, `produces::Apple|Azure`). Relation scores
against this file measure *stability*, and the factual bar lives elsewhere —
MASTER_PLAN Gate D, hand-annotated F1.
