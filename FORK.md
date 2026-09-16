# FORK.md — `gliner25-rs`, cognee prototype fork

This tree is a fork of **`gliner25-rs`**, kept for the Rust cognee GLiNER
graph-extraction prototype. It is not a redistribution target: it exists so
cognee can carry two small changes upstream has no reason to make.

| | |
|---|---|
| Upstream repo | <https://github.com/dariofinardi/gliner25-rs> |
| Upstream commit | `580c57b2a33e1f70a72204db82ef95a44707ec3a` ("feat: grouped Hub layout - one self-contained folder per precision", 2026-08-26) |
| Upstream crate version | `gliner25-rs` 0.5.6 |
| Upstream licence | Apache-2.0 |
| Fork branch | `cognee-prototype` |
| Vendored from | `~/dev/cognee/gliner-candidates/gliner25-rs/repo` (a working checkout of that commit) |

`target/` and upstream's `.git/` were not copied. Everything else was copied
byte for byte, including the two Apache-2.0 artifacts that must travel with the
code:

- `LICENSE` and `crates/gliner25-rs/LICENSE` — `sha256 bb28c48e3e078166e91cfc2b6db7ffebb8a0973b9e23b3df060561292d8d69ec`
- `NOTICE` and `crates/gliner25-rs/NOTICE` — `sha256 c942c11984900f9ab0237982be0537ecbadc471fb82d5a4111ed24dd3f2a2808`

Apache-2.0 §4(d) requires the `NOTICE` text to be propagated to any derivative
distribution. Any cognee crate that ships this code must carry that `NOTICE`
content too — the closed `cognee-gliner` crate does so in its own `NOTICE`.

To reproduce the full set of local changes:

```bash
git -C <an upstream clone> checkout 580c57b2a33e1f70a72204db82ef95a44707ec3a
diff -ru --exclude=.git --exclude=target <that clone> .
```

---

## Delta 1 — schema descriptions reach `BoundaryEngine`

**Files:** `crates/gliner25-rs/src/boundary.rs` (marked `COGNEE-EVAL PATCH`).
**Status:** written before this fork existed; it is part of the initial vendor
commit rather than a separate diff.

- `BoundaryParams` gains `descriptions: Vec<Vec<(String, String)>>` — per-group
  `label -> description` pairs, empty by default.
- `BoundaryEngine::extract_once` calls
  `SchemaTransformer::transform_with_descriptions(text, tasks, &params.descriptions)`
  where it used to call `transform(text, tasks)`.

**Why.** Python `gliner2`'s `SchemaTransformer._transform_schema` folds a
label's description into that group's single `prompt_str` token (schema index 2).
`transform_with_descriptions` already existed in the crate and implemented
exactly that, but nothing could reach it: `extract_once` hard-coded an empty
description list, so the method was dead code and `BoundaryEngine` could not
reproduce Python's prompt. cognee's ontology-driven schemas carry descriptions,
and dropping them changes the prompt and therefore the model's output. With an
empty `descriptions` vec the behaviour is byte-identical to upstream, so this is
additive.

---

## Delta 2 — `load-dynamic` and the execution providers are opt-in

**Files:** `Cargo.toml` (workspace `ort` entry),
`crates/gliner25-rs/Cargo.toml` (`[features]`),
`crates/gliner25-rs/src/runtime.rs` (marked `COGNEE FORK DELTA`).

Upstream's workspace `ort` entry was:

```toml
ort = { version = "2.0.0-rc.13", default-features = false, features = [
    "std", "ndarray", "load-dynamic",
    "cuda", "rocm", "coreml", "openvino", "directml", "tensorrt", "xnnpack", "qnn", "half",
] }
```

It is now:

```toml
ort = { version = "2.0.0-rc.13", default-features = false, features = [
    "std", "ndarray", "half", "download-binaries", "tls-rustls",
] }
```

and `gliner25-rs` grew nine features, **all off by default**, forwarding to the
`ort` features that were removed:

```toml
load-dynamic = ["ort/load-dynamic"]
cuda     = ["ort/cuda"]
rocm     = ["ort/rocm"]
coreml   = ["ort/coreml"]
openvino = ["ort/openvino"]
directml = ["ort/directml"]
xnnpack  = ["ort/xnnpack"]
qnn      = ["ort/qnn"]
tensorrt = ["ort/tensorrt", "cuda"]
```

### Why `load-dynamic` had to move

Cargo features are additive across a whole build graph, not per crate. A
library that turns one on turns it on for every other crate in the same
artifact, and there is no way for a downstream consumer to turn it back off.

`ort/load-dynamic` implies `ort-sys/disable-linking`, and `ort-sys`'s build
script returns immediately when that is set — nothing is downloaded and nothing
is linked. ONNX Runtime is then expected to be found as a shared library at run
time via `ORT_DYLIB_PATH`.

cognee has a second `ort` consumer: `cognee-embedding`'s `onnx` backend, whose
`ort::init().commit()` sits behind an `.expect(...)`, ahead of any error
handling. Six closed crates reach it, including the only real closed binary
(`cognee-cli-cloud`, via OSS `crates/lib`'s defaults). Had the fork kept
`load-dynamic` on by default, every artifact that linked both — the Python
wheel, the Neon `.node`, the C staticlib, the CLI, the HTTP server image —
would have shipped without an ONNX Runtime and **panicked at startup**. A build
flag is the wrong thing to decide on a library's behalf; it is now the
top-level binary's choice.

### Why `download-binaries` + `tls-rustls` were added

These are not a preference; they are what `load-dynamic` was silently
providing. With `load-dynamic` gone and nothing in its place, `ort-sys` neither
downloads nor links ONNX Runtime and every binary fails with
`undefined symbol: OrtGetApiBase`. `download-binaries` restores a linkable
runtime, and it is also the mode `cognee-embedding`'s `onnx` backend already
builds in (OSS takes `ort`'s default features), so the two consumers now agree
on link mode by construction instead of conflicting.

`tls-rustls` rather than `ort`'s default `tls-native`, for the reason upstream
already gives for the `hf-hub` entry directly below it: `native-tls` drags in
`openssl`, a C library and its CVE stream, for no gain here.

### Why `runtime.rs` needed changes

`ort` gates the provider types themselves — `ort::ep::CUDA`, `ep::ROCm`,
`ep::CoreML`, `ep::DirectML`, `ep::OpenVINO`, `ep::TensorRT`, `ep::XNNPACK` all
live behind `#[cfg(feature = ...)]` in `ort/src/ep/mod.rs`. Upstream could name
them unconditionally because it always enabled every one. Three functions had
to become feature-aware:

1. **`execution_providers()`** — one `#[cfg]` per match arm. With every provider
   feature on, the arms are upstream's, unchanged. With one off, that
   `GLINER2_DEVICE` value falls through to the catch-all and runs on CPU, which
   is the same outcome as asking for a provider whose shared library is absent —
   a case upstream's own doc comment already describes. `"auto"` gets an extra
   `#[cfg(not(feature = "cuda"))]` arm returning no providers, because `auto` is
   what an *unset* `GLINER2_DEVICE` resolves to and it must not print a warning
   on every default run.

2. **`provider_has_device_memory()`** — this decides whether
   `ExecutionMode::Auto` resolves to `IoBinding` (`chain.rs:56`). Upstream could
   answer from the device string alone; here the answer must also depend on
   whether that provider was compiled in, or a CPU-only build would pick
   `IoBinding` for a session that has no device memory.

3. **`allocation_device()`** — same reasoning, so it stays consistent with what
   `execution_providers()` actually registered.

All three reduce to upstream's behaviour exactly when every provider feature is
enabled.

`tensorrt` forwards to `cuda` deliberately: `GLINER2_DEVICE=tensorrt` registers
TensorRT *and* CUDA behind it, because TensorRT falls back to CUDA. Enabling
`tensorrt` alone would leave that arm unable to build its own fallback.

`qnn` has no `GLINER2_DEVICE` arm — upstream had none either. The feature exists
only so a caller can still compile the bindings in.

### Linking caveat when an execution provider is enabled

Inherited from `ort`, not introduced here. With `download-binaries`, `ort-sys`
must find a prebuilt distribution matching the *exact* requested provider set
for the target triple. On `x86_64-unknown-linux-gnu`, `cuda` and `tensorrt`
resolve; `rocm`, `coreml`, `openvino`, `directml`, `xnnpack` and `qnn` have no
dist and fail at link time with `undefined symbol: OrtGetApiBase`. Upstream
never met this because `load-dynamic` skipped dist resolution entirely.

So: enable a provider together with `load-dynamic`, or point `ORT_LIB_LOCATION`
at an ONNX Runtime build that has it.

```bash
# default — CPU, prebuilt runtime downloaded and linked
cargo test -p gliner25-rs

# upstream's exact build
cargo test -p gliner25-rs --features \
  load-dynamic,cuda,rocm,coreml,openvino,directml,tensorrt,xnnpack,qnn
```

---

## Delta 3 — the cognee parity harness is a test, not an example

**Files:** `crates/gliner25-rs/tests/cognee_contract.rs` (new),
`crates/gliner25-rs/tests/fixtures/{cognee_parity_scenarios.json,py_reference.json,README.md}`
(new), `crates/gliner25-rs/examples/cognee_parity.rs` (removed).

`examples/cognee_parity.rs` was a cognee addition (`COGNEE-EVAL`), never
upstream code: an ad-hoc runner that built the Python-`gliner2` prompt layout,
extracted over four scenarios and wrote the candidate-evaluation contract JSON
for a separate Python scorer to grade by hand. Everything that rests on this
backend rests on one measurement it produced, and an example that nobody runs
cannot defend a measurement.

It is now `tests/cognee_contract.rs`: the same prompt construction — group 0
`SchemaTask::Entities` with the descriptions folded into
`BoundaryParams.descriptions[0]`, one `SchemaTask::Relations("<name>: <desc>",
["head","tail"])` per relation, insertion-ordered label parsing — plus a Rust
port of `compare_rust_python.py`'s set arithmetic, so `cargo test` scores itself
and needs no `python3` and no path outside this tree. The two `_shared/`
fixtures are copied in byte for byte and pulled in with `include_str!`;
`tests/fixtures/README.md` carries their provenance and sha256s.

Removing the example rather than keeping both avoids ~200 lines of
prompt-construction logic duplicated across two targets that would have to stay
bit-identical. Its command-line switches survive as env overrides on the test
(`GLINER25_MODELS`, `COGNEE_PARITY_SCENARIOS`, `COGNEE_PARITY_OUT`,
`COGNEE_PARITY_NO_DESCRIPTIONS`); `--model-label` was only ever cosmetic and is
dropped.

The test needs no features, no `--release` and no `ORT_DYLIB_PATH`: the output
is byte-identical under the fork's default static `download-binaries` link mode,
under `--features load-dynamic`, and at either precision. With no export on disk
it skips with a `⚠️` line, so it is inert in a checkout that has no model.

```bash
ln -s <export> models/gliner2.5-base-v1-onnx   # models/ is gitignored
cargo test -p gliner25-rs --test cognee_contract -- --nocapture
```

---

## Delta 4 — `BoundaryOutput` carries relations through chunking

**Files:** `crates/gliner25-rs/src/boundary.rs`,
`crates/gliner25-rs/src/chunker.rs`, `crates/gliner25-rs/src/families.rs`,
`crates/gliner25-rs/src/lib.rs`.

Upstream decodes relations *after* merging the windows, from merged mention
text alone (`boundary::pair_relations`, a type-compatible cartesian product).
That is wrong for a document: it manufactures edges spanning hundreds of words.
`gliner2` instead scores relations **inside each window** and merges the decoded
edges (`inference/chunking.py::_merge_relation_maps`), and it has no choice —
the model's relation scorer indexes both endpoints against one padded length, so
a pair whose ends live in different windows has no shared frame at all.

This delta is the plumbing for that ordering; the scorer itself is a later
change. Added:

- `boundary::RelationEndpoint` and `boundary::RelationEdge` — self-contained
  span + surface text for each end, plus the relation's `prompt_str` and a
  score.
- `BoundaryOutput.relations: Vec<RelationEdge>`. The struct already derives
  `Default`, and `boundary.rs`'s own construction site already spread it, so
  only the six literal sites needed touching: five in `chunker.rs` (one in
  `merge`, four in its tests) and one in `families.rs` — where the merge body
  was split out of `run_families` into a model-free `merge_family_outputs`, so
  that the fold is reachable from a test without a model behind it.
- `chunker::remap` shifts both endpoints of every edge by `chunk.byte_start` /
  `chunk.word_start` and re-slices their surface text, mirroring what it
  already does for mentions.
- `chunker::merge_relations` + `chunker::RelationKeyMode`, a port of the
  non-span branch of `_dedupe_items` (`chunking.py`): a canonical key with the
  score stripped, first-seen insertion order, and replacement only on a
  **strictly** greater score. `RelationKeyMode::SpanAndText` is the default
  because the checked-in Python reference (`tests/fixtures/py_reference.json`)
  was produced with `include_spans=True`; `TextOnly` reproduces the bare
  `(head, tail)` shape for a caller that asks for neither spans nor confidence.
  The survivor rule is mode-dependent, because Python's is: a tuple is neither a
  `dict` nor a `list`, so `_representative_confidence` scores every `TextOnly`
  item `0.0` and the incumbent is **never** replaced, whatever the scores.
- Families run every pass over the *same* text, so their outputs share one
  coordinate frame and `merge_family_outputs` folds their relations with
  `merge_relations` too, rather than discarding them.

**One behaviour worth naming, because it looks like a bug.** `merge`'s seam
pass deletes mentions, so it can delete a mention that a relation from another
window points at. Those relations are kept anyway — `_merge_relation_maps`
never consults the entity set either, and an edge carries its own spans and
text. A referential-integrity filter here would be a divergence invented by the
fork that no parity test could catch. `merge`'s doc comment says so, and
`chunker::tests::relation_outlives_its_endpoint_mention` pins it.

Nothing produces relations yet, so `BoundaryOutput.relations` is empty on every
path today and no existing behaviour changes. `pair_relations` is untouched.

---

## Deliberately not changed

- **Formatting.** Upstream's tree is not `cargo fmt`-clean under default
  rustfmt settings (16 files differ, and there is no `rustfmt.toml`). Running
  `cargo fmt` would bury the deltas in thousands of unrelated lines. The fork
  leaves formatting exactly as upstream has it; the *same* 16 files differ here
  as upstream, and no more. (This note said 17 until Delta 3: the seventeenth
  was the cognee-authored `examples/cognee_parity.rs`, which that delta
  removed. Files the fork *adds* are rustfmt-clean —
  `tests/cognee_contract.rs` is.)
- **Upstream's clippy findings.** `cargo clippy -p gliner25-rs --all-targets
  -- -D warnings` fails on two lints in code the fork did not write, both of
  them lints that post-date the upstream commit: `clippy::collapsible_if` at
  `src/boundary.rs:292` and `clippy::manual_is_multiple_of` at
  `examples/long_document.rs:25` (Rust 1.93.0). Fixing them would touch
  upstream lines for no cognee reason, so they stay; a clean run is
  `-A clippy::collapsible_if -A clippy::manual_is_multiple_of`, under which the
  whole crate, `tests/cognee_contract.rs` included, is warning-free.
- **`ort`'s `api-NN` features.** Upstream set none (it passes
  `default-features = false`), so neither does the fork. Inside cognee, OSS's
  own `ort` entry uses default features and contributes `api-27` through
  unification.
