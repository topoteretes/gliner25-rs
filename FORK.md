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

## Deliberately not changed

- **Formatting.** Upstream's tree is not `cargo fmt`-clean under default
  rustfmt settings (17 files differ, and there is no `rustfmt.toml`). Running
  `cargo fmt` would bury both deltas in thousands of unrelated lines. The fork
  leaves formatting exactly as upstream has it; the *same* 17 files differ here
  as upstream, and no more.
- **`ort`'s `api-NN` features.** Upstream set none (it passes
  `default-features = false`), so neither does the fork. Inside cognee, OSS's
  own `ort` entry uses default features and contributes `api-27` through
  unification.
