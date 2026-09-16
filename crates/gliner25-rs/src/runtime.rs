// Copyright 2026 Dario Finardi. Published by Jugaad s.r.l. — Apache-2.0

//! Shared helpers layered on top of `ort` 2.0.0-rc.13.
//!
//! ## What changed since rc.9
//!
//! | rc.9                                      | rc.13                                            |
//! |-------------------------------------------|--------------------------------------------------|
//! | `ort::init().commit()?`                   | `commit()` returns `bool`, not `Result`          |
//! | `Session::run(&self, …)`                  | `run(&mut self, …)`; the outputs borrow the session |
//! | `builder.with_x()?` inside `anyhow`       | `BuilderResult` carries a non-`Send` `Error<SessionBuilder>`: convert with `ort::Error::<()>::from` |
//! | `commit_from_file(self, …)`               | `commit_from_file(&mut self, …)`                 |
//! | `try_extract_tensor() -> (Vec<i64>, &[T])`| `-> (&Shape, &[T])`; the ndarray view is `try_extract_array` |
//! | public fields on `Outlet`                 | `name()` / `dtype()` accessors                   |
//! | `ndarray 0.16`                            | `ndarray 0.17`                                   |
//! | `download-binaries` in `ort-sys` defaults | only in `ort`'s; with `load-dynamic` set `ORT_DYLIB_PATH` |
//!
//! Because `run` takes `&mut self`, an engine owning several sessions runs them
//! in sequence and its extraction methods take `&mut self`.

use anyhow::{Context, Result, anyhow};
use half::f16;
use ort::ep::ExecutionProviderDispatch;
// COGNEE FORK DELTA: the `ort::ep::{CUDA, ROCm, …}` types are themselves gated
// behind `ort`'s per-provider features, which are now opt-in (see FORK.md), so
// the module import only exists when at least one of them is compiled in.
#[cfg(any(
    feature = "cuda",
    feature = "tensorrt",
    feature = "rocm",
    feature = "coreml",
    feature = "directml",
    feature = "openvino",
    feature = "xnnpack",
))]
use ort::ep;
use ort::session::Session;
use ort::session::builder::SessionBuilder;
use ort::value::{DynValue, Tensor};
use std::path::{Path, PathBuf};

/// Precision variant of the fragments on disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// `*_fp32.onnx` — FP32 weights and I/O. Universal fallback, OpenVINO, CPU.
    Fp32,
    /// `*_fp16.onnx` — FP16 weights, FP32 I/O (`keep_io_types=True`). Required by CoreML.
    Fp16,
    /// `*_fp16_iobinding.onnx` — FP16 weights and I/O. CUDA / ROCm / QNN with IOBinding.
    Fp16IoBinding,
}

impl Precision {
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Fp32 => "_fp32",
            Self::Fp16 => "_fp16",
            Self::Fp16IoBinding => "_fp16_iobinding",
        }
    }

    /// Element type of the tensors flowing in and out of the fragments.
    pub fn io_dtype(self) -> IoDType {
        match self {
            Self::Fp32 | Self::Fp16 => IoDType::F32,
            Self::Fp16IoBinding => IoDType::F16,
        }
    }

    /// Subfolder a legacy export keeps this variant in.
    ///
    /// Variants to try, in order, when the preferred one is not published.
    ///
    /// Not every export ships all three variants. Asking the Hub for one that
    /// does not exist should degrade to one that does, not fail the load.
    pub fn fallback_chain(self) -> &'static [Precision] {
        match self {
            Self::Fp16IoBinding => &[Self::Fp16IoBinding, Self::Fp16, Self::Fp32],
            Self::Fp16 => &[Self::Fp16, Self::Fp32],
            Self::Fp32 => &[Self::Fp32],
        }
    }

    /// Subfolders this variant may live in, in search order.
    ///
    /// An export can be flat or grouped by precision, and the group names
    /// differ by lineage: the GLiNER2.5 exports use `fp32_25/` and `fp16_25/`,
    /// the GLiNER2 ones `fp32_v2/` and `fp16_v2/`. Both are accepted, so a
    /// directory copied from either family loads without being renamed. The
    /// FP16 folder holds `_fp16` and `_fp16_iobinding` together.
    pub fn subdirs(self) -> &'static [&'static str] {
        match self {
            Self::Fp32 => &["fp32_25", "fp32_v2"],
            Self::Fp16 | Self::Fp16IoBinding => &["fp16_25", "fp16_v2"],
        }
    }

    /// The subfolder this crate's own exports are written into.
    pub fn legacy_subdir(self) -> &'static str {
        self.subdirs()[0]
    }

    /// Picks the best variant available for the current platform.
    ///
    /// On Linux and Windows the `_fp16_iobinding` variants maximise CUDA/ROCm;
    /// on macOS CoreML demands FP32 I/O, so `_fp16` is used instead.
    /// `GLINER2_PRECISION=fp32|fp16|fp16_iobinding` overrides the choice.
    pub fn autodetect(dir: &Path, stem_probe: &str) -> Self {
        if let Ok(forced) = std::env::var("GLINER2_PRECISION") {
            match forced.as_str() {
                "fp32" => return Self::Fp32,
                "fp16" => return Self::Fp16,
                "fp16_iobinding" => return Self::Fp16IoBinding,
                other => eprintln!("GLINER2_PRECISION={other} not recognised, ignoring"),
            }
        }
        let exists = |p: Precision| resolve_fragment(dir, stem_probe, p).is_some();
        let prefer_iobinding = !cfg!(target_os = "macos") && !cfg!(target_os = "ios");
        if prefer_iobinding && exists(Self::Fp16IoBinding) {
            Self::Fp16IoBinding
        } else if exists(Self::Fp16) {
            Self::Fp16
        } else {
            Self::Fp32
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoDType {
    F32,
    F16,
}

/// Resolves one fragment, accepting both directory layouts.
///
/// Flat, as produced by `export_span_v3.py`:
///
/// ```text
/// models/encoder_fp16_iobinding.onnx
/// ```
///
/// Legacy, as published on the Hub by the earlier exporter:
///
/// ```text
/// models/fp16_v2/encoder_fp16_iobinding.onnx
/// ```
pub fn resolve_fragment(dir: &Path, stem: &str, precision: Precision) -> Option<PathBuf> {
    let file = format!("{stem}{}.onnx", precision.suffix());
    let flat = dir.join(&file);
    if flat.exists() {
        return Some(flat);
    }
    for sub in precision.subdirs() {
        let nested = dir.join(sub).join(&file);
        if nested.exists() {
            return Some(nested);
        }
    }
    None
}

/// Locates the tokenizer, which the legacy layout keeps inside each variant
/// subfolder rather than at the root.
pub fn resolve_tokenizer(dir: &Path, precision: Precision) -> Option<PathBuf> {
    resolve_aux(dir, "tokenizer.json", precision)
}

/// Locates a precision-independent file that sits beside the fragments.
///
/// `boundary_manifest.json` is the same whatever precision is loaded, but an
/// export organised into `fp32_v2/` and `fp16_v2/` keeps a copy in each rather
/// than one at the root, so both places have to be tried.
pub fn resolve_aux(dir: &Path, name: &str, precision: Precision) -> Option<PathBuf> {
    let mut candidates = vec![dir.join(name)];
    // The requested precision's folders first, then the other family's, so a
    // half-populated export still resolves rather than failing outright.
    for sub in precision.subdirs() {
        candidates.push(dir.join(sub).join(name));
    }
    for sub in ["fp32_25", "fp16_25", "fp32_v2", "fp16_v2"] {
        candidates.push(dir.join(sub).join(name));
    }
    candidates.into_iter().find(|c| c.exists())
}

/// Which execution providers to register, from `GLINER2_DEVICE`.
///
/// | value | meaning |
/// |---|---|
/// | unset or `auto` | CUDA first, then CPU — ONNX Runtime falls back on its own |
/// | `cpu` | CPU only, no provider registered |
/// | `cuda` / `cuda:N` | NVIDIA CUDA on device 0 or N, CPU behind it |
/// | `tensorrt` | TensorRT, with CUDA and CPU behind it |
/// | `rocm`, `coreml`, `directml`, `openvino`, `xnnpack` | the matching provider, CPU behind it |
///
/// Registration is a request, not a guarantee: if the provider's shared library
/// is missing — the plain `onnxruntime` wheel ships none of them — ONNX Runtime
/// logs and runs on CPU. Nothing here reports which one actually served the
/// session, so treat a device flag as intent and confirm with a benchmark.
pub fn execution_providers() -> Vec<ExecutionProviderDispatch> {
    let requested = std::env::var("GLINER2_DEVICE").unwrap_or_else(|_| "auto".into());
    let (name, device_id) = match requested.split_once(':') {
        Some((n, id)) => (n.to_string(), id.parse::<i32>().unwrap_or(0)),
        None => (requested.clone(), 0),
    };

    // COGNEE FORK DELTA: one `#[cfg]` per arm, because each `ep::*` type only
    // exists when its `ort` feature is on and those are opt-in here. With every
    // provider feature enabled this matches upstream arm for arm. With one off,
    // its device name falls through to the catch-all and runs on CPU — the same
    // outcome as asking for a provider whose shared library is missing.
    #[cfg(not(any(feature = "cuda", feature = "tensorrt")))]
    let _ = device_id;
    match name.trim().to_lowercase().as_str() {
        "cpu" => Vec::new(),
        #[cfg(feature = "cuda")]
        "cuda" | "auto" => vec![ep::CUDA::default().with_device_id(device_id).build()],
        // `auto` must still be a recognised value when CUDA is not compiled in:
        // it is what an unset `GLINER2_DEVICE` resolves to, so warning about it
        // would print on every default run.
        #[cfg(not(feature = "cuda"))]
        "auto" => Vec::new(),
        #[cfg(feature = "tensorrt")]
        "tensorrt" => vec![
            ep::TensorRT::default().build(),
            ep::CUDA::default().with_device_id(device_id).build(),
        ],
        #[cfg(feature = "rocm")]
        "rocm" => vec![ep::ROCm::default().build()],
        #[cfg(feature = "coreml")]
        "coreml" => vec![ep::CoreML::default().build()],
        #[cfg(feature = "directml")]
        "directml" => vec![ep::DirectML::default().build()],
        #[cfg(feature = "openvino")]
        "openvino" => vec![ep::OpenVINO::default().build()],
        #[cfg(feature = "xnnpack")]
        "xnnpack" => vec![ep::XNNPACK::default().build()],
        other => {
            eprintln!(
                "GLINER2_DEVICE={other} not available in this build \
                 (unrecognised, or its gliner25-rs feature is off), running on CPU"
            );
            Vec::new()
        }
    }
}

/// Parses `GLINER2_DEVICE` once, into `(provider, device id)`.
fn requested_device() -> (String, i32) {
    let requested = std::env::var("GLINER2_DEVICE").unwrap_or_else(|_| "auto".into());
    match requested.split_once(':') {
        Some((n, id)) => (n.trim().to_lowercase(), id.parse::<i32>().unwrap_or(0)),
        None => (requested.trim().to_lowercase(), 0),
    }
}

/// The device ordinal `GLINER2_DEVICE=cuda:1` asked for.
pub fn device_id() -> i32 {
    requested_device().1
}

/// Whether the configured provider owns memory that `IoBinding` can bind to.
///
/// This decides what `ExecutionMode::Auto` resolves to. CPU and the graph
/// optimisers that run on it have nothing to bind: their "device memory" is
/// host memory, so binding would add bookkeeping and save no copy.
pub fn provider_has_device_memory() -> bool {
    // COGNEE FORK DELTA: a device name only implies device memory if that
    // provider was actually compiled in. Upstream could answer from the string
    // alone because every provider was always present; here a build with the
    // features off would otherwise pick `IoBinding` for a session that is
    // running on CPU. With all features on this is upstream's `matches!`.
    match requested_device().0.as_str() {
        "cuda" | "auto" => cfg!(feature = "cuda"),
        "tensorrt" => cfg!(feature = "tensorrt"),
        "rocm" => cfg!(feature = "rocm"),
        "directml" => cfg!(feature = "directml"),
        _ => false,
    }
}

/// The ORT allocation device matching the configured provider.
pub fn allocation_device() -> ort::memory::AllocationDevice {
    use ort::memory::AllocationDevice;
    // COGNEE FORK DELTA: same reasoning as `provider_has_device_memory` — the
    // guards keep this consistent with what `execution_providers` registered.
    // With every provider feature on, the guards are all true and this is
    // upstream's mapping unchanged.
    match requested_device().0.as_str() {
        "rocm" if cfg!(feature = "rocm") => AllocationDevice::HIP,
        "directml" if cfg!(feature = "directml") => AllocationDevice::DIRECTML,
        // CUDA also backs the TensorRT provider, which falls back to it.
        "cuda" | "auto" if cfg!(feature = "cuda") => AllocationDevice::CUDA,
        "tensorrt" if cfg!(feature = "tensorrt") => AllocationDevice::CUDA,
        _ => AllocationDevice::CPU,
    }
}

/// Builds a session with the common options applied.
///
/// `ort::init()` is the caller's responsibility and is not repeated here:
/// in rc.13 `commit()` returns `bool` and a second call would simply be
/// ignored.
pub fn build_session(path: &Path, intra_threads: usize) -> Result<Session> {
    if !path.exists() {
        return Err(anyhow!("missing ONNX fragment: {}", path.display()));
    }
    let mut builder: SessionBuilder = Session::builder()?
        .with_intra_threads(intra_threads)
        .map_err(ort::Error::<()>::from)?;

    let providers = execution_providers();
    if !providers.is_empty() {
        builder = builder
            .with_execution_providers(providers)
            .map_err(ort::Error::<()>::from)?;
    }

    builder
        .commit_from_file(path)
        .with_context(|| format!("while loading {}", path.display()))
}

/// Creates a float tensor in whichever precision the fragment expects.
pub fn float_tensor(dtype: IoDType, shape: Vec<i64>, data: Vec<f32>) -> Result<DynValue> {
    Ok(match dtype {
        IoDType::F32 => Tensor::from_array((shape, data))?.into_dyn(),
        IoDType::F16 => {
            let half: Vec<f16> = data.into_iter().map(f16::from_f32).collect();
            Tensor::from_array((shape, half))?.into_dyn()
        }
    })
}

pub fn i64_tensor(shape: Vec<i64>, data: Vec<i64>) -> Result<DynValue> {
    Ok(Tensor::from_array((shape, data))?.into_dyn())
}

/// Extracts a float tensor as `(shape, data as f32)`, whatever precision the
/// fragment produced it in.
pub fn take_float(value: &DynValue, dtype: IoDType) -> Result<(Vec<i64>, Vec<f32>)> {
    Ok(match dtype {
        IoDType::F32 => {
            let (shape, data) = value.try_extract_tensor::<f32>()?;
            (shape.iter().copied().collect(), data.to_vec())
        }
        IoDType::F16 => {
            let (shape, data) = value.try_extract_tensor::<f16>()?;
            (shape.iter().copied().collect(), data.iter().map(|v| v.to_f32()).collect())
        }
    })
}

pub fn take_i64(value: &DynValue) -> Result<(Vec<i64>, Vec<i64>)> {
    let (shape, data) = value.try_extract_tensor::<i64>()?;
    Ok((shape.iter().copied().collect(), data.to_vec()))
}

pub fn take_bool(value: &DynValue) -> Result<(Vec<i64>, Vec<bool>)> {
    let (shape, data) = value.try_extract_tensor::<bool>()?;
    Ok((shape.iter().copied().collect(), data.to_vec()))
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

pub fn softmax(xs: &[f32]) -> Vec<f32> {
    let max = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = xs.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}
