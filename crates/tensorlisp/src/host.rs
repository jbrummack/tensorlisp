//! Host values for pre- and postprocessing: tokenizers, images, audio and
//! tensors that live on the Rust side (autopro) and are referenced from
//! Scheme by id.
//!
//! Values created while a program loads belong to the model (e.g. its
//! tokenizer); values created while processing one example belong to that
//! call and are freed right after it. All calls happen on the Scheme thread.
use std::{
    collections::HashMap,
    ffi::{CStr, CString, c_char},
    panic::AssertUnwindSafe,
    sync::{Arc, Mutex},
};

use autopro::{
    cluster::{Dbscan, Metric, centroids},
    detect::{BoxFormat, Detector},
    audio::{Audio, LogMode, MelNorm, MelScale, MelSpectrogram, WaveformProcessor, WhisperFeatures},
    image::{ChannelOrder, Filter, ImageProcessor, Layout, Size, TileProcessor},
    ocr::{DbDecoder, Quad},
    text::{Padding, TextOptions, Tokenizer},
};
use image::{DynamicImage, RgbImage};
use ndarray::{Array1, Array2, ArrayD, Axis, Ix2, IxDyn, Slice};

pub(crate) enum HostValue {
    Bytes(Arc<Vec<u8>>),
    Tokenizer(Arc<Tokenizer>),
    Image(RgbImage),
    Audio(Audio),
    Tensor(ArrayD<f32>),
    /// Strings by index, e.g. a CTC recognizer's characters.
    Vocabulary(Arc<Vec<String>>),
    /// A `(run entry ...)` being assembled or finished.
    Run(Box<RunCall>),
    /// An input stream of the running pipeline.
    Stream(crate::stream::StreamReader),
    /// Where the running pipeline's `emit` goes.
    Emitter(crate::stream::Emit),
    /// A `(client-call ...)` or `(client-sample ...)` being assembled or answered.
    Callback(Box<CallbackCall>),
    /// Gemma 4's audio front end, fed audio as it arrives.
    Mel(Box<crate::usm_mel::UsmMel>),
    /// NemotronLabs VoiceChat's (NeMo) audio front end, fed audio as it arrives.
    NemoMel(Box<crate::nemo_mel::NemoMel>),
    /// A constrained-decoding state (outlines-core).
    Guide(Box<crate::guide::Guide>),
}

pub(crate) enum CallbackKind {
    Call(String),
    Sample { k: usize, full: bool },
}

pub(crate) struct CallbackCall {
    kind: CallbackKind,
    args: Vec<crate::stream::Chunk>,
    logits: Option<ArrayD<f32>>,
    history: Vec<i32>,
    /// After the driver ran it: 0 failed (see `error`), 1 array, 2 audio, 3 image, 4 text, 5 nothing,
    /// 6 token.
    status: i64,
    item: i64,
    text: CString,
    token: i64,
    error: CString,
}

pub(crate) struct RunCall {
    program: i64,
    entry: String,
    inputs: Vec<(String, ArrayD<f32>)>,
    /// After exec: output names and their host ids.
    outputs: Vec<(CString, i64)>,
    /// After exec: the error, if it failed.
    error: CString,
    /// Whether exec ran and succeeded.
    ok: bool,
}

impl HostValue {
    fn kind(&self) -> &'static str {
        match self {
            HostValue::Bytes(_) => "bytes",
            HostValue::Tokenizer(_) => "tokenizer",
            HostValue::Image(_) => "image",
            HostValue::Audio(_) => "audio",
            HostValue::Tensor(_) => "tensor",
            HostValue::Vocabulary(_) => "vocabulary",
            HostValue::Run(_) => "run",
            HostValue::Stream(_) => "stream",
            HostValue::Emitter(_) => "emitter",
            HostValue::Callback(_) => "callback",
            HostValue::Mel(_) | HostValue::NemoMel(_) => "melstream",
            HostValue::Guide(_) => "guide",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Scope {
    Model(i64),
    /// Values of one preprocess, postprocess or pipeline call, by call id.
    Call(u64),
}

/// A scope for a new call.
pub(crate) fn new_call_scope() -> Scope {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    Scope::Call(NEXT.fetch_add(1, Ordering::Relaxed))
}

struct Registry {
    next: i64,
    values: HashMap<i64, (Scope, HostValue)>,
    scope: Scope,
    error: CString,
    /// Last text returned to Scheme (e.g. by detokenize); Chez copies it.
    text: CString,
}

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);

fn with_registry<R>(f: impl FnOnce(&mut Registry) -> R) -> R {
    let mut guard = REGISTRY.lock().unwrap();
    let registry = guard.get_or_insert_with(|| Registry {
        next: 1,
        values: HashMap::new(),
        scope: Scope::Call(0),
        error: CString::default(),
        text: CString::default(),
    });
    f(registry)
}

/// Runs `f` with new values going to `scope`.
pub(crate) fn in_scope<R>(scope: Scope, f: impl FnOnce() -> R) -> R {
    let previous = with_registry(|r| std::mem::replace(&mut r.scope, scope));
    let result = f();
    with_registry(|r| r.scope = previous);
    result
}

pub(crate) fn insert(value: HostValue) -> i64 {
    with_registry(|r| {
        let scope = r.scope;
        insert_in(r, scope, value)
    })
}

fn insert_in(r: &mut Registry, scope: Scope, value: HostValue) -> i64 {
    let id = r.next;
    r.next += 1;
    r.values.insert(id, (scope, value));
    id
}

pub(crate) fn take_tensor(id: i64) -> Option<ArrayD<f32>> {
    with_registry(|r| match r.values.remove(&id) {
        Some((_, HostValue::Tensor(t))) => Some(t),
        Some(other) => {
            r.values.insert(id, other);
            None
        }
        None => None,
    })
}

/// A copy of a tensor that must stay (e.g. one result listed twice).
pub(crate) fn clone_tensor(id: i64) -> Option<ArrayD<f32>> {
    tensor(id).ok()
}

/// Frees every value a call created.
pub(crate) fn clear_scope(scope: Scope) {
    with_registry(|r| r.values.retain(|_, (s, _)| *s != scope));
}

pub(crate) fn drop_model(model: i64) {
    with_registry(|r| r.values.retain(|_, (scope, _)| *scope != Scope::Model(model)));
}

fn get<R>(id: i64, f: impl FnOnce(&HostValue) -> Result<R, String>) -> Result<R, String> {
    with_registry(|r| match r.values.get(&id) {
        Some((_, v)) => f(v),
        None => Err(format!("host value {id} no longer exists")),
    })
}

fn expect_kind<'a>(v: &'a HostValue, kind: &str) -> Result<&'a HostValue, String> {
    if v.kind() == kind { Ok(v) } else { Err(format!("expected a {kind}, got a {}", v.kind())) }
}

fn image(id: i64) -> Result<RgbImage, String> {
    get(id, |v| match expect_kind(v, "image")? {
        HostValue::Image(i) => Ok(i.clone()),
        _ => unreachable!(),
    })
}

fn audio(id: i64) -> Result<Audio, String> {
    get(id, |v| match expect_kind(v, "audio")? {
        HostValue::Audio(a) => Ok(a.clone()),
        _ => unreachable!(),
    })
}

fn tensor(id: i64) -> Result<ArrayD<f32>, String> {
    get(id, |v| match expect_kind(v, "tensor")? {
        HostValue::Tensor(t) => Ok(t.clone()),
        _ => unreachable!(),
    })
}

fn matrix(id: i64, what: &str) -> Result<Array2<f32>, String> {
    let t = tensor(id)?;
    let shape = t.shape().to_vec();
    t.into_dimensionality::<Ix2>().map_err(|_| format!("{what} must be a 2-d array, got shape {shape:?}"))
}

fn boxes(id: i64) -> Result<Array2<f32>, String> {
    let b = matrix(id, "boxes")?;
    if b.ncols() != 4 {
        return Err(format!("boxes must be [n, 4], got {:?}", b.shape()));
    }
    Ok(b)
}

/// A 1-d array of whole numbers >= 0 (indices or classes).
fn indices(id: i64, what: &str) -> Result<Vec<usize>, String> {
    let t = tensor(id)?;
    if t.ndim() != 1 {
        return Err(format!("{what} must be a 1-d array, got shape {:?}", t.shape()));
    }
    t.iter()
        .map(|&v| if v >= 0.0 && v.fract() == 0.0 { Ok(v as usize) } else { Err(format!("{what} must hold whole numbers >= 0, got {v}")) })
        .collect()
}

fn box_format(code: i32) -> Result<BoxFormat, String> {
    match code {
        0 => Ok(BoxFormat::Xyxy),
        1 => Ok(BoxFormat::Xywh),
        2 => Ok(BoxFormat::Cxcywh),
        _ => Err(format!("unknown box format {code}")),
    }
}

fn to_f32<T: Copy + Into<f64>>(values: impl IntoIterator<Item = T>) -> ArrayD<f32> {
    Array1::from_iter(values.into_iter().map(|v| v.into() as f32)).into_dyn()
}

/// Runs an FFI body: errors (and panics) become id 0 plus a message for
/// `tl_host_error`, so nothing unwinds into Chez.
fn ffi(body: impl FnOnce() -> Result<i64, String>) -> i64 {
    let result = std::panic::catch_unwind(AssertUnwindSafe(body)).unwrap_or_else(|p| {
        Err(p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_else(|| "panic".into()))
    });
    match result {
        Ok(v) => v,
        Err(e) => {
            with_registry(|r| r.error = CString::new(e.replace('\0', " ")).unwrap_or_default());
            0
        }
    }
}

fn opt(v: i64) -> Option<usize> {
    (v >= 0).then_some(v as usize)
}

/// Not a Pillow filter: torch's bilinear without antialiasing (OpenCV-like).
const FILTER_BILINEAR_NO_ANTIALIAS: i32 = 6;

fn filter(code: i32) -> Result<Filter, String> {
    Filter::from_pil(code as i64).ok_or_else(|| format!("unknown filter code {code}"))
}

fn text_arg<'a>(s: *const c_char) -> Result<&'a str, String> {
    unsafe { CStr::from_ptr(s) }.to_str().map_err(|_| "text is not valid UTF-8".to_string())
}

// --- FFI functions registered with Chez (see scheme/core.ss for the Scheme side).

extern "C" fn tl_host_error() -> *const c_char {
    with_registry(|r| r.error.as_ptr())
}

extern "C" fn tl_host_tokenizer(bytes: i64) -> i64 {
    ffi(|| {
        let data = get(bytes, |v| match expect_kind(v, "bytes")? {
            HostValue::Bytes(b) => Ok(b.clone()),
            _ => unreachable!(),
        })?;
        let tokenizer = Tokenizer::from_bytes(&data).map_err(|e| e.to_string())?;
        Ok(insert(HostValue::Tokenizer(Arc::new(tokenizer))))
    })
}

/// Returns the id of the ids tensor; the attention mask tensor is id + 1.
extern "C" fn tl_host_tokenize(
    tok: i64,
    text: *const c_char,
    lowercase: i32,
    special: i32,
    max_length: i64,
    pad_to: i64,
    pad_id: i64,
) -> i64 {
    ffi(|| {
        let tokenizer = get(tok, |v| match expect_kind(v, "tokenizer")? {
            HostValue::Tokenizer(t) => Ok(t.clone()),
            _ => unreachable!(),
        })?;
        let text = text_arg(text)?;
        let options = TextOptions {
            lowercase: lowercase != 0,
            add_special_tokens: special != 0,
            max_length: opt(max_length),
            padding: opt(pad_to).map_or(Padding::None, Padding::Fixed),
            pad_id: (pad_id >= 0).then_some(pad_id as u32),
        };
        let encoded = options.encode_batch(&tokenizer, &[text]).map_err(|e| e.to_string())?;
        let to_f32 = |a: ndarray::Array2<i64>| a.row(0).mapv(|v| v as f32).into_dyn();
        let ids = insert(HostValue::Tensor(to_f32(encoded.ids)));
        let mask = insert(HostValue::Tensor(to_f32(encoded.attention_mask)));
        debug_assert_eq!(mask, ids + 1);
        Ok(ids)
    })
}

extern "C" fn tl_host_image_width(id: i64) -> i64 {
    ffi(|| Ok(image(id)?.width() as i64))
}

extern "C" fn tl_host_image_height(id: i64) -> i64 {
    ffi(|| Ok(image(id)?.height() as i64))
}

/// kind 0: exact (a = width, b = height); 1: shortest edge a (longest capped
/// at b if >= 0); 2: longest edge a; 3: multiple of a within [b, c] pixels.
extern "C" fn tl_host_image_resize(id: i64, kind: i32, a: i64, b: i64, c: i64, filter_code: i32) -> i64 {
    ffi(|| {
        let img = image(id)?;
        let size = match kind {
            0 => Size::Exact { width: a as u32, height: b as u32 },
            1 => Size::ShortestEdge { edge: a as u32, max_longest: (b >= 0).then_some(b as u32) },
            2 => Size::LongestEdge(a as u32),
            3 => Size::Multiple { multiple: a as u32, min_pixels: b as u64, max_pixels: c as u64 },
            _ => return Err(format!("unknown resize kind {kind}")),
        };
        let (w, h) = size.output(img.width(), img.height());
        if w == 0 || h == 0 {
            return Err(format!("resize to {w}x{h}"));
        }
        let resized = if filter_code == FILTER_BILINEAR_NO_ANTIALIAS {
            autopro::image::resize_bilinear_no_antialias(&img, w, h)
        } else {
            autopro::image::resize(&img, w, h, filter(filter_code)?)
        };
        Ok(insert(HostValue::Image(resized)))
    })
}

extern "C" fn tl_host_image_center_crop(id: i64, width: i64, height: i64) -> i64 {
    ffi(|| Ok(insert(HostValue::Image(autopro::image::center_crop(&image(id)?, width as u32, height as u32)))))
}

#[allow(clippy::too_many_arguments)]
extern "C" fn tl_host_image_to_tensor(
    id: i64,
    scale: f64,
    normalize: i32,
    m0: f64,
    m1: f64,
    m2: f64,
    s0: f64,
    s1: f64,
    s2: f64,
    bgr: i32,
    hwc: i32,
) -> i64 {
    ffi(|| {
        let processor = ImageProcessor {
            resize: None,
            center_crop: None,
            rescale: (scale >= 0.0).then_some(scale),
            normalize: (normalize != 0).then_some(([m0 as f32, m1 as f32, m2 as f32], [s0 as f32, s1 as f32, s2 as f32])),
            channel_order: if bgr != 0 { ChannelOrder::Bgr } else { ChannelOrder::Rgb },
            layout: if hwc != 0 { Layout::Hwc } else { Layout::Chw },
        };
        Ok(insert(HostValue::Tensor(processor.to_tensor(&image(id)?).into_dyn())))
    })
}

extern "C" fn tl_host_audio_rate(id: i64) -> i64 {
    ffi(|| Ok(audio(id)?.sample_rate as i64))
}

extern "C" fn tl_host_audio_length(id: i64) -> i64 {
    ffi(|| Ok(audio(id)?.samples.len() as i64))
}

extern "C" fn tl_host_audio_resample(id: i64, rate: i64) -> i64 {
    ffi(|| Ok(insert(HostValue::Audio(audio(id)?.resample(rate as u32)))))
}

extern "C" fn tl_host_audio_pad(id: i64, length: i64) -> i64 {
    ffi(|| Ok(insert(HostValue::Audio(audio(id)?.pad_or_truncate(length as usize)))))
}

/// Returns the values tensor; the attention mask is id + 1.
extern "C" fn tl_host_audio_to_tensor(id: i64, normalize: i32, length: i64) -> i64 {
    ffi(|| {
        let a = audio(id)?;
        let processor = WaveformProcessor { sample_rate: a.sample_rate, normalize: normalize != 0, length: opt(length), padding_value: 0.0 };
        let (values, mask) = processor.process(&a).map_err(|e| e.to_string())?;
        let values = insert(HostValue::Tensor(values.into_dyn()));
        insert(HostValue::Tensor(mask.mapv(|v| v as f32).into_dyn()));
        Ok(values)
    })
}

/// flags: bit 0 center; bits 1-2 scale (0 htk, 1 slaney, 2 kaldi); bit 3 slaney
/// norm; bits 4-5 log (0 none, 1 ln, 2 log10).
#[allow(clippy::too_many_arguments)]
extern "C" fn tl_host_log_mel(
    id: i64,
    n_fft: i64,
    hop: i64,
    win: i64,
    mels: i64,
    flags: i64,
    f_min: f64,
    f_max: f64,
    power: f64,
    floor: f64,
) -> i64 {
    ffi(|| {
        let a = audio(id)?;
        let mel = MelSpectrogram {
            sample_rate: a.sample_rate,
            n_fft: n_fft as usize,
            hop_length: hop as usize,
            win_length: if win > 0 { win as usize } else { n_fft as usize },
            n_mels: mels as usize,
            f_min,
            f_max: if f_max > 0.0 { f_max } else { a.sample_rate as f64 / 2.0 },
            power,
            center: flags & 1 != 0,
            mel_scale: match (flags >> 1) & 3 {
                0 => MelScale::Htk,
                1 => MelScale::Slaney,
                _ => MelScale::Kaldi,
            },
            norm: if flags & 8 != 0 { MelNorm::Slaney } else { MelNorm::None },
            mel_floor: floor,
            log: match (flags >> 4) & 3 {
                1 => LogMode::Ln,
                2 => LogMode::Log10,
                _ => LogMode::None,
            },
        };
        if mel.win_length > mel.n_fft || mel.n_fft == 0 || mel.hop_length == 0 {
            return Err("invalid n-fft / hop / win".into());
        }
        Ok(insert(HostValue::Tensor(mel.compute(&a.samples).into_dyn())))
    })
}

extern "C" fn tl_host_whisper_features(id: i64, mels: i64) -> i64 {
    ffi(|| {
        let a = audio(id)?;
        if a.sample_rate != 16000 {
            return Err(format!("whisper features need 16000 Hz audio, got {} Hz (use audio-resample)", a.sample_rate));
        }
        Ok(insert(HostValue::Tensor(WhisperFeatures::new(mels as usize).compute(&a.samples).into_dyn())))
    })
}

extern "C" fn tl_host_tensor_rank(id: i64) -> i64 {
    ffi(|| Ok(tensor(id)?.ndim() as i64))
}

extern "C" fn tl_host_tensor_dim(id: i64, i: i64) -> i64 {
    ffi(|| tensor(id)?.shape().get(i as usize).map(|&d| d as i64).ok_or_else(|| format!("no dimension {i}")))
}

extern "C" fn tl_host_tensor_affine(id: i64, scale: f64, bias: f64) -> i64 {
    ffi(|| Ok(insert(HostValue::Tensor(tensor(id)?.mapv(|v| (v as f64 * scale + bias) as f32)))))
}

extern "C" fn tl_host_tensor_reshape(id: i64, d0: i64, d1: i64, d2: i64, d3: i64) -> i64 {
    ffi(|| {
        let dims: Vec<usize> = [d0, d1, d2, d3].into_iter().take_while(|&d| d > 0).map(|d| d as usize).collect();
        let t = tensor(id)?;
        let t = t.as_standard_layout().into_owned();
        let reshaped = ArrayD::from_shape_vec(IxDyn(&dims), t.into_raw_vec_and_offset().0)
            .map_err(|_| format!("can't reshape to {dims:?}"))?;
        Ok(insert(HostValue::Tensor(reshaped)))
    })
}

// --- Postprocessing and general array functions.

/// A new 1-d array of `n` zeros, filled with `tl_host_tensor_set`.
extern "C" fn tl_host_tensor_new(n: i64) -> i64 {
    ffi(|| Ok(insert(HostValue::Tensor(ArrayD::zeros(IxDyn(&[n.max(0) as usize]))))))
}

/// A new zero array of up to four dims (numpy order; a trailing 0 means no such axis).
extern "C" fn tl_host_tensor_zeros(d0: i64, d1: i64, d2: i64, d3: i64) -> i64 {
    ffi(|| {
        let dims: Vec<usize> = [d0, d1, d2, d3].into_iter().take_while(|&d| d > 0).map(|d| d as usize).collect();
        if dims.is_empty() {
            return Err("zeros needs at least one positive dimension".into());
        }
        Ok(insert(HostValue::Tensor(ArrayD::zeros(IxDyn(&dims)))))
    })
}

/// `a` and `b` joined along `axis` (numpy order).
extern "C" fn tl_host_tensor_concat(a: i64, b: i64, axis: i64) -> i64 {
    ffi(|| {
        let (a, b) = (tensor(a)?, tensor(b)?);
        if axis < 0 || axis as usize >= a.ndim() || a.ndim() != b.ndim() {
            return Err(format!("can't join shapes {:?} and {:?} along axis {axis}", a.shape(), b.shape()));
        }
        let joined = ndarray::concatenate(Axis(axis as usize), &[a.view(), b.view()]).map_err(|e| e.to_string())?;
        Ok(insert(HostValue::Tensor(joined)))
    })
}

extern "C" fn tl_host_tensor_set(id: i64, i: i64, v: f64) -> i64 {
    ffi(|| {
        with_registry(|r| match r.values.get_mut(&id) {
            Some((_, HostValue::Tensor(t))) => {
                let n = t.len();
                let slot = t.as_slice_mut().and_then(|s| s.get_mut(i as usize)).ok_or_else(|| format!("index {i} out of {n}"))?;
                *slot = v as f32;
                Ok(1)
            }
            _ => Err(format!("host value {id} is not an array")),
        })
    })
}

extern "C" fn tl_host_tensor_len(id: i64) -> i64 {
    ffi(|| get(id, |v| match expect_kind(v, "tensor")? {
        HostValue::Tensor(t) => Ok(t.len() as i64),
        _ => unreachable!(),
    }))
}

/// Element `i` in row-major order. Errors can't be signalled; check the length first.
extern "C" fn tl_host_tensor_get(id: i64, i: i64) -> f64 {
    get(id, |v| match v {
        HostValue::Tensor(t) => Ok(match t.as_slice() {
            Some(s) => s.get(i as usize).copied(),
            None => t.iter().nth(i as usize).copied(),
        }
        .map_or(f64::NAN, |v| v as f64)),
        _ => Ok(f64::NAN),
    })
    .unwrap_or(f64::NAN)
}

/// x[start:end] along `axis`; negative bounds count from the end, end = i64::MAX means to the end.
extern "C" fn tl_host_tensor_slice(id: i64, axis: i64, start: i64, end: i64) -> i64 {
    ffi(|| {
        let t = tensor(id)?;
        let axis = usize::try_from(axis).ok().filter(|&a| a < t.ndim()).ok_or_else(|| format!("no axis {axis} in shape {:?}", t.shape()))?;
        let n = t.shape()[axis] as i64;
        let clamp = |v: i64| if v < 0 { (n + v).max(0) } else { v.min(n) };
        let (start, end) = (clamp(start), clamp(end));
        if start > end {
            return Err(format!("empty slice {start}:{end} of axis {axis} (size {n})"));
        }
        Ok(insert(HostValue::Tensor(t.slice_axis(Axis(axis), Slice::from(start as usize..end as usize)).to_owned())))
    })
}

extern "C" fn tl_host_tensor_transpose(id: i64) -> i64 {
    ffi(|| Ok(insert(HostValue::Tensor(tensor(id)?.reversed_axes().as_standard_layout().into_owned()))))
}

/// Entries of `x` along axis 0 at the indices in the 1-d array `rows`.
extern "C" fn tl_host_tensor_take(id: i64, rows: i64) -> i64 {
    ffi(|| {
        let t = tensor(id)?;
        let rows = indices(rows, "indices")?;
        let n = t.shape().first().copied().unwrap_or(0);
        if let Some(bad) = rows.iter().find(|&&r| r >= n) {
            return Err(format!("index {bad} out of range for {n} rows"));
        }
        Ok(insert(HostValue::Tensor(t.select(Axis(0), &rows))))
    })
}

/// Index of the largest value along the last axis (first one on ties).
extern "C" fn tl_host_tensor_argmax(id: i64) -> i64 {
    ffi(|| {
        let best = get(id, |v| match expect_kind(v, "tensor")? {
            HostValue::Tensor(t) => argmax_last(t),
            _ => unreachable!(),
        })?;
        Ok(insert(HostValue::Tensor(best)))
    })
}

fn argmax_last(t: &ArrayD<f32>) -> Result<ArrayD<f32>, String> {
    if t.ndim() == 0 || t.shape()[t.ndim() - 1] == 0 {
        return Err(format!("argmax of an array of shape {:?}", t.shape()));
    }
    let last = Axis(t.ndim() - 1);
    let best = t.map_axis(last, |lane| {
        lane.iter().enumerate().fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0 as f32
    });
    Ok(best)
}

extern "C" fn tl_host_boxes_convert(id: i64, from: i32, to: i32) -> i64 {
    ffi(|| {
        let b = boxes(id)?;
        Ok(insert(HostValue::Tensor(autopro::detect::convert(b.view(), box_format(from)?, box_format(to)?).into_dyn())))
    })
}

/// Kept indices (1-d array) by descending score; `classes` <= 0 means none.
extern "C" fn tl_host_nms(boxes_id: i64, scores: i64, classes: i64, iou: f64) -> i64 {
    ffi(|| {
        let b = boxes(boxes_id)?;
        let s = tensor(scores)?.into_dimensionality::<ndarray::Ix1>().map_err(|_| "scores must be a 1-d array".to_string())?;
        if s.len() != b.nrows() {
            return Err(format!("{} boxes but {} scores", b.nrows(), s.len()));
        }
        let classes = if classes > 0 { Some(indices(classes, "classes")?) } else { None };
        if classes.as_ref().is_some_and(|c| c.len() != b.nrows()) {
            return Err("one class per box".into());
        }
        let keep = autopro::detect::nms(b.view(), s.view(), classes.as_deref(), iou as f32);
        Ok(insert(HostValue::Tensor(to_f32(keep.into_iter().map(|i| i as u32)))))
    })
}

/// Returns boxes [n, 4] xyxy; scores, classes and indices [n] are id + 1, + 2, + 3.
#[allow(clippy::too_many_arguments)]
extern "C" fn tl_host_detect(
    boxes_id: i64,
    scores: i64,
    format: i32,
    score_threshold: f64,
    iou: f64,
    agnostic: i32,
    multi_label: i32,
    max_candidates: i64,
    max_detections: i64,
) -> i64 {
    ffi(|| {
        let b = boxes(boxes_id)?;
        let s = matrix(scores, "class scores")?;
        if s.nrows() != b.nrows() {
            return Err(format!("{} boxes but {} rows of class scores", b.nrows(), s.nrows()));
        }
        let detector = Detector {
            format: box_format(format)?,
            score_threshold: score_threshold as f32,
            iou_threshold: iou as f32,
            class_agnostic: agnostic != 0,
            multi_label: multi_label != 0,
            max_candidates: max_candidates as usize,
            max_detections: max_detections as usize,
        };
        let d = detector.detect(b.view(), s.view());
        let first = insert(HostValue::Tensor(d.boxes.into_dyn()));
        insert(HostValue::Tensor(d.scores.into_dyn()));
        insert(HostValue::Tensor(to_f32(d.classes.into_iter().map(|c| c as u32))));
        insert(HostValue::Tensor(to_f32(d.indices.into_iter().map(|i| i as u32))));
        Ok(first)
    })
}

extern "C" fn tl_host_boxes_scale(id: i64, sx: f64, sy: f64) -> i64 {
    ffi(|| Ok(insert(HostValue::Tensor(autopro::detect::scale(boxes(id)?.view(), sx as f32, sy as f32).into_dyn()))))
}

extern "C" fn tl_host_boxes_clip(id: i64, width: f64, height: f64) -> i64 {
    ffi(|| {
        let mut b = boxes(id)?;
        autopro::detect::clip(&mut b, width as f32, height as f32);
        Ok(insert(HostValue::Tensor(b.into_dyn())))
    })
}

extern "C" fn tl_host_boxes_unletterbox(id: i64, mw: i64, mh: i64, ow: i64, oh: i64) -> i64 {
    ffi(|| {
        let b = autopro::detect::unletterbox(boxes(id)?.view(), (mw as u32, mh as u32), (ow as u32, oh as u32));
        Ok(insert(HostValue::Tensor(b.into_dyn())))
    })
}

extern "C" fn tl_host_image_letterbox(id: i64, width: i64, height: i64, r: i64, g: i64, b: i64, filter_code: i32) -> i64 {
    ffi(|| {
        let fill = [r, g, b].map(|c| c.clamp(0, 255) as u8);
        let img = autopro::image::letterbox(&image(id)?, width as u32, height as u32, fill, filter(filter_code)?);
        Ok(insert(HostValue::Image(img)))
    })
}

/// Docling-style image splitting (Idefics3/SmolVLM `do_image_splitting`):
/// downscale to fit `resize_longest_edge`, split into non-overlapping
/// `tile_edge` x `tile_edge` tiles, append one more tile - the whole image
/// resized down - as a low-resolution global view. Returns the tiles stacked
/// on a new leading axis, `[n, 3, tile_edge, tile_edge]`; rows and cols
/// (id + 1, + 2) are both 0 when the image was small enough not to split (a
/// single tile, the resized whole image).
#[allow(clippy::too_many_arguments)]
extern "C" fn tl_host_image_tile(
    id: i64,
    resize_longest_edge: i64,
    tile_edge: i64,
    do_splitting: i32,
    scale: f64,
    normalize: i32,
    m0: f64,
    m1: f64,
    m2: f64,
    s0: f64,
    s1: f64,
    s2: f64,
    filter_code: i32,
) -> i64 {
    ffi(|| {
        let img = DynamicImage::ImageRgb8(image(id)?);
        let processor = TileProcessor {
            resize_longest_edge: resize_longest_edge as u32,
            tile_edge: tile_edge as u32,
            filter: filter(filter_code)?,
            do_image_splitting: do_splitting != 0,
            rescale: (scale >= 0.0).then_some(scale),
            normalize: (normalize != 0).then_some(([m0 as f32, m1 as f32, m2 as f32], [s0 as f32, s1 as f32, s2 as f32])),
            channel_order: ChannelOrder::Rgb,
        };
        let tiles = processor.process(&img);
        let views: Vec<_> = tiles.tiles.iter().map(|t| t.view()).collect();
        let stacked = ndarray::stack(Axis(0), &views).map_err(|e| e.to_string())?;
        let first = insert(HostValue::Tensor(stacked.into_dyn()));
        insert(HostValue::Tensor(ArrayD::from_elem(IxDyn(&[1]), tiles.rows as f32)));
        insert(HostValue::Tensor(ArrayD::from_elem(IxDyn(&[1]), tiles.cols as f32)));
        Ok(first)
    })
}

/// Cluster label per row of x [n, d]; -1 is noise.
extern "C" fn tl_host_dbscan(id: i64, eps: f64, min_samples: i64, metric: i32) -> i64 {
    ffi(|| {
        let x = matrix(id, "points")?;
        let metric = if metric == 1 { Metric::Cosine } else { Metric::Euclidean };
        let clustering = Dbscan { eps, min_samples: min_samples as usize, metric }.fit(x.view());
        Ok(insert(HostValue::Tensor(to_f32(clustering.labels.iter().copied()))))
    })
}

/// Mean row of x [n, d] per cluster label [n]: [clusters, d], noise ignored.
extern "C" fn tl_host_cluster_centroids(id: i64, labels: i64) -> i64 {
    ffi(|| {
        let x = matrix(id, "points")?;
        let labels = tensor(labels)?;
        if labels.ndim() != 1 || labels.len() != x.nrows() {
            return Err(format!("labels must be [{}], got {:?}", x.nrows(), labels.shape()));
        }
        let labels: Array1<i32> = labels.iter().map(|&v| v as i32).collect();
        let n = labels.iter().copied().max().map_or(0, |m| (m + 1).max(0) as usize);
        Ok(insert(HostValue::Tensor(centroids(x.view(), &labels, n).into_dyn())))
    })
}

// --- OCR: text boxes from DB probability maps, text-line crops, CTC decoding.

/// A probability map [H, W], possibly with leading dimensions of size 1.
fn prob_map(id: i64) -> Result<Array2<f32>, String> {
    let t = tensor(id)?;
    let shape = t.shape().to_vec();
    let n = shape.len();
    if n < 2 || shape[..n - 2].iter().any(|&d| d != 1) {
        return Err(format!("expected a probability map [H, W] (or [1, .., H, W]), got shape {shape:?}"));
    }
    Ok(t.into_shape_with_order((shape[n - 2], shape[n - 1])).map_err(|e| e.to_string())?)
}

fn quads(id: i64) -> Result<Vec<Quad>, String> {
    let t = tensor(id)?;
    if t.ndim() != 3 || t.shape()[1..] != [4, 2] {
        return Err(format!("text boxes must be [n, 4, 2], got {:?}", t.shape()));
    }
    let t = t.as_standard_layout();
    Ok(t.as_slice().unwrap().chunks(8).map(|c| [[c[0], c[1]], [c[2], c[3]], [c[4], c[5]], [c[6], c[7]]]).collect())
}

/// Returns boxes [n, 4, 2] (corners tl, tr, br, bl as x, y); scores [n] are id + 1.
#[allow(clippy::too_many_arguments)]
extern "C" fn tl_host_text_boxes(
    prob: i64,
    width: i64,
    height: i64,
    threshold: f64,
    box_threshold: f64,
    max_candidates: i64,
    unclip_ratio: f64,
    min_size: f64,
) -> i64 {
    ffi(|| {
        let map = prob_map(prob)?;
        let decoder = DbDecoder {
            threshold: threshold as f32,
            box_threshold: box_threshold as f32,
            max_candidates: max_candidates.max(0) as usize,
            unclip_ratio: unclip_ratio as f32,
            min_size: min_size as f32,
        };
        let found = decoder.decode(map.view(), width as u32, height as u32);
        let flat: Vec<f32> = found.boxes.iter().flatten().flatten().copied().collect();
        let boxes = ArrayD::from_shape_vec(IxDyn(&[found.boxes.len(), 4, 2]), flat).map_err(|e| e.to_string())?;
        let first = insert(HostValue::Tensor(boxes));
        insert(HostValue::Tensor(to_f32(found.scores)));
        Ok(first)
    })
}

/// Reading order of text boxes [n, 4, 2]: indices [n].
extern "C" fn tl_host_text_boxes_order(id: i64) -> i64 {
    ffi(|| {
        let order = autopro::ocr::sort_boxes(&quads(id)?);
        Ok(insert(HostValue::Tensor(to_f32(order.into_iter().map(|i| i as u32)))))
    })
}

/// The straightened text line in box `i` of boxes [n, 4, 2].
extern "C" fn tl_host_image_crop_text(img: i64, boxes: i64, i: i64) -> i64 {
    ffi(|| {
        let quads = quads(boxes)?;
        let quad = usize::try_from(i).ok().and_then(|i| quads.get(i)).ok_or_else(|| format!("no box {i} of {}", quads.len()))?;
        let crop = autopro::ocr::crop_text_line(&image(img)?, quad);
        if crop.width() == 0 || crop.height() == 0 {
            return Err(format!("box {i} is empty"));
        }
        Ok(insert(HostValue::Image(crop)))
    })
}

/// Greedy CTC decoding of probabilities [T, classes] (leading 1s allowed):
/// class ids [k]; the mean probability [1] is id + 1.
extern "C" fn tl_host_ctc_greedy(probs: i64) -> i64 {
    ffi(|| {
        let (ids, score) = autopro::ocr::ctc_greedy(prob_map(probs)?.view());
        let first = insert(HostValue::Tensor(to_f32(ids.into_iter().map(|i| i as u32))));
        insert(HostValue::Tensor(to_f32([score])));
        Ok(first)
    })
}

fn vocabulary(id: i64) -> Result<Arc<Vec<String>>, String> {
    get(id, |v| match expect_kind(v, "vocabulary")? {
        HostValue::Vocabulary(v) => Ok(v.clone()),
        _ => unreachable!(),
    })
}

/// A vocabulary from UTF-8 text, one entry per line.
extern "C" fn tl_host_vocabulary(bytes: i64) -> i64 {
    ffi(|| {
        let data = get(bytes, |v| match expect_kind(v, "bytes")? {
            HostValue::Bytes(b) => Ok(b.clone()),
            _ => unreachable!(),
        })?;
        let text = std::str::from_utf8(&data).map_err(|_| "the vocabulary is not UTF-8".to_string())?;
        let text = text.strip_suffix('\n').unwrap_or(text);
        Ok(insert(HostValue::Vocabulary(Arc::new(text.split('\n').map(|l| l.strip_suffix('\r').unwrap_or(l).to_string()).collect()))))
    })
}

extern "C" fn tl_host_vocabulary_size(id: i64) -> i64 {
    ffi(|| Ok(vocabulary(id)?.len() as i64))
}

/// The entries at ids (a 1-d array), concatenated; null on error.
extern "C" fn tl_host_vocabulary_text(vocab: i64, ids: i64) -> *const c_char {
    let ok = ffi(|| {
        let v = vocabulary(vocab)?;
        let mut text = String::new();
        for i in indices(ids, "ids")? {
            text.push_str(v.get(i).ok_or_else(|| format!("id {i} is outside the vocabulary ({} entries)", v.len()))?);
        }
        with_registry(|r| r.text = CString::new(text.replace('\0', "")).unwrap_or_default());
        Ok(1)
    });
    if ok == 0 { std::ptr::null() } else { with_registry(|r| r.text.as_ptr()) }
}

// --- Pipelines: running entries from Scheme, and text from tokens.

extern "C" fn tl_host_run_begin(program: i64, entry: *const c_char) -> i64 {
    ffi(|| {
        let entry = text_arg(entry)?.to_string();
        Ok(insert(HostValue::Run(Box::new(RunCall {
            program,
            entry,
            inputs: Vec::new(),
            outputs: Vec::new(),
            error: CString::default(),
            ok: false,
        }))))
    })
}

extern "C" fn tl_host_run_input(run: i64, name: *const c_char, array: i64) -> i64 {
    ffi(|| {
        let name = text_arg(name)?.to_string();
        let array = tensor(array)?;
        with_registry(|r| match r.values.get_mut(&run) {
            Some((_, HostValue::Run(call))) => {
                call.inputs.push((name, array));
                Ok(1)
            }
            _ => Err(format!("host value {run} is not a run")),
        })
    })
}

/// Runs the entry of a `(run ...)` that the pipeline driver found the pipeline waiting for, on the
/// driver's thread (not the Scheme thread), with the outputs created in `scope`. The result is
/// read back by `tl_host_run_status`.
pub(crate) fn exec_run(run: i64, scope: Scope, state: Option<crate::model::ModelState>) {
    let finish = |outcome: Result<Vec<(CString, i64)>, String>| {
        with_registry(|r| {
            if let Some((_, HostValue::Run(call))) = r.values.get_mut(&run) {
                match outcome {
                    Ok(outputs) => {
                        call.outputs = outputs;
                        call.ok = true;
                    }
                    Err(e) => call.error = CString::new(e.replace('\0', " ")).unwrap_or_default(),
                }
            }
        })
    };
    // Take the inputs out so the registry isn't locked while the model runs.
    let taken = with_registry(|r| match r.values.get_mut(&run) {
        Some((_, HostValue::Run(call))) => Ok((call.program, call.entry.clone(), std::mem::take(&mut call.inputs))),
        _ => Err(format!("host value {run} is not a run")),
    });
    let outcome = taken.and_then(|(program, entry, inputs)| {
        let outputs = crate::model::run_program_entry(program, &entry, inputs, state).map_err(|e| e.to_string())?;
        Ok(with_registry(|r| {
            outputs
                .into_iter()
                .map(|(name, array)| (CString::new(name).unwrap_or_default(), insert_in(r, scope, HostValue::Tensor(array))))
                .collect()
        }))
    });
    finish(outcome);
}

/// After the driver ran the entry: the number of outputs + 1, or 0 if it failed (see
/// tl_host_run_error).
extern "C" fn tl_host_run_status(run: i64) -> i64 {
    with_registry(|r| match r.values.get(&run) {
        Some((_, HostValue::Run(call))) if call.ok => call.outputs.len() as i64 + 1,
        _ => 0,
    })
}

extern "C" fn tl_host_run_error(run: i64) -> *const c_char {
    with_registry(|r| match r.values.get(&run) {
        Some((_, HostValue::Run(call))) => call.error.as_ptr(),
        _ => c"the run is gone".as_ptr(),
    })
}

extern "C" fn tl_host_run_output_name(run: i64, i: i64) -> *const c_char {
    with_registry(|r| match r.values.get(&run) {
        Some((_, HostValue::Run(call))) => call.outputs.get(i as usize).map_or(std::ptr::null(), |(n, _)| n.as_ptr()),
        _ => std::ptr::null(),
    })
}

extern "C" fn tl_host_run_output(run: i64, i: i64) -> i64 {
    ffi(|| {
        with_registry(|r| match r.values.get(&run) {
            Some((_, HostValue::Run(call))) => call.outputs.get(i as usize).map(|(_, id)| *id).ok_or_else(|| format!("no output {i}")),
            _ => Err(format!("host value {run} is not a run")),
        })
    })
}

// --- Gemma 4's streaming log-mel front end: `(usm-mel-new)`, `(usm-mel-push! ms audio)`, `(usm-mel-window ms from n)`.

extern "C" fn tl_host_mel_new() -> i64 {
    ffi(|| Ok(insert(HostValue::Mel(Box::new(crate::usm_mel::UsmMel::new())))))
}

/// Feeds an audio chunk (16 kHz mono); returns the number of frames so far + 1.
extern "C" fn tl_host_mel_push(mel: i64, audio: i64) -> i64 {
    ffi(|| {
        let samples = get(audio, |v| match v {
            HostValue::Audio(a) if a.sample_rate == 16000 => Ok(a.samples.clone()),
            HostValue::Audio(a) => Err(format!("the audio front end takes 16 kHz audio, this is {} Hz", a.sample_rate)),
            other => Err(format!("expected audio, not a {}", other.kind())),
        })?;
        with_registry(|r| match r.values.get_mut(&mel) {
            Some((_, HostValue::Mel(m))) => {
                m.push(&samples);
                Ok(m.frames() as i64 + 1)
            }
            _ => Err(format!("host value {mel} is not an audio front end")),
        })
    })
}

/// Frames so far, + 1.
extern "C" fn tl_host_mel_frames(mel: i64) -> i64 {
    ffi(|| {
        with_registry(|r| match r.values.get(&mel) {
            Some((_, HostValue::Mel(m))) => Ok(m.frames() as i64 + 1),
            _ => Err(format!("host value {mel} is not an audio front end")),
        })
    })
}

/// Frames `from .. from + n` as an [n, 128] array, zeros where there are none.
extern "C" fn tl_host_mel_window(mel: i64, from: i64, n: i64) -> i64 {
    ffi(|| {
        let window = with_registry(|r| match r.values.get(&mel) {
            Some((_, HostValue::Mel(m))) => Ok(m.window(from, n.max(0) as usize)),
            _ => Err(format!("host value {mel} is not an audio front end")),
        })?;
        Ok(insert(HostValue::Tensor(window.into_dyn())))
    })
}

// --- Random numbers for samplers in pipelines: `(random-seed! n)`, `(random-uniform n)`, `(random-normal n)` (arrays of n values).

thread_local! {
    static RNG_STATE: std::cell::Cell<u64> = const { std::cell::Cell::new(0x9E37_79B9_7F4A_7C15) };
}

/// splitmix64: a full-period generator that is good enough for sampling and needs no dependency.
fn next_u64() -> u64 {
    RNG_STATE.with(|s| {
        let z = s.get().wrapping_add(0x9E37_79B9_7F4A_7C15);
        s.set(z);
        let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    })
}

/// Uniform in the open interval (0, 1).
fn next_uniform() -> f64 {
    ((next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

extern "C" fn tl_host_random_seed(seed: i64) -> i64 {
    ffi(|| {
        RNG_STATE.with(|s| s.set(seed as u64));
        Ok(1)
    })
}

/// kind 0: uniform (0, 1); kind 1: standard normal (Box-Muller).
extern "C" fn tl_host_random(kind: i64, n: i64) -> i64 {
    ffi(|| {
        if !(0..=1 << 24).contains(&n) {
            return Err(format!("random array of {n} values"));
        }
        let values: Vec<f32> = (0..n)
            .map(|_| match kind {
                0 => next_uniform() as f32,
                _ => ((-2.0 * next_uniform().ln()).sqrt() * (2.0 * std::f64::consts::PI * next_uniform()).cos()) as f32,
            })
            .collect();
        Ok(insert(HostValue::Tensor(ArrayD::from_shape_vec(vec![n as usize], values).map_err(|e| e.to_string())?)))
    })
}

// --- NeMo's log-mel front end (VoiceChat): `(nemo-mel-new)`, `(nemo-mel-push! ms audio)`, `(nemo-mel-finish! ms)`,
// `(nemo-mel-window ms from n)`.

extern "C" fn tl_host_nemo_mel_new() -> i64 {
    ffi(|| Ok(insert(HostValue::NemoMel(Box::new(crate::nemo_mel::NemoMel::new())))))
}

fn with_nemo_mel<T>(mel: i64, f: impl FnOnce(&mut crate::nemo_mel::NemoMel) -> T) -> Result<T, String> {
    with_registry(|r| match r.values.get_mut(&mel) {
        Some((_, HostValue::NemoMel(m))) => Ok(f(m)),
        _ => Err(format!("host value {mel} is not a NeMo audio front end")),
    })
}

/// Feeds an audio chunk (16 kHz mono); returns the number of frames so far + 1.
extern "C" fn tl_host_nemo_mel_push(mel: i64, audio: i64) -> i64 {
    ffi(|| {
        let samples = get(audio, |v| match v {
            HostValue::Audio(a) if a.sample_rate == 16000 => Ok(a.samples.clone()),
            HostValue::Audio(a) => Err(format!("the audio front end takes 16 kHz audio, this is {} Hz", a.sample_rate)),
            other => Err(format!("expected audio, not a {}", other.kind())),
        })?;
        with_nemo_mel(mel, |m| {
            m.push(&samples);
            m.frames() as i64 + 1
        })
    })
}

/// Ends the audio; returns the number of frames + 1.
extern "C" fn tl_host_nemo_mel_finish(mel: i64) -> i64 {
    ffi(|| {
        with_nemo_mel(mel, |m| {
            m.finish();
            m.frames() as i64 + 1
        })
    })
}

extern "C" fn tl_host_nemo_mel_frames(mel: i64) -> i64 {
    ffi(|| with_nemo_mel(mel, |m| m.frames() as i64 + 1))
}

/// Frames `from .. from + n` as an [n, 128] array, zeros where there are none.
extern "C" fn tl_host_nemo_mel_window(mel: i64, from: i64, n: i64) -> i64 {
    ffi(|| {
        let window = with_nemo_mel(mel, |m| m.window(from, n.max(0) as usize))?;
        Ok(insert(HostValue::Tensor(window.into_dyn())))
    })
}

// --- Constrained decoding: `(tool-call-regex tools-json)`, `(guide-new tokenizer-json regex eos-id)`, `(guide-mask g)`,
// `(guide-advance! g token)`, `(guide-final? g)`, `(guide-reset! g)`.

extern "C" fn tl_host_tool_call_regex(tools: *const c_char, max_calls: i64) -> *const c_char {
    let ok = ffi(|| {
        let re = crate::guide::tool_call_regex(text_arg(tools)?, max_calls.max(1) as usize)?;
        with_registry(|r| r.text = CString::new(re.replace('\0', "")).unwrap_or_default());
        Ok(1)
    });
    if ok == 0 { std::ptr::null() } else { with_registry(|r| r.text.as_ptr()) }
}

/// The tools as the system prompt shows them (see guide::tools_prompt_json).
extern "C" fn tl_host_tools_prompt(tools: *const c_char) -> *const c_char {
    let ok = ffi(|| {
        let text = crate::guide::tools_prompt_json(text_arg(tools)?)?;
        with_registry(|r| r.text = CString::new(text.replace('\0', "")).unwrap_or_default());
        Ok(1)
    });
    if ok == 0 { std::ptr::null() } else { with_registry(|r| r.text.as_ptr()) }
}

extern "C" fn tl_host_guide_new(tokenizer_json: i64, regex: *const c_char, eos: i64) -> i64 {
    ffi(|| {
        let data = get(tokenizer_json, |v| match expect_kind(v, "bytes")? {
            HostValue::Bytes(b) => Ok(b.clone()),
            _ => unreachable!(),
        })?;
        let guide = crate::guide::Guide::new(&data, text_arg(regex)?, eos as u32)?;
        Ok(insert(HostValue::Guide(Box::new(guide))))
    })
}

fn with_guide<T>(g: i64, f: impl FnOnce(&mut crate::guide::Guide) -> T) -> Result<T, String> {
    with_registry(|r| match r.values.get_mut(&g) {
        Some((_, HostValue::Guide(x))) => Ok(f(x)),
        _ => Err(format!("host value {g} is not a guide")),
    })
}

/// The allowed-token mask, an [1, vocab] array of 0 / -1e30.
extern "C" fn tl_host_guide_mask(g: i64) -> i64 {
    ffi(|| {
        let m = with_guide(g, |x| x.mask())?;
        let n = m.len();
        Ok(insert(HostValue::Tensor(ArrayD::from_shape_vec(vec![1, n], m).map_err(|e| e.to_string())?)))
    })
}

/// 2 if the token was accepted, 1 if not.
extern "C" fn tl_host_guide_advance(g: i64, token: i64) -> i64 {
    ffi(|| with_guide(g, |x| if x.advance(token as u32) { 2 } else { 1 }))
}

extern "C" fn tl_host_guide_final(g: i64) -> i64 {
    ffi(|| with_guide(g, |x| if x.is_final() { 2 } else { 1 }))
}

extern "C" fn tl_host_guide_reset(g: i64) -> i64 {
    ffi(|| with_guide(g, |x| {
        x.reset();
        1
    }))
}

// --- Callbacks: `(client-call name arg ...)` and `(client-sample logits ...)` in pipelines.

fn new_callback(kind: CallbackKind, logits: Option<ArrayD<f32>>, history: Vec<i32>) -> i64 {
    insert(HostValue::Callback(Box::new(CallbackCall {
        kind,
        args: Vec::new(),
        logits,
        history,
        status: 0,
        item: 0,
        text: CString::default(),
        token: 0,
        error: CString::default(),
    })))
}

extern "C" fn tl_host_cb_call(name: *const c_char) -> i64 {
    ffi(|| Ok(new_callback(CallbackKind::Call(text_arg(name)?.to_string()), None, Vec::new())))
}

/// Starts a sample request: logits (any shape, flattened), history (token ids), the number of
/// candidates to send and whether to send the whole array too.
extern "C" fn tl_host_cb_sample(logits: i64, history: i64, k: i64, full: i32) -> i64 {
    ffi(|| {
        let logits = tensor(logits)?;
        let history: Vec<i32> = tensor(history)?.iter().map(|&v| v as i32).collect();
        if k < 1 {
            return Err("k must be at least 1".into());
        }
        Ok(new_callback(CallbackKind::Sample { k: k as usize, full: full != 0 }, Some(logits), history))
    })
}

fn add_arg(cb: i64, chunk: crate::stream::Chunk) -> i64 {
    ffi(|| {
        with_registry(|r| match r.values.get_mut(&cb) {
            Some((_, HostValue::Callback(call))) => {
                call.args.push(chunk);
                Ok(1)
            }
            _ => Err(format!("host value {cb} is not a callback")),
        })
    })
}

extern "C" fn tl_host_cb_arg(cb: i64, value: i64) -> i64 {
    use crate::stream::Chunk;
    let chunk = get(value, |v| match v {
        HostValue::Tensor(t) => Ok(Chunk::Array(t.clone())),
        HostValue::Audio(a) => Ok(Chunk::Audio(a.clone())),
        HostValue::Image(i) => Ok(Chunk::Image(i.clone())),
        other => Err(format!("can't send a {} to the client", other.kind())),
    });
    match chunk {
        Ok(chunk) => add_arg(cb, chunk),
        Err(e) => ffi(|| Err(e)),
    }
}

extern "C" fn tl_host_cb_arg_text(cb: i64, text: *const c_char) -> i64 {
    match text_arg(text) {
        Ok(t) => add_arg(cb, crate::stream::Chunk::Text(t.to_string())),
        Err(e) => ffi(|| Err(e)),
    }
}

extern "C" fn tl_host_cb_arg_number(cb: i64, value: f64) -> i64 {
    add_arg(cb, crate::stream::Chunk::Array(ndarray::arr0(value as f32).into_dyn()))
}

/// Answers the callback the pipeline is waiting for, on the driver's thread; `step` counts this run's
/// sample requests. The result is read back by `tl_host_cb_status` and friends.
pub(crate) fn exec_callback(
    id: i64,
    scope: Scope,
    callbacks: Option<&dyn crate::stream::Callbacks>,
    step: &mut u32,
) {
    use crate::stream::{Chunk, SampleRequest};
    let taken = with_registry(|r| match r.values.get_mut(&id) {
        Some((_, HostValue::Callback(call))) => {
            let kind = match &call.kind {
                CallbackKind::Call(n) => CallbackKind::Call(n.clone()),
                CallbackKind::Sample { k, full } => CallbackKind::Sample { k: *k, full: *full },
            };
            Ok((kind, std::mem::take(&mut call.args), call.logits.take(), std::mem::take(&mut call.history)))
        }
        _ => Err(format!("host value {id} is not a callback")),
    });
    // (status, result chunk, text, token) or an error
    let outcome: Result<(i64, Option<Chunk>, i64), String> = taken.and_then(|(kind, args, logits, history)| {
        let callbacks = callbacks.ok_or_else(|| "this pipeline was started without callbacks".to_string())?;
        match kind {
            CallbackKind::Call(name) => match callbacks.call(&name, args)? {
                None => Ok((5, None, 0)),
                Some(chunk) => {
                    let code = match &chunk {
                        Chunk::Array(_) => 1,
                        Chunk::Audio(_) => 2,
                        Chunk::Image(_) => 3,
                        Chunk::Text(_) => 4,
                    };
                    Ok((code, Some(chunk), 0))
                }
            },
            CallbackKind::Sample { k, full } => {
                let logits = logits.ok_or("no logits")?;
                let flat: Vec<f32> = logits.iter().copied().collect();
                let k = k.min(flat.len());
                // Highest first, ties by index. Only the k best are sorted (the vocabulary can be 262k).
                let order = |&a: &usize, &b: &usize| flat[b].partial_cmp(&flat[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b));
                let mut ids: Vec<usize> = (0..flat.len()).collect();
                if k < ids.len() {
                    ids.select_nth_unstable_by(k - 1, order);
                    ids.truncate(k);
                }
                ids.sort_by(order);
                let candidates = ids.iter().map(|&i| (i as i32, flat[i])).collect();
                let request = SampleRequest { step: *step, candidates, history, full: full.then_some(logits) };
                *step += 1;
                Ok((6, None, callbacks.sample(request)? as i64))
            }
        }
    });
    with_registry(|r| {
        let outcome = outcome.map(|(status, chunk, token)| {
            let (item, text) = match chunk {
                Some(Chunk::Array(a)) => (insert_in(r, scope, HostValue::Tensor(a)), CString::default()),
                Some(Chunk::Audio(a)) => (insert_in(r, scope, HostValue::Audio(a)), CString::default()),
                Some(Chunk::Image(i)) => (insert_in(r, scope, HostValue::Image(i)), CString::default()),
                Some(Chunk::Text(t)) => (0, CString::new(t.replace('\0', "")).unwrap_or_default()),
                None => (0, CString::default()),
            };
            (status, item, text, token)
        });
        if let Some((_, HostValue::Callback(call))) = r.values.get_mut(&id) {
            match outcome {
                Ok((status, item, text, token)) => {
                    call.status = status;
                    call.item = item;
                    call.text = text;
                    call.token = token;
                }
                Err(e) => call.error = CString::new(e.replace('\0', " ")).unwrap_or_default(),
            }
        }
    });
}

fn callback<R>(id: i64, f: impl FnOnce(&CallbackCall) -> R) -> Option<R> {
    with_registry(|r| match r.values.get(&id) {
        Some((_, HostValue::Callback(call))) => Some(f(call)),
        _ => None,
    })
}

extern "C" fn tl_host_cb_status(cb: i64) -> i64 {
    callback(cb, |c| c.status).unwrap_or(0)
}

extern "C" fn tl_host_cb_item(cb: i64) -> i64 {
    callback(cb, |c| c.item).unwrap_or(0)
}

extern "C" fn tl_host_cb_token(cb: i64) -> i64 {
    callback(cb, |c| c.token).unwrap_or(0)
}

extern "C" fn tl_host_cb_text(cb: i64) -> *const c_char {
    callback(cb, |c| c.text.as_ptr()).unwrap_or(c"".as_ptr())
}

extern "C" fn tl_host_cb_error(cb: i64) -> *const c_char {
    callback(cb, |c| c.error.as_ptr()).unwrap_or(c"the callback is gone".as_ptr())
}

// --- Streams: `(stream-recv s)` and `(emit name value)` in pipelines.

/// Takes the next chunk of an input stream if there is one: -2 if not yet (the pipeline then
/// yields), 0 at the end, 1 array, 2 audio, 3 image, 4 text (read it with tl_host_stream_text), -1 on
/// error. Array, audio and image chunks are then fetched with tl_host_stream_item.
extern "C" fn tl_host_stream_poll(stream: i64) -> i64 {
    use crate::stream::{Chunk, Poll};
    let reader = get(stream, |v| match v {
        HostValue::Stream(r) => Ok(r.clone()),
        _ => Err(format!("host value {stream} is not a stream")),
    });
    let Ok(reader) = reader else {
        with_registry(|r| r.error = CString::new("not a stream").unwrap());
        return -1;
    };
    let chunk = match reader.inner.poll() {
        Poll::Empty => return -2,
        Poll::Closed => return 0,
        Poll::Chunk(chunk) => chunk,
    };
    let (code, id) = match chunk {
        Chunk::Array(a) => (1, insert(HostValue::Tensor(a))),
        Chunk::Audio(a) => (2, insert(HostValue::Audio(a))),
        Chunk::Image(i) => (3, insert(HostValue::Image(i))),
        Chunk::Text(t) => {
            *reader.inner.text.lock().unwrap() = CString::new(t.replace('\0', "")).unwrap_or_default();
            (4, 0)
        }
    };
    *reader.inner.last.lock().unwrap() = id;
    code
}

/// The input stream behind a host id, for the driver to wait on.
pub(crate) fn stream_reader(id: i64) -> Option<crate::stream::StreamReader> {
    get(id, |v| match v {
        HostValue::Stream(r) => Ok(r.clone()),
        _ => Err(String::new()),
    })
    .ok()
}

extern "C" fn tl_host_stream_item(stream: i64) -> i64 {
    ffi(|| {
        get(stream, |v| match v {
            HostValue::Stream(r) => Ok(*r.inner.last.lock().unwrap()),
            _ => Err(format!("host value {stream} is not a stream")),
        })
    })
}

extern "C" fn tl_host_stream_text(stream: i64) -> *const c_char {
    get(stream, |v| match v {
        HostValue::Stream(r) => Ok(r.inner.text.lock().unwrap().as_ptr()),
        _ => Err(String::new()),
    })
    .unwrap_or(c"".as_ptr())
}

fn emitter(id: i64) -> Result<crate::stream::Emit, String> {
    get(id, |v| match v {
        HostValue::Emitter(e) => Ok(e.clone()),
        _ => Err(format!("host value {id} is not an emitter")),
    })
}

fn send(emit: i64, name: *const c_char, chunk: crate::stream::Chunk) -> i64 {
    ffi(|| {
        let name = text_arg(name)?.to_string();
        if emitter(emit)?(&name, chunk) { Ok(1) } else { Err(format!("nobody is listening to the output {name:?} any more")) }
    })
}

/// Emits an array, audio or image host value (cloned: the pipeline may keep using it).
extern "C" fn tl_host_emit(emit: i64, name: *const c_char, value: i64) -> i64 {
    use crate::stream::Chunk;
    let chunk = get(value, |v| match v {
        HostValue::Tensor(t) => Ok(Chunk::Array(t.clone())),
        HostValue::Audio(a) => Ok(Chunk::Audio(a.clone())),
        HostValue::Image(i) => Ok(Chunk::Image(i.clone())),
        other => Err(format!("can't emit a {}", other.kind())),
    });
    match chunk {
        Ok(chunk) => send(emit, name, chunk),
        Err(e) => {
            with_registry(|r| r.error = CString::new(e.replace('\0', " ")).unwrap_or_default());
            0
        }
    }
}

extern "C" fn tl_host_emit_text(emit: i64, name: *const c_char, text: *const c_char) -> i64 {
    match text_arg(text) {
        Ok(t) => send(emit, name, crate::stream::Chunk::Text(t.to_string())),
        Err(e) => {
            with_registry(|r| r.error = CString::new(e).unwrap_or_default());
            0
        }
    }
}

fn tokenizer(id: i64) -> Result<Arc<Tokenizer>, String> {
    get(id, |v| match expect_kind(v, "tokenizer")? {
        HostValue::Tokenizer(t) => Ok(t.clone()),
        _ => unreachable!(),
    })
}

/// Text of token ids (a 1-d array of whole numbers); null on error (see tl_host_error).
extern "C" fn tl_host_detokenize(tok: i64, ids: i64, skip_special: i32) -> *const c_char {
    let ok = ffi(|| {
        let tokenizer = tokenizer(tok)?;
        let ids: Vec<u32> = indices(ids, "token ids")?.into_iter().map(|i| i as u32).collect();
        let text = tokenizer.decode(&ids, skip_special != 0).map_err(|e| e.to_string())?;
        with_registry(|r| r.text = CString::new(text.replace('\0', "")).unwrap_or_default());
        Ok(1)
    });
    if ok == 0 { std::ptr::null() } else { with_registry(|r| r.text.as_ptr()) }
}

/// Id of a token, or -1.
extern "C" fn tl_host_token_id(tok: i64, token: *const c_char) -> i64 {
    let (Ok(tokenizer), Ok(token)) = (tokenizer(tok), text_arg(token)) else { return -1 };
    tokenizer.token_to_id(token).map_or(-1, |id| id as i64)
}

/// (name, address) of every host function, to register with Chez.
pub(crate) fn symbols() -> Vec<(&'static str, *const std::ffi::c_void)> {
    vec![
        ("tl_host_error", tl_host_error as *const _),
        ("tl_host_tokenizer", tl_host_tokenizer as *const _),
        ("tl_host_tokenize", tl_host_tokenize as *const _),
        ("tl_host_image_width", tl_host_image_width as *const _),
        ("tl_host_image_height", tl_host_image_height as *const _),
        ("tl_host_image_resize", tl_host_image_resize as *const _),
        ("tl_host_image_center_crop", tl_host_image_center_crop as *const _),
        ("tl_host_image_to_tensor", tl_host_image_to_tensor as *const _),
        ("tl_host_audio_rate", tl_host_audio_rate as *const _),
        ("tl_host_audio_length", tl_host_audio_length as *const _),
        ("tl_host_audio_resample", tl_host_audio_resample as *const _),
        ("tl_host_audio_pad", tl_host_audio_pad as *const _),
        ("tl_host_audio_to_tensor", tl_host_audio_to_tensor as *const _),
        ("tl_host_log_mel", tl_host_log_mel as *const _),
        ("tl_host_whisper_features", tl_host_whisper_features as *const _),
        ("tl_host_tensor_rank", tl_host_tensor_rank as *const _),
        ("tl_host_tensor_dim", tl_host_tensor_dim as *const _),
        ("tl_host_tensor_affine", tl_host_tensor_affine as *const _),
        ("tl_host_tensor_reshape", tl_host_tensor_reshape as *const _),
        ("tl_host_tensor_new", tl_host_tensor_new as *const _),
        ("tl_host_tensor_set", tl_host_tensor_set as *const _),
        ("tl_host_tensor_len", tl_host_tensor_len as *const _),
        ("tl_host_tensor_get", tl_host_tensor_get as *const _),
        ("tl_host_tensor_slice", tl_host_tensor_slice as *const _),
        ("tl_host_tensor_zeros", tl_host_tensor_zeros as *const _),
        ("tl_host_tensor_concat", tl_host_tensor_concat as *const _),
        ("tl_host_tensor_transpose", tl_host_tensor_transpose as *const _),
        ("tl_host_tensor_take", tl_host_tensor_take as *const _),
        ("tl_host_tensor_argmax", tl_host_tensor_argmax as *const _),
        ("tl_host_boxes_convert", tl_host_boxes_convert as *const _),
        ("tl_host_nms", tl_host_nms as *const _),
        ("tl_host_detect", tl_host_detect as *const _),
        ("tl_host_boxes_scale", tl_host_boxes_scale as *const _),
        ("tl_host_boxes_clip", tl_host_boxes_clip as *const _),
        ("tl_host_boxes_unletterbox", tl_host_boxes_unletterbox as *const _),
        ("tl_host_image_letterbox", tl_host_image_letterbox as *const _),
        ("tl_host_image_tile", tl_host_image_tile as *const _),
        ("tl_host_dbscan", tl_host_dbscan as *const _),
        ("tl_host_cluster_centroids", tl_host_cluster_centroids as *const _),
        ("tl_host_run_begin", tl_host_run_begin as *const _),
        ("tl_host_run_input", tl_host_run_input as *const _),
        ("tl_host_run_status", tl_host_run_status as *const _),
        ("tl_host_run_error", tl_host_run_error as *const _),
        ("tl_host_run_output_name", tl_host_run_output_name as *const _),
        ("tl_host_run_output", tl_host_run_output as *const _),
        ("tl_host_mel_new", tl_host_mel_new as *const _),
        ("tl_host_mel_push", tl_host_mel_push as *const _),
        ("tl_host_mel_window", tl_host_mel_window as *const _),
        ("tl_host_mel_frames", tl_host_mel_frames as *const _),
        ("tl_host_random_seed", tl_host_random_seed as *const _),
        ("tl_host_random", tl_host_random as *const _),
        ("tl_host_nemo_mel_new", tl_host_nemo_mel_new as *const _),
        ("tl_host_tool_call_regex", tl_host_tool_call_regex as *const _),
        ("tl_host_tools_prompt", tl_host_tools_prompt as *const _),
        ("tl_host_guide_new", tl_host_guide_new as *const _),
        ("tl_host_guide_mask", tl_host_guide_mask as *const _),
        ("tl_host_guide_advance", tl_host_guide_advance as *const _),
        ("tl_host_guide_final", tl_host_guide_final as *const _),
        ("tl_host_guide_reset", tl_host_guide_reset as *const _),
        ("tl_host_nemo_mel_push", tl_host_nemo_mel_push as *const _),
        ("tl_host_nemo_mel_finish", tl_host_nemo_mel_finish as *const _),
        ("tl_host_nemo_mel_frames", tl_host_nemo_mel_frames as *const _),
        ("tl_host_nemo_mel_window", tl_host_nemo_mel_window as *const _),
        ("tl_host_cb_call", tl_host_cb_call as *const _),
        ("tl_host_cb_sample", tl_host_cb_sample as *const _),
        ("tl_host_cb_arg", tl_host_cb_arg as *const _),
        ("tl_host_cb_arg_text", tl_host_cb_arg_text as *const _),
        ("tl_host_cb_arg_number", tl_host_cb_arg_number as *const _),
        ("tl_host_cb_status", tl_host_cb_status as *const _),
        ("tl_host_cb_item", tl_host_cb_item as *const _),
        ("tl_host_cb_token", tl_host_cb_token as *const _),
        ("tl_host_cb_text", tl_host_cb_text as *const _),
        ("tl_host_cb_error", tl_host_cb_error as *const _),
        ("tl_host_stream_poll", tl_host_stream_poll as *const _),
        ("tl_host_stream_item", tl_host_stream_item as *const _),
        ("tl_host_stream_text", tl_host_stream_text as *const _),
        ("tl_host_emit", tl_host_emit as *const _),
        ("tl_host_emit_text", tl_host_emit_text as *const _),
        ("tl_host_detokenize", tl_host_detokenize as *const _),
        ("tl_host_token_id", tl_host_token_id as *const _),
        ("tl_host_text_boxes", tl_host_text_boxes as *const _),
        ("tl_host_text_boxes_order", tl_host_text_boxes_order as *const _),
        ("tl_host_image_crop_text", tl_host_image_crop_text as *const _),
        ("tl_host_ctc_greedy", tl_host_ctc_greedy as *const _),
        ("tl_host_vocabulary", tl_host_vocabulary as *const _),
        ("tl_host_vocabulary_size", tl_host_vocabulary_size as *const _),
        ("tl_host_vocabulary_text", tl_host_vocabulary_text as *const _),
    ]
}
