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
}

pub(crate) struct RunCall {
    program: i64,
    entry: String,
    inputs: Vec<(String, ArrayD<f32>)>,
    /// After exec: output names and their host ids.
    outputs: Vec<(CString, i64)>,
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
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Scope {
    Model(i64),
    Call,
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
        scope: Scope::Call,
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
        let id = r.next;
        r.next += 1;
        r.values.insert(id, (r.scope, value));
        id
    })
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

/// Frees every value created by preprocessing calls.
pub(crate) fn clear_calls() {
    with_registry(|r| r.values.retain(|_, (scope, _)| *scope != Scope::Call));
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
        Ok(insert(HostValue::Run(Box::new(RunCall { program, entry, inputs: Vec::new(), outputs: Vec::new() }))))
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

/// Runs the entry; returns the number of outputs + 1 (0 on error).
extern "C" fn tl_host_run_exec(run: i64) -> i64 {
    ffi(|| {
        // Take the inputs out so the registry isn't locked while the model runs
        // (which may build a graph through Scheme).
        let (program, entry, inputs) = with_registry(|r| match r.values.get_mut(&run) {
            Some((_, HostValue::Run(call))) => Ok((call.program, call.entry.clone(), std::mem::take(&mut call.inputs))),
            _ => Err(format!("host value {run} is not a run")),
        })?;
        let outputs = crate::model::run_program_entry(program, &entry, inputs).map_err(|e| e.to_string())?;
        let outputs: Vec<(CString, i64)> = outputs
            .into_iter()
            .map(|(name, array)| (CString::new(name).unwrap_or_default(), insert(HostValue::Tensor(array))))
            .collect();
        let n = outputs.len() as i64;
        with_registry(|r| {
            if let Some((_, HostValue::Run(call))) = r.values.get_mut(&run) {
                call.outputs = outputs;
            }
        });
        Ok(n + 1)
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
        ("tl_host_run_exec", tl_host_run_exec as *const _),
        ("tl_host_run_output_name", tl_host_run_output_name as *const _),
        ("tl_host_run_output", tl_host_run_output as *const _),
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
