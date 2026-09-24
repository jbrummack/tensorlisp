//! Host values for preprocessing: tokenizers, images, audio and tensors that
//! live on the Rust side (autopro) and are referenced from Scheme by id.
//!
//! Values created while a program loads belong to the model (e.g. its
//! tokenizer); values created while preprocessing one example belong to that
//! call and are freed right after it. All calls happen on the Scheme thread.
use std::{
    collections::HashMap,
    ffi::{CStr, CString, c_char},
    panic::AssertUnwindSafe,
    sync::{Arc, Mutex},
};

use autopro::{
    audio::{Audio, LogMode, MelNorm, MelScale, MelSpectrogram, WaveformProcessor, WhisperFeatures},
    image::{ChannelOrder, Filter, ImageProcessor, Layout, Size},
    text::{Padding, TextOptions, Tokenizer},
};
use image::RgbImage;
use ndarray::{ArrayD, IxDyn};

pub(crate) enum HostValue {
    Bytes(Arc<Vec<u8>>),
    Tokenizer(Arc<Tokenizer>),
    Image(RgbImage),
    Audio(Audio),
    Tensor(ArrayD<f32>),
}

impl HostValue {
    fn kind(&self) -> &'static str {
        match self {
            HostValue::Bytes(_) => "bytes",
            HostValue::Tokenizer(_) => "tokenizer",
            HostValue::Image(_) => "image",
            HostValue::Audio(_) => "audio",
            HostValue::Tensor(_) => "tensor",
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
}

static REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);

fn with_registry<R>(f: impl FnOnce(&mut Registry) -> R) -> R {
    let mut guard = REGISTRY.lock().unwrap();
    let registry = guard.get_or_insert_with(|| Registry {
        next: 1,
        values: HashMap::new(),
        scope: Scope::Call,
        error: CString::default(),
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
        Ok(insert(HostValue::Image(autopro::image::resize(&img, w, h, filter(filter_code)?))))
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
    ]
}
