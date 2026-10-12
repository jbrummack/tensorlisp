//! The log-mel front end of Gemma 4's audio tower (transformers' `Gemma4AudioFeatureExtractor`), computed
//! incrementally: frames are produced as soon as the samples they cover have arrived.
//!
//! 16 kHz mono audio, 20 ms frames (320 samples) every 10 ms, a periodic Hann window, a 512-point FFT,
//! magnitudes, 128 HTK mel filters from 0 to 8 kHz, then ln(mel + 0.001). The waveform is conceptually preceded
//! by 160 zeros (the first frame is centred on t = 0), and a frame exists once its 320 samples plus one more
//! (the extractor unfolds 321-sample windows) are there, so frame i needs padded samples [160 i, 160 i + 321).
use ndarray::Array2;

const SAMPLE_RATE: u32 = 16000;
const FRAME: usize = 320;
const HOP: usize = 160;
const FFT: usize = 512;
const BINS: usize = FFT / 2 + 1;
const MELS: usize = 128;
const MEL_FLOOR: f64 = 1e-3;

pub struct UsmMel {
    window: Vec<f64>,
    /// [BINS][MELS]
    filters: Vec<f64>,
    twiddles: Vec<(f64, f64)>,
    /// The padded waveform from sample `base` on (the first 160 samples are the zero padding).
    samples: Vec<f32>,
    base: usize,
    /// Every frame computed so far, MELS values each.
    frames: Vec<f32>,
}

impl UsmMel {
    pub fn new() -> UsmMel {
        let mel = autopro::audio::mel::MelSpectrogram {
            sample_rate: SAMPLE_RATE,
            n_fft: FFT,
            hop_length: HOP,
            win_length: FRAME,
            n_mels: MELS,
            f_min: 0.0,
            f_max: 8000.0,
            power: 1.0,
            center: false,
            mel_scale: autopro::audio::mel::MelScale::Htk,
            norm: autopro::audio::mel::MelNorm::None,
            mel_floor: MEL_FLOOR,
            log: autopro::audio::mel::LogMode::None,
        };
        let bank = mel.filter_bank();
        let mut filters = vec![0.0; BINS * MELS];
        for b in 0..BINS {
            for m in 0..MELS {
                filters[b * MELS + m] = bank[[b, m]];
            }
        }
        UsmMel {
            window: (0..FRAME).map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / FRAME as f64).cos()).collect(),
            filters,
            twiddles: (0..FFT / 2).map(|k| {
                let a = -2.0 * std::f64::consts::PI * k as f64 / FFT as f64;
                (a.cos(), a.sin())
            }).collect(),
            samples: vec![0.0; HOP],
            base: 0,
            frames: Vec::new(),
        }
    }

    pub fn frames(&self) -> usize {
        self.frames.len() / MELS
    }

    /// Appends samples and computes the frames they complete.
    pub fn push(&mut self, samples: &[f32]) {
        self.samples.extend_from_slice(samples);
        loop {
            let i = self.frames();
            let start = HOP * i - self.base;
            if self.samples.len() < start + FRAME + 1 {
                break;
            }
            let frame = self.frame(&self.samples[start..start + FRAME]);
            self.frames.extend(frame);
        }
        // Forget samples no later frame needs.
        let keep_from = HOP * self.frames();
        if keep_from > self.base {
            self.samples.drain(..keep_from - self.base);
            self.base = keep_from;
        }
    }

    /// Frames `from .. from + n` as an `[n, 128]` array; frames that do not exist (before the start, past the
    /// end) are zero, like the extractor's padding.
    pub fn window(&self, from: i64, n: usize) -> Array2<f32> {
        let mut out = Array2::<f32>::zeros((n, MELS));
        for r in 0..n {
            let i = from + r as i64;
            if i >= 0 && (i as usize) < self.frames() {
                let src = &self.frames[i as usize * MELS..(i as usize + 1) * MELS];
                out.row_mut(r).iter_mut().zip(src).for_each(|(o, &v)| *o = v);
            }
        }
        out
    }

    fn frame(&self, samples: &[f32]) -> Vec<f32> {
        // Windowed frame, zero padded to the FFT size, as a complex sequence.
        let mut re = vec![0.0f64; FFT];
        let mut im = vec![0.0f64; FFT];
        for (n, &s) in samples.iter().enumerate() {
            // the extractor multiplies float32 values
            re[n] = (s * self.window[n] as f32) as f64;
        }
        self.fft(&mut re, &mut im);
        let magnitude: Vec<f64> = (0..BINS).map(|b| (re[b] * re[b] + im[b] * im[b]).sqrt()).collect();
        (0..MELS)
            .map(|m| {
                let mut acc = 0.0;
                for (b, mag) in magnitude.iter().enumerate() {
                    acc += mag * self.filters[b * MELS + m];
                }
                (acc + MEL_FLOOR).ln() as f32
            })
            .collect()
    }

    /// In-place radix-2 FFT of length 512.
    fn fft(&self, re: &mut [f64], im: &mut [f64]) {
        let n = FFT;
        let bits = n.trailing_zeros();
        for i in 0..n {
            let j = (i as u32).reverse_bits() >> (32 - bits);
            if j as usize > i {
                re.swap(i, j as usize);
                im.swap(i, j as usize);
            }
        }
        let mut len = 2;
        while len <= n {
            let step = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..len / 2 {
                    let (wr, wi) = self.twiddles[k * step];
                    let (a, b) = (start + k, start + k + len / 2);
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len <<= 1;
        }
    }
}

impl Default for UsmMel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A sine's mel frames are the same whether it arrives in one piece or in odd-sized chunks.
    #[test]
    fn chunking_does_not_change_the_frames() {
        let samples: Vec<f32> = (0..16000).map(|i| (i as f32 * 0.07).sin() * 0.3 + (i as f32 * 0.013).cos() * 0.1).collect();
        let mut whole = UsmMel::new();
        whole.push(&samples);
        let mut parts = UsmMel::new();
        for chunk in samples.chunks(777) {
            parts.push(chunk);
        }
        assert_eq!(whole.frames(), (16000 + 160 - 321) / 160 + 1);
        assert_eq!(whole.frames(), parts.frames());
        assert_eq!(whole.window(0, whole.frames()), parts.window(0, parts.frames()));
        // Outside the audio the window is zero.
        let w = whole.window(-2, 3);
        assert!(w.row(0).iter().chain(w.row(1).iter()).all(|&v| v == 0.0));
        assert!(w.row(2).iter().any(|&v| v != 0.0));
    }
}
