//! NeMo's `AudioToMelSpectrogramPreprocessor` as the NemotronLabs VoiceChat speech encoder is configured
//! (`FilterbankFeatures`, inference mode), computed incrementally.
//!
//! 16 kHz mono audio; pre-emphasis `x[n] - 0.97 x[n-1]`; `torch.stft` with a 512-point FFT, hop 160 and a
//! non-periodic Hann window of 400 samples (zero padded to 512 on both sides), `center=True` with reflect
//! padding of 256 samples; power spectrum; 128 Slaney mel filters from 0 to 8 kHz (librosa's); `ln(mel + 2^-24)`;
//! no normalization (`normalize: NA`); the window and the filters are rounded to bfloat16 like the checkpoint's
//! buffers. There are `1 + samples / 160` frames.
//!
//! Frame `i` covers the padded samples `[160 i, 160 i + 512)`, i.e. pre-emphasized samples
//! `[160 i - 256, 160 i + 256)`, of which only the middle 400 (the window) count. Before the end of the audio is known a frame is produced once its samples
//! are there; [`NemoMel::finish`] adds the reflect padding at the end and produces the rest.
use ndarray::Array2;

const HOP: usize = 160;
const WIN: usize = 400;
const FFT: usize = 512;
const BINS: usize = FFT / 2 + 1;
const MELS: usize = 128;
const PREEMPH: f32 = 0.97;
const LOG_GUARD: f64 = 5.960464477539063e-8; // 2^-24

/// Rounds to bfloat16 (nearest even) and back. The checkpoint's `window` and mel filterbank buffers are the
/// analytic ones rounded this way (the model was saved from bf16; checked equal for every element).
fn bf16(x: f64) -> f64 {
    let bits = (x as f32).to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded) as f64
}

pub struct NemoMel {
    window: Vec<f64>,
    /// [BINS][MELS]
    filters: Vec<f64>,
    twiddles: Vec<(f64, f64)>,
    /// Pre-emphasized samples from index `base` on.
    samples: Vec<f32>,
    base: usize,
    /// Raw samples received so far (the next one is pre-emphasized against the last received).
    received: usize,
    last_raw: f32,
    finished: bool,
    frames: Vec<f32>,
}

impl NemoMel {
    pub fn new() -> NemoMel {
        let mel = autopro::audio::mel::MelSpectrogram {
            sample_rate: 16000,
            n_fft: FFT,
            hop_length: HOP,
            win_length: WIN,
            n_mels: MELS,
            f_min: 0.0,
            f_max: 8000.0,
            power: 2.0,
            center: true,
            mel_scale: autopro::audio::mel::MelScale::Slaney,
            norm: autopro::audio::mel::MelNorm::Slaney,
            mel_floor: 0.0,
            log: autopro::audio::mel::LogMode::None,
        };
        let bank = mel.filter_bank();
        let mut filters = vec![0.0; BINS * MELS];
        for b in 0..BINS {
            for m in 0..MELS {
                filters[b * MELS + m] = bf16(bank[[b, m]]);
            }
        }
        NemoMel {
            // torch.hann_window(400, periodic=False)
            window: (0..WIN).map(|n| bf16(0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (WIN - 1) as f64).cos())).collect(),
            filters,
            twiddles: (0..FFT / 2)
                .map(|k| {
                    let a = -2.0 * std::f64::consts::PI * k as f64 / FFT as f64;
                    (a.cos(), a.sin())
                })
                .collect(),
            samples: Vec::new(),
            base: 0,
            received: 0,
            last_raw: 0.0,
            finished: false,
            frames: Vec::new(),
        }
    }

    pub fn frames(&self) -> usize {
        self.frames.len() / MELS
    }

    /// Appends samples and computes the frames they complete.
    pub fn push(&mut self, samples: &[f32]) {
        assert!(!self.finished, "audio pushed after finish");
        for &s in samples {
            // the first sample is not pre-emphasized
            self.samples.push(if self.received == 0 { s } else { s - PREEMPH * self.last_raw });
            self.last_raw = s;
            self.received += 1;
        }
        self.compute_ready();
    }

    /// Marks the end of the audio: the remaining frames use the reflect padding at the end.
    pub fn finish(&mut self) {
        self.finished = true;
        self.compute_ready();
    }

    fn total_frames(&self) -> usize {
        1 + self.received / HOP
    }

    /// Pre-emphasized sample `i` of the reflect-padded signal, if it is known yet.
    fn at(&self, i: i64) -> Option<f32> {
        let n = self.received as i64;
        let j = if i < 0 { -i } else { i };
        let j = if j >= n {
            if !self.finished {
                return None;
            }
            2 * (n - 1) - j
        } else {
            j
        };
        if j < 0 || j >= n {
            return Some(0.0); // fewer samples than the padding; torch would raise
        }
        // A negative index reflects to a positive one, which may not have arrived yet.
        let j = j as usize;
        if j >= self.base + self.samples.len() { None } else { Some(self.samples[j - self.base]) }
    }

    fn compute_ready(&mut self) {
        loop {
            let i = self.frames();
            if self.received == 0 || (self.finished && i >= self.total_frames()) {
                break;
            }
            let start = (HOP * i) as i64 - (FFT / 2) as i64;
            let mut re = vec![0.0f64; FFT];
            let mut ok = true;
            for k in 0..FFT {
                // the 400-sample window sits in the middle of the 512 FFT
                let w = k as i64 - ((FFT - WIN) / 2) as i64;
                if w < 0 || w >= WIN as i64 {
                    continue;
                }
                match self.at(start + k as i64) {
                    Some(v) => re[k] = (v as f64) * self.window[w as usize],
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                break;
            }
            let frame = self.frame(re);
            self.frames.extend(frame);
        }
        // Forget samples no later frame needs (the reflect at the start needs the first 257 for frame 0 only).
        let keep_from = (HOP * self.frames()).saturating_sub(FFT / 2 + 1);
        if keep_from > self.base + 1 && self.frames() > 2 {
            let drop = (keep_from - self.base).min(self.samples.len());
            self.samples.drain(..drop);
            self.base += drop;
        }
    }

    /// Frames `from .. from + n` as an `[n, 128]` array; frames that do not exist are zero.
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

    fn frame(&self, mut re: Vec<f64>) -> Vec<f32> {
        let mut im = vec![0.0f64; FFT];
        self.fft(&mut re, &mut im);
        let power: Vec<f64> = (0..BINS).map(|b| re[b] * re[b] + im[b] * im[b]).collect();
        (0..MELS)
            .map(|m| {
                let mut acc = 0.0;
                for (b, p) in power.iter().enumerate() {
                    acc += p * self.filters[b * MELS + m];
                }
                (acc + LOG_GUARD).ln() as f32
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

impl Default for NemoMel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.07).sin() * 0.3 + (i as f32 * 0.013).cos() * 0.1).collect()
    }

    fn run(chunks: &[usize], samples: &[f32]) -> NemoMel {
        let mut mel = NemoMel::new();
        let mut at = 0;
        for &c in chunks.iter().cycle() {
            if at >= samples.len() {
                break;
            }
            let end = (at + c).min(samples.len());
            mel.push(&samples[at..end]);
            at = end;
        }
        mel.finish();
        mel
    }

    /// Frame count is 1 + n / 160 and chunking changes nothing.
    #[test]
    fn chunking_does_not_change_the_frames() {
        let samples = signal(16000 + 77);
        let whole = run(&[1 << 20], &samples);
        let parts = run(&[777, 13, 4000], &samples);
        assert_eq!(whole.frames(), 1 + samples.len() / HOP);
        assert_eq!(whole.window(0, whole.frames()), parts.window(0, parts.frames()));
    }

    /// Streaming emits a frame once its samples are there: the window ends 200 samples after the frame's centre
    /// 160 i, so frame i after 160 i + 200 samples; frame 0 also reflects x[200] into its left half, so it
    /// needs 201.
    #[test]
    fn frames_appear_when_their_samples_do() {
        let samples = signal(3000);
        let mut mel = NemoMel::new();
        mel.push(&samples[..200]);
        assert_eq!(mel.frames(), 0);
        mel.push(&samples[200..201]);
        assert_eq!(mel.frames(), 1);
        mel.push(&samples[201..160 * 5 + 200]);
        assert_eq!(mel.frames(), 6);
    }
}
