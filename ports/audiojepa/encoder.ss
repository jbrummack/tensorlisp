;; Audio-JEPA encoder (ltuncay/Audio-JEPA, ICME 2025): a ViT-B with a 16 x 16 patch
;; embedding over a log-mel spectrogram, fixed 2D sin-cos positions and no class token.
;; Ported from the upstream VisionTransformer (flash_attn's MHA is plain attention
;; with the checkpoint's qkv/proj parameters).
;;
;; Raw input:            clip, audio of any rate (resampled to 32 kHz, mono) -> mel
;; Input (numpy order):  mel [batch, 256, 128]  Kaldi log-fbank of a 10 s clip at
;;                       32 kHz (256 frames, 128 mel bins); shorter clips are zero padded
;; Output:               embedding [batch, 128, 768]  8 frequency x 16 time patches,
;;                       time-major (patch p is time p / 8, frequency p % 8), after the final norm
;; Taps:                 embed, block.00 .. block.11, norm   [batch, 128, 768]
;;
;; Shapes in comments are ggml order (innermost first): D = 768, N = 128 patches, B = batch.

(import (tl tensor) (tl nn) (tl attn))

(define heads 12)
(define patch 16)
(define layers 12)
(define ln-eps 1e-5)   ; nn.LayerNorm default
;; Fused attention kernel (K/V in f16): faster on GPUs, slightly less exact.
(define flash-attention #f)

;; --- Preprocessing: inference_example.py's compute_mel_spec. Kaldi fbank of the waveform minus
;; its mean, 2.5 x the hop (10 s / 256) per frame, 128 mels from 20 Hz to Nyquist, Hanning window,
;; zero padded / truncated to 256 frames.

(preprocess ([clip audio])
  (model-inputs
    [mel (kaldi-fbank (if (= (audio-rate clip) 32000) clip (audio-resample clip 32000))
                      'mels 128 'frame-length 97.65625 'frame-shift 39.0625 'low 20 'window 'hanning
                      'frames 256 'subtract-mean #t)]))

(define (block x i)
  (let* ([p (lambda (name) (format "blocks.~a.~a" i name))]
         [x (ggml-add x (attn:multi-head (nn:layer-norm x (p "norm1") 'eps ln-eps) (p "attn") heads
                                         'names 'timm 'flash flash-attention))])
    (ggml-add x (nn:linear (nn:gelu (nn:linear (nn:layer-norm x (p "norm2") 'eps ln-eps) (p "mlp.fc1")))
                           (p "mlp.fc2")))))

(model (inputs [mel f32 (128 256 batch)])
  (define batch (tensor:dim mel 2))
  (define x0
    (tap "embed" (ggml-add (nn:patch-embed (ggml-reshape-4d mel 128 256 1 batch) "patch_embed.proj" patch) (weight "pos_embed")) 3))
  (define x
    (let loop ([i 0] [x x0])
      (if (= i layers)
          x
          (loop (+ i 1) (tap (format "block.~2,'0d" i) (block x i) 3)))))
  (outputs [embedding (tap "norm" (nn:layer-norm x "norm" 'eps ln-eps) 3) 3]))
