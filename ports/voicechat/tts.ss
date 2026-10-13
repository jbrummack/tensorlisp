;; NemotronLabs VoiceChat's speech output, ported from NeMo (speechlm2: ear_tts_model.py, ear_tts_vae_codec.py,
;; duplex_ear_tts.py). Append to backbone.ss and audio.ss (it uses their `dim`, `tok`, `eps`).
;;
;; The text-to-speech decoder (EAR-TTS): per 80 ms frame, one text token (pad, bos, eos or a subword) and the previous
;; frame's 31 residual codes give the next frame's codes. Entries (all with a guidance pair: branch 0 conditioned on the
;; text, branch 1 on a learned null embedding):
;;
;;   tts-cond      token [n] i32, mask [n, 1] -> emb [n, 1152]       the character-aware subword embedding of each token
;;   tts-inputs    ids [n, 31] i32, valid [n, 31], use_latent [n, 1], bos [n, 1], cond [n, 1152] -> x [2, n, 1152]
;;   tts-backbone  x [2, n, 1152], pos [n] i32 -> hidden [2, 1152]   28 Gemma3 layers over a KV cache (last position's output)
;;   tts-mog       ids [31] i32, valid [31], hidden [2, 1152] -> logits [1024], x [1152]      mixture weights of one sampling iteration
;;   tts-mog-z x [1152], rows64 [64] i32, rows512 [512] i32, noise [1, 512] -> z [512];  tts-quant-0..3 z, active [31] -> the codes of levels 0-7, 8-15, 16-23, 24-30
;;
;; The audio codec's decoder (the last stage: 31 residual codes per 80 ms frame -> 22.05 kHz audio):
;;
;;   codec-decode  codes [T, 31] i32, idft [16, 18], window [16] -> wav [T * 1764]
;;
;; prvq.decode sums the 31 codebook rows into a 512-d latent; Latent2Wav upsamples it 9x, 7x, 7x with ConvTranspose1d
;; (kernel = stride, so a matmul and a reshape), each followed by three ConvNeXt blocks (causal depthwise conv of
;; kernel 7, LayerNorm over channels, 1x1 conv up to 4x, GELU, 1x1 conv down, residual), then a 1x1 conv to
;; 9 magnitudes + 9 phases of a 16-point STFT at hop 4. The inverse STFT is a matrix product with `idft` (the
;; irfft: row n holds the coefficients of Re_0..8 and Im_0..8) and an overlap-add of shifted adds.

(define codec-rates '(9 7 7))
(define codec-blocks 3)
(define codec-kernel 7)
(define codec-n-fft 16)
(define codec-hop 4)
(define codec-quantizers 31)

(define (codec-name i . rest) (apply string-append (format "codec.decoder.layers.~a" i) rest))
(define (codec-w name) (weight name))

;; x [Cin, T] -> [Cout, T * s]: ConvTranspose1d with kernel = stride = s (weight [Cin, Cout, s] in torch).
(define (codec-conv-transpose x i s)
  (let* ([w (tensor:as-f32 (codec-w (codec-name i ".weight")))]                  ; ggml [s, Cout, Cin]
         [cin (dim w 2)] [cout (dim w 1)] [t (dim x 1)]
         [taps (ggml-cont (ggml-transpose (ggml-reshape-2d w (* cout s) cin)))]  ; [Cin, Cout * s]
         [y (ggml-reshape-3d (ggml-mul-mat taps x) s cout t)])                   ; (j, o, t)
    (ggml-reshape-2d (ggml-cont (ggml-permute y 1 0 2 3)) cout (* s t))))

(define (codec-pointwise x name n-out)
  (ggml-add (ggml-mul-mat (ggml-reshape-2d (codec-w (string-append name ".weight")) (dim x 0) n-out) x)
            (codec-w (string-append name ".bias"))))

;; x [C, T]
(define (codec-convnext x i)
  (let* ([c (dim x 0)] [t (dim x 1)]
         [history (ggml-pad-ext (ggml-cont (ggml-transpose x)) (- codec-kernel 1) 0 0 0 0 0 0 0)]   ; [T + 6, C]
         [kernel (ggml-cast (ggml-reshape-2d (codec-w (codec-name i ".dwconv.weight")) codec-kernel c) GGML_TYPE_F32)]
         [y (ggml-reshape-2d (ggml-ssm-conv (ggml-reshape-3d history (+ t (- codec-kernel 1)) c 1) kernel) c t)]
         [y (ggml-add y (codec-w (codec-name i ".dwconv.bias")))]
         [y (nn:layer-norm y (codec-name i ".norm") 'eps 1e-6)]
         [y (ggml-gelu-erf (codec-pointwise y (codec-name i ".pwconv1") (* 4 c)))]
         [y (codec-pointwise y (codec-name i ".pwconv2") c)])
    (ggml-add x y)))

;; codes [31, T] i32 -> the summed codebook rows [512, T]
(define (codec-dequantize codes)
  (let loop ([i 0] [z #f])
    (if (= i codec-quantizers)
        z
        (let* ([ids (ggml-reshape-1d (ggml-cont (tensor:slice codes 0 i 1)) (dim codes 1))]
               [rows (ggml-get-rows (tensor:as-f32 (weight (format "codec.prvq.mus_list.~a" i))) ids)])
          (loop (+ i 1) (if z (ggml-add z rows) rows))))))

;; Overlap-add of frames [16, T] at hop 4 -> [4 * (T + 3)] samples.
(define (codec-overlap-add frames)
  (let* ([t (dim frames 1)]
         [block (lambda (j)
                  (ggml-pad-ext (ggml-cont (ggml-view-2d frames 4 t (* 4 codec-n-fft) (* 4 4 j))) 0 0 j (- 3 j) 0 0 0 0))])
    (ggml-reshape-1d (fold-left ggml-add (block 0) (map block '(1 2 3))) (* 4 (+ t 3)))))

;; spec [18, T] (magnitude and phase logits) -> the waveform [4 * T]
(define (codec-istft spec idft window)
  (let* ([t (dim spec 1)]
         [mag (ggml-cont (tensor:slice spec 0 0 9))]
         [ph (ggml-cont (tensor:slice spec 0 9 9))]
         ;; mag = 100 * exp(-softplus(-mag + ln 100)), bounded by 100
         [mag (ggml-scale (ggml-exp (ggml-neg (ggml-softplus (ggml-scale-bias mag -1.0 (log 100.0))))) 100.0)]
         [parts (ggml-concat (ggml-mul mag (ggml-cos ph)) (ggml-mul mag (ggml-sin ph)) 0)]       ; Re, Im: [18, T]
         [frames (ggml-mul-mat idft parts)]                                                      ; [16, T]
         [w (ggml-reshape-2d window codec-n-fft 1)]
         ;; constrain_value_range: |frame| <= window
         [frames (ggml-add (ggml-sub frames (ggml-relu (ggml-sub frames w))) (ggml-relu (ggml-sub (ggml-neg frames) w)))]
         [windowed (ggml-mul frames w)]
         [envelope (codec-overlap-add (ggml-repeat (ggml-mul w w) frames))]
         [wav (ggml-div (codec-overlap-add windowed) envelope)]
         [pad (quotient (- codec-n-fft codec-hop) 2)])
    (ggml-view-1d wav (* codec-hop t) (* 4 pad))))

(model codec-decode (inputs [codes i32 (31 _)] [idft f32 (18 16)] [window f32 (16)])
  (define z (tap "codec.z" (codec-dequantize codes) 2))
  (define x
    (let stage ([s 0] [layer 0] [x z])
      (if (= s 3)
          x
          (let block ([b 0] [x (tap (format "codec.up~a" s) (codec-conv-transpose x layer (list-ref codec-rates s)) 2)])
            (if (= b codec-blocks)
                (stage (+ s 1) (+ layer 1 codec-blocks) (tap (format "codec.stage~a" s) x 2))
                (block (+ b 1) (codec-convnext x (+ layer 1 b))))))))
  (define spec (tap "codec.spec" (ggml-mul-mat (ggml-reshape-2d (codec-w "codec.decoder.layers.12.weight") 384 18) x) 2))
  (outputs [wav (codec-istft spec idft window) 1]))
;; --- the same decoder one token at a time, for streaming: the depthwise convs keep their last six inputs, the
;; overlap-add its last 12 samples (and the window envelope's), nothing needs a look-ahead. Token t completes the
;; waveform samples [1764 t - 6, 1764 t + 1758) (the 6 at each end are the trimmed padding), so the stream is
;; 6 samples behind the codes and one more call-free `edge` gives the last 6 at the end.

(define codec-samples 1764)       ; one token: 441 STFT frames of hop 4
(define codec-channels '(1536 768 384))
(for-each (lambda (s)
            (for-each (lambda (b) (define-state (format "cd.~a" (+ (* 4 s) 1 b)) f32 (- codec-kernel 1) (list-ref codec-channels s)))
                      (iota codec-blocks)))
          '(0 1 2))
(define-state "cd.tail" f32 12)
(define-state "cd.etail" f32 12)

(define (codec-convnext-step x i)
  (let* ([c (dim x 0)] [n (dim x 1)]
         [hp (ggml-concat (state (format "cd.~a" i)) (ggml-cont (ggml-transpose x)) 0)]                  ; [6 + n, C]
         [_ (effect (ggml-cpy (ggml-view-2d hp (- codec-kernel 1) c (* 4 (+ n (- codec-kernel 1))) (* 4 n)) (state (format "cd.~a" i))))]
         [kernel (ggml-cast (ggml-reshape-2d (codec-w (codec-name i ".dwconv.weight")) codec-kernel c) GGML_TYPE_F32)]
         [y (ggml-reshape-2d (ggml-ssm-conv (ggml-reshape-3d hp (+ n (- codec-kernel 1)) c 1) kernel) c n)]
         [y (ggml-add y (codec-w (codec-name i ".dwconv.bias")))]
         [y (nn:layer-norm y (codec-name i ".norm") 'eps 1e-6)]
         [y (ggml-gelu-erf (codec-pointwise y (codec-name i ".pwconv1") (* 4 c)))]
         [y (codec-pointwise y (codec-name i ".pwconv2") c)])
    (ggml-add x y)))

(model codec-step (inputs [codes i32 (31 1)] [idft f32 (18 16)] [window f32 (16)])
  (define z (codec-dequantize codes))
  (define x
    (let stage ([s 0] [layer 0] [x z])
      (if (= s 3)
          x
          (let block ([b 0] [x (codec-conv-transpose x layer (list-ref codec-rates s))])
            (if (= b codec-blocks)
                (stage (+ s 1) (+ layer 1 codec-blocks) x)
                (block (+ b 1) (codec-convnext-step x (+ layer 1 b))))))))
  (define spec (ggml-mul-mat (ggml-reshape-2d (codec-w "codec.decoder.layers.12.weight") 384 18) x))      ; [18, 441]
  (define t (dim spec 1))
  (define mag (ggml-cont (tensor:slice spec 0 0 9)))
  (define ph (ggml-cont (tensor:slice spec 0 9 9)))
  (define mag2 (ggml-scale (ggml-exp (ggml-neg (ggml-softplus (ggml-scale-bias mag -1.0 (log 100.0))))) 100.0))
  (define frames0 (ggml-mul-mat idft (ggml-concat (ggml-mul mag2 (ggml-cos ph)) (ggml-mul mag2 (ggml-sin ph)) 0)))
  (define w (ggml-reshape-2d window codec-n-fft 1))
  (define frames (ggml-add (ggml-sub frames0 (ggml-relu (ggml-sub frames0 w))) (ggml-relu (ggml-sub (ggml-neg frames0) w))))
  (define ola (codec-overlap-add (ggml-mul frames w)))                                                  ; [1776]
  (define env (codec-overlap-add (ggml-repeat (ggml-mul w w) frames)))
  (define total (ggml-add ola (ggml-pad-ext (state "cd.tail") 0 codec-samples 0 0 0 0 0 0)))
  (define etotal (ggml-add env (ggml-pad-ext (state "cd.etail") 0 codec-samples 0 0 0 0 0 0)))
  (define wav (ggml-div total etotal))
  (effect (ggml-cpy (ggml-cont (ggml-view-1d ola 12 (* 4 codec-samples))) (state "cd.tail")))
  (effect (ggml-cpy (ggml-cont (ggml-view-1d env 12 (* 4 codec-samples))) (state "cd.etail")))
  (outputs [wav (ggml-cont (ggml-view-1d wav codec-samples 0)) 1]
           [edge (ggml-div (ggml-cont (ggml-view-1d ola 6 (* 4 codec-samples))) (ggml-cont (ggml-view-1d env 6 (* 4 codec-samples)))) 1]))

;; ---------------------------------------------------------------------------------------------------------------
;; EAR-TTS

(define tts-d 1152)
(define tts-heads 16)
(define tts-hd 72)
(define tts-layers 28)
(define tts-cap 1024)             ; frames the KV cache holds (37 warm-up frames + the conversation)
(define tts-codebook 1024)
(define tts-latent 512)
(define tts-prompt-frames 37)

;; Gemma-style RMSNorm, (1 + weight).
(define (tnorm x name) (nn:rms-norm x name 'eps 1e-6 'offset 1))
(define (tts-gated-mlp x p)
  (nn:linear (ggml-mul (nn:gelu-tanh (nn:linear x (string-append p ".gate_proj"))) (nn:linear x (string-append p ".up_proj")))
             (string-append p ".down_proj")))

(for-each (lambda (l)
            (define-state (format "tk.~a" l) f16 tts-d tts-cap 2)
            (define-state (format "tv.~a" l) f16 tts-d tts-cap 2))
          (iota tts-layers))

;; --- the character-aware subword encoder: a subword is its characters through one T5Gemma encoder layer,
;; mean-pooled and projected, plus a continuation flag and a bos/eos flag embedding.

(define cas-max 76)

(model tts-cond (inputs [token i32 (_)] [mask f32 (1 _)])
  (define n (dim token 0))
  (define chars (ggml-get-rows (weight "tts.sub.chars") token))                                  ; [76, n] i32
  (define lenf (ggml-cast (ggml-get-rows (ggml-reshape-2d (weight "tts.sub.charlen") 1 131072) token) GGML_TYPE_F32))  ; [1, n]
  (define like (ggml-cast chars GGML_TYPE_F32))
  (define char-pos (ggml-repeat (ggml-reshape-2d (ggml-arange 0.0 (inexact cas-max) 1.0) cas-max 1) like))
  (define valid (ggml-step (ggml-scale-bias (ggml-sub (ggml-repeat lenf like) char-pos) 1.0 -0.5)))      ; [76, n]: char < length
  (define emb (ggml-scale (ggml-reshape-4d (ggml-get-rows (weight "tts.sub.embed_tokens.weight") (ggml-reshape-1d chars (* cas-max n)))
                                           tts-d cas-max 1 n)
                          (sqrt (inexact tts-d))))
  (define p "tts.sub.enc.layers.0")
  (define (project h name) (ggml-reshape-4d (nn:linear h (string-append p ".self_attn." name)) tts-hd tts-heads cas-max n))
  (define h (tnorm emb (string-append p ".pre_self_attn_layernorm")))
  (define rope-pos (ggml-cast (ggml-arange 0.0 (inexact cas-max) 1.0) GGML_TYPE_I32))
  (define q (attn:rope (project h "q_proj") rope-pos 'base 10000.0))
  (define k (attn:rope (project h "k_proj") rope-pos 'base 10000.0))
  (define v (project h "v_proj"))
  (define att (attn:sdpa (ggml-permute q 0 2 1 3) (ggml-permute k 0 2 1 3) (ggml-permute v 0 2 1 3)
                         'mask (attn:padding-mask valid cas-max) 'scale (/ 1.0 16.0)))                          ; [1152, 76, n]
  (define x1 (ggml-add emb (tnorm (nn:linear (ggml-reshape-4d att tts-d cas-max 1 n) (string-append p ".self_attn.o_proj"))
                                  (string-append p ".post_self_attn_layernorm"))))
  (define x2 (ggml-add x1 (tnorm (tts-gated-mlp (tnorm x1 (string-append p ".pre_feedforward_layernorm")) (string-append p ".mlp"))
                                 (string-append p ".post_feedforward_layernorm"))))
  (define enc (tnorm x2 "tts.sub.enc.norm"))
  ;; mean over the real characters
  (define summed (ggml-sum-rows (ggml-mul (ggml-cont (ggml-permute enc 1 0 2 3)) (ggml-reshape-4d valid cas-max 1 1 n))))   ; [1, 1152, 1, n]
  (define pooled (ggml-reshape-2d (ggml-div summed (ggml-reshape-4d (ggml-clamp lenf 1.0 1e9) 1 1 1 n)) tts-d n))
  (define base (ggml-mul (nn:linear pooled "tts.sub.proj_embedding") (ggml-repeat mask pooled)))
  (define (flag table-name emb-name count)
    (ggml-get-rows (weight emb-name)
                   (ggml-reshape-1d (ggml-get-rows (ggml-reshape-2d (weight table-name) 1 count) token) n)))
  (define out (ggml-add (ggml-add base (flag "tts.sub.subword_flag_emb.is_continuation" "tts.sub.subword_flag_emb.cont_emb.weight" 131073))
                        (flag "tts.sub.bos_eos_emb.special_flags" "tts.sub.bos_eos_emb.special_emb.weight" 131072)))
  (outputs [emb out 2]))

;; --- fusion of the previous frame's codes with the text conditioning

;; The summed codebook rows of ids [31, n] (rows of rvq_embs flattened to [512, 31 * 1024]) where valid, [512, n].
(define (tts-depthsum ids valid)
  (let* ([n (dim ids 1)]
         [table (ggml-reshape-2d (weight "tts.rvq_embs") tts-latent (* 31 tts-codebook))]
         [rows (ggml-mul (ggml-get-rows table (ggml-reshape-1d ids (* 31 n))) (ggml-reshape-2d valid 1 (* 31 n)))]   ; [512, 31 n]
         [by-position (ggml-cont (ggml-permute (ggml-reshape-3d rows tts-latent 31 n) 1 0 2 3))])                    ; [31, 512, n]
    (ggml-reshape-2d (ggml-sum-rows by-position) tts-latent n)))

(define (tts-fuse code-embed cond)
  (let* ([p "tts.gated_fusion_audio_text"]
         [audio (nn:linear (ggml-scale code-embed (/ 1.0 31.0)) (string-append p ".audio_proj"))]
         [text (nn:linear cond (string-append p ".text_proj"))]
         [gate (ggml-repeat (ggml-reshape-2d (ggml-sigmoid (weight (string-append p ".gate"))) tts-d 1) audio)]
         [res (ggml-sigmoid (weight (string-append p ".residual_scale")))]
         [mixed (ggml-add (ggml-mul gate audio) (ggml-mul (ggml-scale-bias gate -1.0 1.0) text))])
    (tnorm (ggml-mul mixed (ggml-repeat res mixed)) (string-append p ".final_norm"))))

(model tts-inputs (inputs [ids i32 (31 _)] [valid f32 (31 _)] [use-latent f32 (1 _)] [bos f32 (1 _)] [cond f32 (1152 _)])
  (define n (dim ids 1))
  (define code-embed (nn:linear (tts-depthsum ids valid) "tts.embed_code"))
  ;; the warm-up (37 frames) replaces the frames before the bos with the speaker's pre-computed prompt latent
  (define mixed (if (= n tts-prompt-frames)
                    (let ([use (ggml-repeat use-latent code-embed)])
                      (ggml-add (ggml-mul use (ggml-reshape-2d (weight "tts_audio_prompt_latents.Aria") tts-d n))
                                (ggml-mul (ggml-scale-bias use -1.0 1.0) code-embed)))
                    code-embed))
  (define with-bos (ggml-add mixed (ggml-mul (ggml-repeat bos mixed) (ggml-repeat (ggml-reshape-2d (weight "tts.bos_emb") tts-d 1) mixed))))
  (define conditioned (tts-fuse with-bos cond))
  (define unconditioned (tts-fuse with-bos (ggml-repeat (ggml-reshape-2d (weight "tts.null_emb") tts-d 1) with-bos)))
  (outputs [x (ggml-concat (ggml-reshape-3d conditioned tts-d n 1) (ggml-reshape-3d unconditioned tts-d n 1) 2) 3]))

;; --- the Gemma3 decoder (28 layers, 16 heads of 72, sliding layers rope 1e4, every 6th layer global rope 1e6)

(define (tts-mask pos)
  (let* ([n (dim pos 0)]
         [q (ggml-repeat-4d (ggml-reshape-2d (ggml-cast pos GGML_TYPE_F32) 1 n) tts-cap n 1 1)]
         [dist (ggml-sub q (ggml-reshape-2d (ggml-arange 0.0 (inexact tts-cap) 1.0) tts-cap 1))])
    (ggml-scale-bias (ggml-step (ggml-scale-bias dist 1.0 1.0)) (- attn:blocked) attn:blocked)))

(define (tts-layer x l pos mask)                       ; x [1152, n, 2]
  (let* ([p (format "tts.backbone.layers.~a" l)]
         [n (dim x 1)]
         [h (tnorm x (string-append p ".input_layernorm"))]
         [heads (lambda (name) (ggml-reshape-4d (nn:linear h (string-append p ".self_attn." name)) tts-hd tts-heads n 2))]
         [base (if (= (remainder (+ l 1) 6) 0) 1000000.0 10000.0)]
         [q (attn:rope (tnorm (heads "q_proj") (string-append p ".self_attn.q_norm")) pos 'base base)]
         [k (attn:rope (tnorm (heads "k_proj") (string-append p ".self_attn.k_norm")) pos 'base base)]
         [v (heads "v_proj")]
         [cache (lambda (kind) (state (format "~a.~a" kind l)))]
         [view (lambda (c) (ggml-view-4d c tts-hd tts-cap tts-heads 2 (* 2 tts-d) (* 2 tts-hd) (* 2 tts-d tts-cap) 0))])
    (effect (ggml-set-rows (cache "tk") (ggml-reshape-3d k tts-d n 2) pos))
    (effect (ggml-set-rows (cache "tv") (ggml-reshape-3d v tts-d n 2) pos))
    (let* ([att (attn:sdpa (ggml-permute q 0 2 1 3) (view (cache "tk")) (view (cache "tv")) 'mask mask 'scale (/ 1.0 16.0) 'flash #t)]   ; [1152, n, 2]
           [x (ggml-add x (tnorm (nn:linear att (string-append p ".self_attn.o_proj")) (string-append p ".post_attention_layernorm")))]
           [ff (tts-gated-mlp (tnorm x (string-append p ".pre_feedforward_layernorm")) (string-append p ".mlp"))])
      (ggml-add x (tnorm ff (string-append p ".post_feedforward_layernorm"))))))

(model tts-backbone (inputs [x f32 (1152 _ 2)] [pos i32 (_)])
  (define n (dim x 1))
  (define mask (tts-mask pos))
  (define out (let loop ([l 0] [x x]) (if (= l tts-layers) x (loop (+ l 1) (tts-layer x l pos mask)))))
  (outputs [hidden (ggml-reshape-2d (tnorm (tensor:slice out 1 (- n 1) 1) "tts.backbone.norm") tts-d 2) 2]))

;; --- the mixture-of-Gaussians head: per sampling iteration the mixture logits, then the chosen component's mean

(define mog-guidance 0.2)                     ; inference_guidance_scale
(define mog-noise 0.001)                      ; inference_noise_scale
(define mog-components 1024)
(define mog-rank 64)

(define (mog-layer x l)
  (let* ([p (format "tts.mog_head.mlp_stack.~a" l)]
         [y (tts-gated-mlp (tnorm x (string-append p ".pre_norm")) (string-append p ".mlp"))])
    (ggml-add x (tnorm y (string-append p ".post_norm")))))

(model tts-mog (inputs [ids i32 (31)] [valid f32 (31)] [hidden f32 (1152 2)])
  (define code-embed (nn:linear (tts-depthsum (ggml-reshape-2d ids 31 1) (ggml-reshape-2d valid 31 1)) "tts.embed_code"))   ; [1152, 1]
  (define stack (tnorm (let loop ([l 0] [x (ggml-add hidden (ggml-repeat code-embed hidden))])
                         (if (= l 3) x (loop (+ l 1) (mog-layer x l))))
                       "tts.mog_head.mlp_stack.3"))
  (define cond (tensor:slice stack 1 0 1))
  (define uncond (tensor:slice stack 1 1 1))
  (define x (ggml-add cond (ggml-scale (ggml-sub cond uncond) mog-guidance)))                            ; classifier-free guidance
  (outputs [logits (ggml-mul-mat (weight "tts.mog_head.proj_logits.weight") x) 1] [x x 1]))

;; z: the chosen component's mean plus noise, [512, 1]
(model tts-mog-z (inputs [x f32 (1152 1)] [rows64 i32 (64)] [rows512 i32 (512)] [noise f32 (512 1)])
  (define mu-low (ggml-mul-mat (ggml-get-rows (weight "tts.mog_head.proj_mus.weight") rows64) x))                      ; [64, 1]
  (define low (ggml-get-rows (ggml-reshape-2d (weight "tts.mog_head.low_mat") mog-rank (* 512 mog-components)) rows512))   ; [64, 512]
  (define mu (ggml-mul-mat low mu-low))                                                                            ; [512, 1]
  (define mu-res (ggml-mul-mat (weight "tts.mog_head.proj_else.weight") x))
  (define logs (ggml-clamp (ggml-mul-mat (weight "tts.mog_head.proj_logs.weight") x) -4.0 1e9))                       ; [1, 1]
  (define std (ggml-exp logs))
  (outputs [z (ggml-add (ggml-add (ggml-mul mu (ggml-repeat std mu)) mu-res)
                        (ggml-mul (ggml-scale noise mog-noise) (ggml-repeat std noise))) 1]))

;; Residual quantization of z over the levels lo .. hi - 1, only on the levels this iteration fills (`active`, 31).
;; Split in four graphs so an iteration runs only the ones with active levels; the levels before the first active one
;; subtract nothing, so each starts from z.
(define (join-codes picked) (fold-left (lambda (acc t) (ggml-concat acc t 0)) (car picked) (cdr picked)))
(define (quantize-levels z active lo hi)
  (let ([table (weight "tts.rvq_embs")])
    (let loop ([i lo] [r z] [picked '()])
      (if (= i hi)
          (cons (join-codes (reverse picked)) r)
          (let* ([book (ggml-view-2d table tts-latent tts-codebook (* 4 tts-latent) (* 4 tts-latent tts-codebook i))]
                 [score (ggml-sub (ggml-scale (ggml-mul-mat book r) 2.0)
                                  (ggml-reshape-2d (ggml-sum-rows (ggml-sqr book)) tts-codebook 1))]
                 [idx (ggml-argmax score)]
                 [chosen (ggml-get-rows book idx)])
            (loop (+ i 1)
                  (ggml-sub r (ggml-mul chosen (ggml-repeat (tensor:slice active 0 i 1) chosen)))
                  (cons (ggml-cast idx GGML_TYPE_F32) picked)))))))

(model tts-quant-0 (inputs [z f32 (512 1)] [active f32 (31)])
  (define q (quantize-levels z active 0 8))
  (outputs [codes (car q) 1] [residual (cdr q) 1]))
(model tts-quant-1 (inputs [z f32 (512 1)] [active f32 (31)])
  (define q (quantize-levels z active 8 16))
  (outputs [codes (car q) 1] [residual (cdr q) 1]))
(model tts-quant-2 (inputs [z f32 (512 1)] [active f32 (31)])
  (define q (quantize-levels z active 16 24))
  (outputs [codes (car q) 1] [residual (cdr q) 1]))
(model tts-quant-3 (inputs [z f32 (512 1)] [active f32 (31)])
  (define q (quantize-levels z active 24 31))
  (outputs [codes (car q) 1] [residual (cdr q) 1]))

(model tts-silence (inputs [dummy f32 (1)])
  (outputs [codes (ggml-cast (weight "tts_codec_silence_tokens") GGML_TYPE_F32) 1]))

;; ---------------------------------------------------------------------------------------------------------------
;; The driver (NeMo's NemotronVoiceChat.offline_inference TTS half, DuplexEARTTS.infer_codes_one_step, RVQEARTTSModel.generate_step)

(define tts-iterations 8)         ; num_iter
(define tts-top-p 0.95)           ; inference_top_p_or_k
(define tts-pad-code (util:make-list 31 tts-codebook))     ; "no code yet": the zero row of the codebook table

(define (tts-array-of values . dims) (apply array-reshape (list->array (map inexact values)) dims))
(define (tts-ids code) (map (lambda (c i) (+ (* i tts-codebook) (min c (- tts-codebook 1)))) code (iota 31)))
(define (tts-valid code) (map (lambda (c) (if (< c tts-codebook) 1.0 0.0)) code))

;; How many of the 31 codebook levels each sampling iteration settles: levels masked after iteration i are
;; ceil((1 - (i/8)^3)^(1/3) * 31); the difference to the next iteration's count are the levels this one fills.
(define (tts-level-counts)
  (let ([masked (map (lambda (i)
                       (let ([r (/ (inexact i) tts-iterations)])
                         (exact (ceiling (* (expt (- 1.0 (* r r r)) (/ 1.0 3.0)) 31.0)))))
                     (iota tts-iterations))])
    (map - masked (append (cdr masked) '(0)))))

;; Index drawn from the mixture logits (a list) after top-p filtering like transformers' TopPLogitsWarper:
;; the tokens whose cumulative probability, ascending, stays within 1 - top_p are dropped.
(define (tts-sample-component logits)
  (let* ([top (apply max logits)]
         [es (map (lambda (l) (exp (- l top))) logits)]
         [total (apply + es)]
         [ps (map (lambda (e) (/ e total)) es)]
         [mass-below (lambda (t) (fold-left (lambda (a p) (if (<= p t) (+ a p) a)) 0.0 ps))]
         [threshold (let bisect ([lo 0.0] [hi (apply max ps)] [i 0])
                      (if (= i 40)
                          lo
                          (let ([mid (/ (+ lo hi) 2.0)])
                            (if (<= (mass-below mid) (- 1.0 tts-top-p)) (bisect mid hi (+ i 1)) (bisect lo mid (+ i 1))))))]
         [kept (map (lambda (p) (if (> p threshold) p 0.0)) ps)]
         [u (* (car (array->list (random-uniform 1))) (apply + kept))])
    (let pick ([i 0] [acc 0.0] [rest kept])
      (let ([acc (+ acc (car rest))])
        (if (or (>= acc u) (null? (cdr rest))) i (pick (+ i 1) acc (cdr rest)))))))

(define (run-quant c z active)   ; z: the residual so far
  (case c
    [(0) (run tts-quant-0 [z z] [active active])]
    [(1) (run tts-quant-1 [z z] [active active])]
    [(2) (run tts-quant-2 [z z] [active active])]
    [else (run tts-quant-3 [z z] [active active])]))

(define (tts-silence-codes)
  (util:->integers (array->list (output (run tts-silence [dummy (list->array '(0.0))]) 'codes))))

;; Primes the decoder with the speaker's 37-frame prompt latent: frames 0-35 are the latent, frame 36 is the silence
;; code of frame 35 plus the bos embedding; the text conditioning is on for the last two frames (a pad and the eos).
(define (tts-warmup silence)
  (let* ([n tts-prompt-frames]
         [tokens (append (util:make-list (- n 1) 12) '(2))]
         [mask (append (util:make-list (- n 2) 0.0) '(1.0 1.0))]
         [conditioning (output (run tts-cond [token (list->array tokens)] [mask (tts-array-of mask n 1)]) 'emb)]
         [codes (append (util:make-list (- n 1) tts-pad-code) (list silence))]
         [x (output (run tts-inputs
                         [ids (tts-array-of (apply append (map tts-ids codes)) n 31)]
                         [valid (tts-array-of (apply append (map tts-valid codes)) n 31)]
                         [use-latent (tts-array-of (append (util:make-list (- n 1) 1.0) '(0.0)) n 1)]
                         [bos (tts-array-of (append (util:make-list (- n 1) 0.0) '(1.0)) n 1)]
                         [cond conditioning])
                   'x)])
    (run tts-backbone [x x] [pos (list->array (iota n))])
    #t))

;; One frame: the codes for text token `text-id`, given the previous frame's codes, at KV position `pos`.
(define (tts-frame text-id prev-code pos silence)
  (let* ([prev (if (= text-id 2) silence prev-code)]      ; inference_force_speech_silence_on_eos
         [conditioning (output (run tts-cond [token (list->array (list text-id))] [mask (tts-array-of '(1.0) 1 1)]) 'emb)]
         [x (output (run tts-inputs [ids (tts-array-of (tts-ids prev) 1 31)] [valid (tts-array-of (tts-valid prev) 1 31)]
                         [use-latent (array-zeros 1 1)] [bos (array-zeros 1 1)] [cond conditioning])
                    'x)]
         [hidden (output (run tts-backbone [x x] [pos (list->array (list pos))]) 'hidden)])
    (let loop ([ks (tts-level-counts)] [done 0] [code tts-pad-code])
      (cond
        [(null? ks) code]
        [(= (car ks) 0) (loop (cdr ks) done code)]
        [else
         (let* ([k (car ks)]
                [mog (run tts-mog [ids (tts-array-of (tts-ids code) 31)] [valid (tts-array-of (tts-valid code) 31)] [hidden hidden])]
                [component (tts-sample-component (array->list (output mog 'logits)))]
                [z (output (run tts-mog-z
                                [x (array-reshape (output mog 'x) 1 tts-d)]
                                [rows64 (list->array (map (lambda (j) (+ (* component mog-rank) j)) (iota mog-rank)))]
                                [rows512 (list->array (map (lambda (j) (+ (* component 512) j)) (iota 512)))]
                                [noise (array-reshape (random-normal 512) 1 512)])
                           'z)]
                [active (tts-array-of (map (lambda (i) (if (and (>= i done) (< i (+ done k))) 1.0 0.0)) (iota 31)) 31)]
                ;; fresh codes for the levels done .. done + k - 1 from the quantizer graphs that cover them
                [new (let chunks ([c 0] [r (array-reshape z 1 512)] [out '()])
                       (if (= c 4)
                           (apply append (reverse out))
                           (let ([lo (* 8 c)] [hi (min 31 (* 8 (+ c 1)))])
                             (if (and (< lo (+ done k)) (> hi done))
                                 (let ([q (run-quant c r active)])
                                   (chunks (+ c 1) (array-reshape (output q 'residual) 1 512)
                                           (cons (util:->integers (array->list (output q 'codes))) out)))
                                 (chunks (+ c 1) r (cons (util:make-list (- hi lo) 0) out))))))])
           (loop (cdr ks) (+ done k)
                 (map (lambda (old fresh i) (if (and (>= i done) (< i (+ done k))) fresh old)) code new (iota 31))))]))))

;; --- decoding the codes to audio, in chunks with a few tokens of context either side (the decoder looks back
;; about three tokens, and the inverse STFT one token ahead)

(define (codec-idft)
  (tts-array-of
   (apply append
          (map (lambda (n)
                 (append
                  (map (lambda (k) (* (if (or (= k 0) (= k 8)) 1.0 2.0) (/ 1.0 codec-n-fft) (cos (/ (* 2.0 3.141592653589793 k n) codec-n-fft)))) (iota 9))
                  (map (lambda (k) (- (* (if (or (= k 0) (= k 8)) 1.0 2.0) (/ 1.0 codec-n-fft) (sin (/ (* 2.0 3.141592653589793 k n) codec-n-fft))))) (iota 9))))
               (iota codec-n-fft)))
   codec-n-fft 18))

(define (codec-window)
  (list->array (map (lambda (n) (- 0.5 (* 0.5 (cos (/ (* 2.0 3.141592653589793 n) codec-n-fft))))) (iota codec-n-fft))))

(define codec-samples-per-frame 1764)

(define (tts-decode codes)
  (let ([total (length codes)] [idft (codec-idft)] [window (codec-window)] [chunk 32] [context 4])
    (let loop ([a 0] [audio #f])
      (if (>= a total)
          audio
          (let* ([b (min total (+ a chunk))]
                 [lo (max 0 (- a context))]
                 [hi (min total (+ b 1))]
                 [rows (list-head (list-tail codes lo) (- hi lo))]
                 [wav (output (run codec-decode [codes (tts-array-of (apply append rows) (- hi lo) 31)] [idft idft] [window window]) 'wav)]
                 [mine (array-slice wav (* (- a lo) codec-samples-per-frame) (* (- b lo) codec-samples-per-frame))])
            (loop b (if audio (array-concat audio mine) mine)))))))

;; The agent's speech for a text channel (token ids per 80 ms frame; frame 0 is not synthesized, like NeMo's loop).
(define (tts-speak-values ids)
  (random-seed! 1234)
  (let ([silence (tts-silence-codes)])
    (tts-warmup silence)
    (let loop ([t 1] [rest (cdr ids)] [prev tts-pad-code] [codes (list (util:make-list 31 0))])
      (if (null? rest)
          (let ([codes (reverse codes)])
            (values (tts-decode codes) (tts-array-of (apply append codes) (length codes) 31)))
          (let ([code (tts-frame (car rest) prev (+ tts-prompt-frames (- t 1)) silence)])
            (loop (+ t 1) (cdr rest) code (cons code codes)))))))

(define (tts-speak ids)
  (let-values ([(audio codes) (tts-speak-values ids)])
    (results [audio audio] [codes codes])))

;; A text channel for a sentence like the language model's: bos, the subwords on consecutive frames (the decoder paces
;; the speech itself), `hold` pad frames while it is spoken, then eos (which silences it) and `tail` pad frames.
(define (tts-text-channel text hold tail)
  (let-values ([(ids mask) (tokenize tok text 'special-tokens #f)])
    (append '(12 1)
            (util:->integers (array->list ids))
            (util:make-list hold 12)
            '(2)
            (util:make-list tail 12))))

(pipeline speak ([tokens array])
  (tts-speak (util:->integers (array->list tokens))))

(pipeline speak-text ([text string])
  (tts-speak (tts-text-channel text 40 6)))

;; The whole turn: the duplex language model listens to the user's audio and writes the agent's text channel, the
;; speech decoder speaks it (one 22.05 kHz stream, as long as the input: the agent's reply sits where it was said).
(pipeline converse-speak ([audio audio])
  (let* ([r (duplex-run audio)] [texts (car r)])
    (let-values ([(speech codes) (tts-speak-values texts)])
      (results [text (detokenize tok (filter (lambda (id) (not (memv id '(1 2 12)))) texts))] [tokens texts] [audio speech]))))
