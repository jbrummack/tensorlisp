;; NemotronLabs VoiceChat's audio side, ported from NeMo (nemo/collections/speechlm2, asr/modules/conformer_encoder.py).
;; Append to backbone.ss for the whole model (this file uses its `dim`).
;;
;; Entries (shapes in numpy order):
;;   encode  mel [T, 128] -> emb [T', 1024] (the FastConformer's output, what the RNNT branch reads),
;;           proj [T', 4480] (the same through `perception.proj`: what the language model gets), T' = T/8
;; Pipelines (raw inputs):
;;   duplex  audio [1, 4480] (one 80 ms frame of `proj`), text [1], func [1] (the previous frame's text and function tokens),
;;           pos [1] -> next_text [1], next_func [1], logits [1, vocab]: one step of the full-duplex loop
;;   mel         audio (16 kHz) -> mel [frames, 128]     NeMo's preprocessor (src/nemo_mel.rs)
;;   converse    audio (16 kHz) -> text, tokens, function     runs the duplex loop over the audio (a conversation turn):
;;                                                         the agent's text channel, one token per 80 ms frame (pad = 12)
;;   transcribe  audio (16 kHz) -> text, tokens          greedy RNNT decoding of the encoder output with the model's own
;;                                                         ASR branch (stt_model.rnnt_decoder / rnnt_joint): a check that the
;;                                                         encoder works, and the user-text channel
;;
;; The encoder is a cache-aware streaming FastConformer (Nemotron-Speech-Streaming-En-0.6b): 8x causal
;; depthwise-strided subsampling (80 ms frames), 24 conformer layers (macaron feed-forwards, relative-position
;; self-attention over the last 71 frames, causal depthwise conv of kernel 9), no biases in the linears.
;; Run on a whole utterance it equals running it frame by frame with the caches.

;; ENC-CONFIG-BEGIN
(define enc-layers 24)
(define enc-d 1024)
(define enc-heads 8)
(define enc-left 70)        ; frames of left attention context (att_context_size [70, 0])
(define sub-channels 256)
;; ENC-CONFIG-END
(define enc-hd (quotient enc-d enc-heads))
(define enc-conv-kernel 9)
(define n-mels 128)
(define enc-eps 1e-5)

;; --- subsampling: conv 3x3 / dw 3x3 + pw 1x1 / dw + pw, stride 2 in time and frequency, each padded
;; 2 before and 1 after (causal), ReLU after the first conv and each pointwise conv, then a linear over (channel, freq).

(define (pad-causal x) (ggml-pad-ext x 2 1 2 1 0 0 0 0))

;; mel [128, T] -> [enc-d, T/8 + ...]
(define (subsample mel)
  (let* ([x (ggml-reshape-4d mel n-mels (dim mel 1) 1 1)]                                   ; [freq, time, 1]
         [x (ggml-relu (nn:conv2d (pad-causal x) "enc.pre_encode.conv.0" 'stride 2 'padding 0))]
         [x (nn:conv2d-depthwise (pad-causal x) "enc.pre_encode.conv.2" 'stride 2 'padding 0)]
         [x (ggml-relu (nn:conv2d x "enc.pre_encode.conv.3" 'padding 0))]
         [x (nn:conv2d-depthwise (pad-causal x) "enc.pre_encode.conv.5" 'stride 2 'padding 0)]
         [x (ggml-relu (nn:conv2d x "enc.pre_encode.conv.6" 'padding 0))]                   ; [freq', time', C]
         [f (dim x 0)] [t (dim x 1)]
         [x (ggml-cont (ggml-permute x 0 2 1 3))])                                          ; [freq', C, time']
    (nn:linear (ggml-reshape-2d x (* f sub-channels) t) "enc.pre_encode.out")))

;; --- Transformer-XL relative positions: sin / cos of position * 10000^(-2k/d) for positions t-1 down to -(t-1)
;; (interleaved: sin at even channels, cos at odd), [enc-d, 2t-1].
(define (pos-emb t) (pos-emb-at (ggml-scale-bias (ggml-arange 0.0 (inexact (- (* 2 t) 1)) 1.0) -1.0 (inexact (- t 1))) (- (* 2 t) 1)))

;; the same for the given positions (a [w] tensor)
(define (pos-emb-at pos w)
  (let* ([half (quotient enc-d 2)]
         [div (ggml-exp (ggml-scale (ggml-arange 0.0 (inexact half) 1.0) (/ (* -2.0 (log 10000.0)) enc-d)))]
         [angle (ggml-mul-mat (ggml-reshape-2d div 1 half) (ggml-reshape-2d pos 1 w))]       ; [half, w]
         [s (ggml-reshape-3d (ggml-sin angle) 1 half w)]
         [c (ggml-reshape-3d (ggml-cos angle) 1 half w)])
    (ggml-reshape-2d (ggml-concat s c 0) enc-d w)))

(define (e-prefix l) (format "enc.layers.~a" l))
(define (heads-first x t) (ggml-cont (ggml-permute x 0 2 1 3)))   ; [hd, heads, t] -> [hd, t, heads]

;; RelPositionMultiHeadAttention: scores = (q+u) k^T + rel_shift((q+v) p^T), scaled by 1/sqrt(hd), masked.
;; rel_shift is a strided view: query i, key j reads the position score at column t-1-i+j.
(define (rel-attention x l pe mask t)
  (let* ([p (string-append (e-prefix l) ".self_attn")]
         [w (- (* 2 t) 1)]
         [split (lambda (name n) (ggml-reshape-3d (nn:linear x (string-append p name)) enc-hd enc-heads n))]
         [q (split ".linear_q" t)]
         [k (heads-first (split ".linear_k" t) t)]
         [v (split ".linear_v" t)]
         [bias (lambda (name) (ggml-reshape-3d (weight (string-append p name)) enc-hd enc-heads 1))]
         [qu (heads-first (ggml-add q (bias ".pos_bias_u")) t)]
         [qv (heads-first (ggml-add q (bias ".pos_bias_v")) t)]
         [pp (heads-first (ggml-reshape-3d (nn:linear pe (string-append p ".linear_pos")) enc-hd enc-heads w) w)]
         [ac (ggml-mul-mat k qu)]                                                           ; [key, query, heads]
         [bd (ggml-mul-mat pp qv)]                                                          ; [position, query, heads]
         [bd (ggml-cont (ggml-view-3d bd t t enc-heads (* 4 (- w 1)) (* 4 w t) (* 4 (- t 1))))]
         [probs (ggml-soft-max-ext (ggml-add ac bd) mask (/ 1.0 (sqrt (inexact enc-hd))) 0.0)]
         [vt (ggml-cont (ggml-permute v 1 2 0 3))]                                          ; [key, hd, heads]
         [ctx (ggml-cont (ggml-permute (ggml-mul-mat vt probs) 0 2 1 3))])                  ; [hd, heads, query]
    (nn:linear (ggml-reshape-2d ctx enc-d t) (string-append p ".linear_out"))))

;; pointwise conv (GLU) -> causal depthwise conv (kernel 9) -> LayerNorm over channels -> Swish -> pointwise conv
(define (conv-module x l t)
  (let* ([p (string-append (e-prefix l) ".conv")]
         [pointwise (lambda (name n-out x) (ggml-mul-mat (ggml-reshape-2d (weight (string-append p name ".weight")) (dim x 0) n-out) x))]
         [h (pointwise ".pointwise_conv1" (* 2 enc-d) x)]
         [h (ggml-mul (ggml-cont (tensor:slice h 0 0 enc-d)) (ggml-sigmoid (ggml-cont (tensor:slice h 0 enc-d enc-d))))]
         ;; time-major with the kernel's history in front; ssm_conv is a causal depthwise conv
         [hp (ggml-pad-ext (ggml-cont (ggml-transpose h)) (- enc-conv-kernel 1) 0 0 0 0 0 0 0)]
         [kernel (ggml-cast (ggml-reshape-2d (weight (string-append p ".depthwise_conv.weight")) enc-conv-kernel enc-d) GGML_TYPE_F32)]
         [h (ggml-ssm-conv (ggml-reshape-3d hp (+ t (- enc-conv-kernel 1)) enc-d 1) kernel)]   ; [enc-d, t, 1]
         [h (ggml-silu (nn:layer-norm (ggml-reshape-2d h enc-d t) (string-append p ".batch_norm") 'eps enc-eps))])
    (pointwise ".pointwise_conv2" enc-d h)))

(define (conformer-layer x l pe mask t)
  (let* ([p (e-prefix l)]
         [norm (lambda (x name) (nn:layer-norm x (string-append p "." name) 'eps enc-eps))]
         [feed-forward (lambda (x name)
                         (nn:linear (ggml-silu (nn:linear x (string-append p "." name ".linear1"))) (string-append p "." name ".linear2")))]
         [x (ggml-add x (ggml-scale (feed-forward (norm x "norm_feed_forward1") "feed_forward1") 0.5))]
         [x (ggml-add x (rel-attention (norm x "norm_self_att") l pe mask t))]
         [x (ggml-add x (conv-module (norm x "norm_conv") l t))]
         [x (ggml-add x (ggml-scale (feed-forward (norm x "norm_feed_forward2") "feed_forward2") 0.5))])
    (norm x "norm_out")))

(model encode (inputs [mel f32 (128 _)])
  (define sub (subsample mel))
  (define t (dim sub 1))
  (define pe (pos-emb t))
  (define mask (attn:causal-mask t 'window (+ enc-left 1)))
  (define out (let loop ([l 0] [x sub]) (if (= l enc-layers) x (loop (+ l 1) (conformer-layer x l pe mask t)))))
  (outputs [emb out 2] [proj (nn:linear out "perc.proj") 2]))

;; --- the duplex loop (nemo's DuplexSTTModel._step_inference, fuse_method "add", no user-text channel)
;; Every 80 ms frame the language model reads  embed(previous text token) + audio frame + 2 * embed(previous function
;; token)  and predicts this frame's text token (pad = <SPECIAL_12> while silent, bos <s> when the agent starts
;; speaking, eos </s> when it stops) and function-channel token with two heads. The 2 is duplex_function_channel_weight.

(define text-pad 12)

(model duplex (inputs [audio f32 (4480 1)] [text i32 (1)] [func i32 (1)] [pos i32 (1)] [fmask f32 (131072 1)])
  (define (embed ids) (ggml-reshape-2d (nn:embedding "embed_tokens.weight" ids) hidden 1))
  (define x0 (ggml-add (ggml-add (embed text) audio) (ggml-scale (embed func) 2.0)))
  (define positions (tensor:flatten pos))
  (define mask (position-mask positions))
  (define stack (let loop ([i 0] [x x0]) (if (= i layer-count) x (loop (+ i 1) (block x i positions mask)))))
  (define last (nn:rms-norm stack "llm.norm_f" 'eps eps))
  (define text-logits (ggml-mul-mat (weight "lm_head.weight") last))
  (define func-logits (ggml-add (ggml-mul-mat (weight "function_head.weight") last) fmask))   ; fmask: 0 or -1e30 per token
  (outputs [logits text-logits 2] [next_text (ggml-argmax text-logits) 1] [next_func (ggml-argmax func-logits) 1]))

;; The system prompt, one token per frame ahead of the audio: each prompt position reads embed(token) in place of the
;; audio frame, with the pad token on both text channels (embed(pad) + embed(token) + 2 embed(pad)); nothing is
;; predicted there. n positions at once.
(model prefill (inputs [token i32 (_)] [pos i32 (_)] [pad i32 (1)])
  (define n (dim token 0))
  (define positions (tensor:flatten pos))
  (define mask (position-mask positions))
  (define pad-embed (ggml-reshape-2d (nn:embedding "embed_tokens.weight" pad) hidden 1))
  (define embedded (ggml-reshape-2d (nn:embedding "embed_tokens.weight" token) hidden n))
  (define x0 (ggml-add embedded (ggml-repeat (ggml-scale pad-embed 3.0) embedded)))
  (define stack (let loop ([i 0] [x x0]) (if (= i layer-count) x (loop (+ i 1) (block x i positions mask)))))
  (outputs [last (tensor:slice stack 1 (- n 1) 1) 2]))

(define lm-vocab 131072)
(define no-mask (array-zeros 1 lm-vocab))

(define (->int a) (car (util:->integers (array->list a))))

;; The agent's text and function channel over a stretch of user audio, from silence (pad tokens) at the start:
;; (texts . functions), one token per 80 ms frame.
(define (duplex-run audio)
  (let* ([proj (output (run encode [mel (nemo-mel-of audio)]) 'proj)]
         [frames (car (array-shape proj))])
    (let loop ([t 0] [text text-pad] [func text-pad] [texts '()] [funcs '()])
      (if (= t frames)
          (cons (reverse texts) (reverse funcs))
          (let ([r (run duplex [audio (array-slice proj t (+ t 1))] [text (list->array (list text))]
                               [func (list->array (list func))] [pos (list->array (list t))] [fmask no-mask])])
            (let ([next-text (->int (output r 'next_text))] [next-func (->int (output r 'next_func))])
              (loop (+ t 1) next-text next-func (cons next-text texts) (cons next-func funcs))))))))

(pipeline converse ([audio audio])
  (let* ([r (duplex-run audio)] [texts (car r)])
    (results [text (detokenize tok (filter (lambda (id) (not (memv id '(1 2 12)))) texts))] [tokens texts] [function (cdr r)])))

;; --- RNNT (the transducer head trained on the encoder's output, as in nemo's stt_en_fastconformer_transducer):
;; a 2-layer LSTM prediction network over the labels so far, a joint network relu(enc(f) + pred(g)) -> 1025
;; logits (1024 SentencePiece tokens + blank). Greedy: per encoder frame emit labels until the blank.

(define rnnt-h 640)
(define rnnt-blank 1024)
(for-each (lambda (name) (define-state name f32 rnnt-h 1)) '("rnnt.h0" "rnnt.c0" "rnnt.h1" "rnnt.c1"))

;; torch.nn.LSTM cell, gate order i f g o. x, h, c: [640, 1]. Returns (h . c).
(define (lstm-cell x h c layer)
  (let* ([w (lambda (name) (tensor:as-f32 (weight (format "rnnt_dec.dec_rnn.lstm.~a_l~a" name layer))))]
         [gates (ggml-add (ggml-add (ggml-mul-mat (w "weight_ih") x) (w "bias_ih"))
                          (ggml-add (ggml-mul-mat (w "weight_hh") h) (w "bias_hh")))]
         [gate (lambda (i) (ggml-cont (tensor:slice gates 0 (* i rnnt-h) rnnt-h)))]
         [c2 (ggml-add (ggml-mul (ggml-sigmoid (gate 1)) c) (ggml-mul (ggml-sigmoid (gate 0)) (ggml-tanh (gate 2))))])
    (cons (ggml-mul (ggml-sigmoid (gate 3)) (ggml-tanh c2)) c2)))

;; One prediction step: label `token` (ignored when first = 1: the zero start-of-sequence input and zero state)
;; -> g [640], the new state kept for the next label.
(model rnnt-pred (inputs [token i32 (1)] [first f32 (1)])
  (define keep (ggml-scale-bias first -1.0 1.0))
  (define (stored name) (ggml-mul (state name) keep))
  (define x (ggml-mul (ggml-reshape-2d (tensor:as-f32 (ggml-get-rows (weight "rnnt_dec.embed.weight") token)) rnnt-h 1) keep))
  (define layer0 (lstm-cell x (stored "rnnt.h0") (stored "rnnt.c0") 0))
  (define layer1 (lstm-cell (car layer0) (stored "rnnt.h1") (stored "rnnt.c1") 1))
  (effect (ggml-cpy (car layer0) (state "rnnt.h0")))
  (effect (ggml-cpy (cdr layer0) (state "rnnt.c0")))
  (effect (ggml-cpy (car layer1) (state "rnnt.h1")))
  (effect (ggml-cpy (cdr layer1) (state "rnnt.c1")))
  (outputs [g (car layer1) 1]))

;; The transducer's weights are f16 in the file, biases included (ggml adds only f32 biases to f32 activations).
(define (rnnt-linear x prefix)
  (ggml-add (ggml-mul-mat (weight (string-append prefix ".weight")) x) (tensor:as-f32 (weight (string-append prefix ".bias")))))

(model rnnt-joint (inputs [f f32 (1024 1)] [g f32 (640 1)])
  (define logits (rnnt-linear (ggml-relu (ggml-add (rnnt-linear f "rnnt_joint.enc") (rnnt-linear g "rnnt_joint.pred")))
                              "rnnt_joint.joint_net.2"))
  (outputs [next (ggml-argmax logits) 1]))

(define rnnt-tok (tokenizer (asset "rnnt_tokenizer.model")))
(define rnnt-max-symbols 10)   ; labels per frame

;; Greedy transducer decoding of emb [T', 1024]: the label ids.
(define (rnnt-greedy emb)
  (let ([frames (car (array-shape emb))]
        [pred (lambda (token first) (output (run rnnt-pred [token (list->array (list token))] [first (list->array (list first))]) 'g))])
    (let frame-loop ([t 0] [g (pred 0 1.0)] [labels '()])
      (if (= t frames)
          (reverse labels)
          (let ([f (array-slice emb t (+ t 1))])
            (let label-loop ([k 0] [g g] [labels labels])
              (let ([next (car (util:->integers (array->list (output (run rnnt-joint [f f] [g g]) 'next))))])
                (if (or (= next rnnt-blank) (= k rnnt-max-symbols))
                    (frame-loop (+ t 1) g labels)
                    (label-loop (+ k 1) (pred next 0.0) (cons next labels))))))))))

(define (nemo-mel-of audio)
  (let ([ms (nemo-mel-new)])
    (nemo-mel-push! ms (if (= (audio-rate audio) 16000) audio (audio-resample audio 16000)))
    (nemo-mel-window ms 0 (nemo-mel-finish! ms))))

(pipeline transcribe ([audio audio])
  (let* ([emb (output (run encode [mel (nemo-mel-of audio)]) 'emb)]
         [labels (rnnt-greedy emb)])
    (results [text (detokenize rnnt-tok labels)] [tokens labels])))

;; --- front end check

(pipeline mel ([audio audio])
  (results [mel (nemo-mel-of audio)]))
