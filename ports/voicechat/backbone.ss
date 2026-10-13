;; NVIDIA NemotronLabs VoiceChat 11B: the text backbone (Nemotron-H 9B hybrid Mamba2 /
;; attention / MLP LLM, `stt_model.llm.*` + the embedding and `lm_head`), ported from
;; transformers' modeling_nemotron_h.py (nvidia/NVIDIA-Nemotron-Nano-9B-v2).
;; The FastConformer, the TTS decoder and the duplex loop are not ported yet.
;;
;; Entries (shapes in numpy order; the batch is 1):
;;   step   token [1, n] i32, pos [1, n] i32 -> logits [1, vocab] (last token), next [1] (argmax)
;;          n tokens at positions pos (a prefill chunk or one decode token). The Mamba
;;          layers keep a conv window and an SSM state, the attention layers a KV cache.
;; Pipelines (raw inputs):
;;   generate  prompt -> text, tokens        greedy, at most 48 new tokens
;;
;; Tensor names are the converted file's (ports/voicechat/convert.sh): llm.layers.N.mixer.*.
;; Every layer is x + mixer(rms_norm(x)). Shapes in comments are ggml order (innermost first).

(import (tl tensor) (tl nn) (tl attn) (tl util))

;; --- configuration (config.json of nvidia/NVIDIA-Nemotron-Nano-9B-v2; reference_tiny.py's
;; tiny model replaces these lines)
;; CONFIG-BEGIN
(define pattern "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-")  ; M Mamba2, - MLP, * attention
(define hidden 4480)
(define n-heads 128)      ; Mamba2 heads
(define mamba-head-dim 80)
(define n-group 8)        ; B/C groups
(define d-state 128)
(define q-heads 40)
(define kv-heads 8)
(define head-dim 128)     ; attention head size
;; CONFIG-END
(define eps 1e-5)
(define d-conv 4)
(define capacity 2048)    ; tokens the attention KV cache holds

(define d-inner (* n-heads mamba-head-dim))
(define conv-dim (+ d-inner (* 2 n-group d-state)))   ; x, B and C go through the conv
(define group-size (quotient d-inner n-group))
(define kv-dim (* kv-heads head-dim))
(define layer-count (string-length pattern))

(define dim tensor:dim)
(define (prefix i) (format "llm.layers.~a" i))
(define (mixer-prefix i) (string-append (prefix i) ".mixer"))
(define (layer-kind i)
  (case (string-ref pattern i) [(#\M) 'mamba] [(#\*) 'attention] [else 'mlp]))

;; --- recurrent state and KV cache

(for-each
 (lambda (i)
   (case (layer-kind i)
     [(mamba)
      (define-state (format "conv.~a" i) f32 (- d-conv 1) conv-dim)
      (define-state (format "ssm.~a" i) f32 d-state mamba-head-dim n-heads 1)]
     [(attention)
      (define-state (format "k.~a" i) f16 kv-dim capacity)
      (define-state (format "v.~a" i) f16 kv-dim capacity)]
     [else #f]))
 (iota layer-count))

;; --- Mamba2

;; x [hidden, n] -> [hidden, n]. The scan runs the n tokens in order from the stored state.
(define (mamba x i)
  (let* ([m (mixer-prefix i)]
         [n (dim x 1)]
         [w (lambda (name) (weight (string-append m "." name)))]
         [proj (nn:linear x (string-append m ".in_proj"))]                   ; [z | x B C | dt]
         [z (tensor:slice proj 0 0 d-inner)]
         [xbc (tensor:slice proj 0 d-inner conv-dim)]
         [dt (tensor:slice proj 0 (+ d-inner conv-dim) n-heads)]
         ;; causal depthwise conv over the last d-conv-1 stored columns plus the new ones
         [conv-in (ggml-concat (state (format "conv.~a" i)) (ggml-cont (ggml-transpose xbc)) 0)]  ; [d-conv-1+n, conv-dim]
         [conv-out (ggml-ssm-conv (ggml-reshape-3d conv-in (+ n (- d-conv 1)) conv-dim 1)
                                  (ggml-reshape-2d (w "conv1d.weight") d-conv conv-dim))]
         [xbc (ggml-silu (ggml-add conv-out (w "conv1d.bias")))]               ; [conv-dim, n, 1]
         [row (* 4 conv-dim)]
         [x-ssm (ggml-view-4d xbc mamba-head-dim n-heads n 1 (* 4 mamba-head-dim) row (* row n) 0)]
         [b (ggml-view-4d xbc d-state n-group n 1 (* 4 d-state) row (* row n) (* 4 d-inner))]
         [c (ggml-view-4d xbc d-state n-group n 1 (* 4 d-state) row (* row n) (* 4 (+ d-inner (* n-group d-state))))]
         [dt (ggml-reshape-3d (ggml-add dt (w "dt_bias")) n-heads n 1)]        ; the scan applies softplus
         [a (ggml-reshape-2d (ggml-neg (ggml-exp (w "A_log"))) 1 n-heads)]
         [ids (ggml-cast (ggml-arange 0.0 1.0 1.0) GGML_TYPE_I32)]
         [ssm (state (format "ssm.~a" i))]
         [scan (ggml-ssm-scan ssm x-ssm dt a b c ids 1)]                        ; y, then the new state
         [state-size (* d-state mamba-head-dim n-heads)]
         [y-bytes (* 4 d-inner n)])
    (effect (ggml-cpy (ggml-view-2d conv-in (- d-conv 1) conv-dim (* 4 (+ n (- d-conv 1))) (* 4 n))
                      (state (format "conv.~a" i))))
    (effect (ggml-cpy (ggml-view-1d scan state-size y-bytes) (ggml-view-1d ssm state-size 0)))
    (let* ([y (ggml-view-4d scan mamba-head-dim n-heads n 1 (* 4 mamba-head-dim) (* 4 d-inner) y-bytes 0)]
           [y (ggml-add y (ggml-mul x-ssm (ggml-reshape-3d (w "D") 1 n-heads 1)))]       ; skip connection
           [gated (ggml-mul (ggml-silu (ggml-cont z)) (ggml-reshape-2d y d-inner n))]
           ;; RMSNorm over each group of d-inner / n-group channels (after the gate)
           [normed (ggml-mul (ggml-rms-norm (ggml-reshape-3d gated group-size n-group n) eps)
                             (ggml-reshape-2d (w "norm.weight") group-size n-group))])
      (nn:linear (ggml-reshape-2d normed d-inner n) (string-append m ".out_proj")))))

;; --- MLP (ReLU^2)

(define (mlp x i)
  (let ([m (mixer-prefix i)])
    (nn:linear (ggml-sqr (ggml-relu (nn:linear x (string-append m ".up_proj")))) (string-append m ".down_proj"))))

;; --- attention (grouped-query, causal, no RoPE)

;; Additive causal mask [capacity, n] from the query positions: key j is allowed where j <= pos.
(define (position-mask pos)
  (let* ([n (dim pos 0)]
         [q (ggml-repeat-4d (ggml-reshape-2d (ggml-cast pos GGML_TYPE_F32) 1 n) capacity n 1 1)]
         [dist (ggml-sub q (ggml-reshape-2d (ggml-arange 0.0 (inexact capacity) 1.0) capacity 1))]
         [allowed (ggml-step (ggml-scale-bias dist 1.0 1.0))])
    (ggml-scale-bias allowed (- attn:blocked) attn:blocked)))

(define (attention x i pos mask)
  (let* ([m (mixer-prefix i)]
         [n (dim x 1)]
         [k-cache (state (format "k.~a" i))]
         [v-cache (state (format "v.~a" i))]
         [q (ggml-permute (ggml-reshape-3d (nn:linear x (string-append m ".q_proj")) head-dim q-heads n) 0 2 1 3)])
    (effect (ggml-set-rows k-cache (nn:linear x (string-append m ".k_proj")) pos))
    (effect (ggml-set-rows v-cache (nn:linear x (string-append m ".v_proj")) pos))
    ;; the cache is [kv-dim, capacity]; a head is a 128-wide slice of a row
    (let ([heads-of (lambda (cache)
                      (ggml-view-4d cache head-dim capacity kv-heads 1
                                    (* 2 kv-dim) (* 2 head-dim) (* 2 kv-dim capacity) 0))])
      (nn:linear (attn:sdpa q (heads-of k-cache) (heads-of v-cache) 'mask mask 'flash #t)
                 (string-append m ".o_proj")))))

;; --- model

(define (block x i pos mask)
  (let ([h (nn:rms-norm x (string-append (prefix i) ".norm") 'eps eps)])
    (ggml-add x (case (layer-kind i)
                  [(mamba) (mamba h i)]
                  [(attention) (attention h i pos mask)]
                  [else (mlp h i)]))))

(model step (inputs [token i32 (_ 1)] [pos i32 (_ 1)])
  (define ids (tensor:flatten token))
  (define positions (tensor:flatten pos))
  (define n (dim ids 0))
  (define mask (position-mask positions))
  (define stack (let loop ([i 0] [x (ggml-reshape-2d (nn:embedding "embed_tokens.weight" ids) hidden n)])
                   (if (= i layer-count) x (loop (+ i 1) (block x i positions mask)))))
  (define last (nn:rms-norm (tensor:slice stack 1 (- n 1) 1) "llm.norm_f" 'eps eps))
  (define logits (ggml-mul-mat (weight "lm_head.weight") last))
  (outputs [logits logits 2] [next (ggml-argmax logits) 1]))

;; --- pipeline

(define tok (tokenizer (asset "tokenizer.json")))
(define max-new 48)

(pipeline generate ([prompt string])
  (let-values ([(ids mask) (tokenize tok prompt)])
    (let* ([prompt-ids (util:->integers (array->list ids))]
           [m (length prompt-ids)])
      (when (> (+ m max-new) capacity) (error 'generate "prompt too long for the cache" m))
      (let loop ([r (run step [token ids] [pos (list->array (iota m))])] [tokens '()] [pos m])
        (let* ([next (car (util:->integers (array->list (output r 'next))))]
               [tokens (append tokens (list next))])
          (if (or (memv next '(2 12)) (= (length tokens) max-new))
              (results [text (detokenize tok tokens)] [tokens tokens] [prompt_ids prompt-ids])
              (loop (run step [token (list->array (list next))] [pos (list->array (list pos))])
                    tokens (+ pos 1))))))))
