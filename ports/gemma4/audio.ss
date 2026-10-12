;; Gemma 4's audio tower (a Universal Speech Model conformer) as a streaming encoder, ported from
;; transformers' Gemma4AudioModel on the weights of the mmproj GGUF (a.* tensors, plus mm.a.* for the
;; embedder). Append it to gemma4.ss and audio_clips.ss.
;;
;; The tower is causal: 12 layers of FFN / chunked local attention / light conv / FFN, where a frame
;; attends to itself and the 11 before it (attention_context_left 13 minus 1 for the chunk = a window of 12),
;; the depthwise conv has kernel 5 and looks only back. So it runs on a stream, 12 output tokens (480 ms,
;; 48 mel frames) per call, keeping per layer the last 11 keys and values and the last 4 conv inputs as
;; state:
;;
;;   audio-chunk  mel [51, 128] (the mel frames 48c-3 .. 48c+47, zeros where they don't exist),
;;                mask1 [25] (1 where the first conv's output frame exists), winidx [144] i32, posemb [12, 1024],
;;                amask [12, 12] (0 / -1e9: may query i look d frames back)
;;                -> emb [12, 1536] (the 12 audio tokens in the language model's space), tower [12, 1536]
;;
;; The first conv layer needs 3 mel frames of lookahead (stride 2 twice over a kernel of 3), so token k is
;; ready when mel frame 4k + 3 is. Shapes in comments are ggml order (innermost first).

(define a-hidden 1024)
(define a-heads 8)
(define a-hd 128)
(define a-layers 12)
(define a-chunk 12)               ; tokens per call
(define a-past 11)                ; frames of context kept
(define a-window 12)              ; a frame sees itself and 11 before
(define a-mel-window 51)

(define q-scale (/ (/ 1.0 (sqrt 128.0)) (log 2.0)))
(define k-scale (/ (log (+ 1.0 (exp 1.0))) (log 2.0)))

(define (clip-of name) (cdr (assoc name audio-clips)))
(define (clip-in x name) (let ([c (clip-of name)]) (ggml-clamp x (car c) (cadr c))))
(define (clip-out x name) (let ([c (clip-of name)]) (ggml-clamp x (caddr c) (cadddr c))))
;; Gemma4ClippableLinear: clamp the input, the matrix product, clamp the output.
(define (c-linear x name) (clip-out (nn:linear (clip-in x name) name) name))
(define (a-norm x name) (nn:rms-norm x name 'eps 1e-6))

(for-each (lambda (l)
            (define-state (format "ak.~a" l) f32 a-hidden a-past)
            (define-state (format "av.~a" l) f32 a-hidden a-past)
            (define-state (format "ac.~a" l) f32 a-hidden 4))
          (iota a-layers))

;; --- subsampling: two strided 3x3 convs (time stride 2 twice), channel layer norm, ReLU

;; x [128, 51] -> [1024, 12]. The windows hold the context each conv needs, so only the frequency axis is padded.
(define (subsample mel mask1)
  (let* ([x (ggml-reshape-3d mel 128 a-mel-window 1)]
         [h (nn:conv2d x "a.conv1d.0" 'stride 2 'padding '(0 1))]           ; [64, 25, 128]
         [h (ggml-cont (ggml-permute h 1 2 0 3))]                            ; [128, 64, 25]
         [h (ggml-relu (nn:layer-norm h "a.conv1d.0.norm" 'eps 1e-6))]
         [h (ggml-mul h (ggml-reshape-3d mask1 1 1 25))]                    ; frames the first conv did not produce are zero
         [h (ggml-cont (ggml-permute h 2 0 1 3))]                            ; [64, 25, 128]
         [h (nn:conv2d h "a.conv1d.1" 'stride 2 'padding '(0 1))]            ; [32, 12, 32]
         [h (ggml-cont (ggml-permute h 1 2 0 3))]                            ; [32, 32, 12] = (channel, freq, time)
         [h (ggml-relu (nn:layer-norm h "a.conv1d.1.norm" 'eps 1e-6))])
    (nn:linear (ggml-reshape-2d h 1024 a-chunk) "a.input_projection")))

;; --- conformer layer

(define (ffn x p sfx)
  (let* ([h (a-norm x (string-append p "ffn_norm" sfx))]
         [h (ggml-silu (c-linear h (string-append p "ffn_up" sfx)))]
         [h (c-linear h (string-append p "ffn_down" sfx))]
         [h (a-norm h (string-append p "ffn_post_norm" sfx))])
    (ggml-add x (ggml-scale h 0.5))))

;; Chunked local attention with relative positions, for the 12 new frames against the 11 kept before
;; them. Scores are laid out [d, i, head] (d = how far back the key is, i = query).
(define (attention x p l winidx posemb amask)
  (let* ([m a-chunk]
         [h (a-norm x (string-append p "attn_pre_norm"))]
         [q (c-linear h (string-append p "attn_q"))]
         [k (ggml-scale (c-linear h (string-append p "attn_k")) k-scale)]
         [v (c-linear h (string-append p "attn_v"))]
         [per-dim (ggml-scale (weight (string-append p "per_dim_scale.weight")) q-scale)]   ; the GGUF stores softplus(per_dim_scale) already
         [q3 (ggml-mul (ggml-reshape-3d q a-hd a-heads m) (ggml-reshape-3d per-dim a-hd 1 1))]   ; [128, 8, m]
         ;; keys and values: the kept 11 frames, then the new ones
         [kctx (ggml-concat (state (format "ak.~a" l)) k 1)]
         [vctx (ggml-concat (state (format "av.~a" l)) v 1)])
    (effect (ggml-cpy (tensor:slice kctx 1 m a-past) (state (format "ak.~a" l))))
    (effect (ggml-cpy (tensor:slice vctx 1 m a-past) (state (format "av.~a" l))))
    (let* ([kw (ggml-reshape-4d (ggml-get-rows kctx winidx) a-hd a-heads a-window m)]          ; (c, head, d, i)
           [kw (ggml-cont (ggml-permute kw 0 3 1 2))]                                         ; [128, d, i, head]
           [vw (ggml-reshape-4d (ggml-get-rows vctx winidx) a-hd a-heads a-window m)]
           [vw (ggml-cont (ggml-permute vw 1 3 0 2))]                                         ; [d, 128, i, head]
           [qa (ggml-cont (ggml-permute (ggml-reshape-4d q3 a-hd a-heads m 1) 0 3 2 1))]       ; [128, 1, i, head]
           [ac (ggml-reshape-3d (ggml-mul-mat kw qa) a-window m a-heads)]                     ; [d, i, head]
           ;; position term: queries against the projected sinusoids of the 12 distances
           [rel (ggml-reshape-3d (nn:linear posemb (string-append p "attn_k_rel")) a-hd a-heads a-window)]
           [rel (ggml-cont (ggml-permute rel 0 2 1 3))]                                       ; [128, d, head]
           [qb (ggml-cont (ggml-permute q3 0 2 1 3))]                                         ; [128, i, head]
           [bd (ggml-mul-mat rel qb)]                                                         ; [d, i, head]
           [s (ggml-add ac bd)]
           [s (ggml-scale (ggml-tanh (ggml-scale s (/ 1.0 50.0))) 50.0)]
           [w (ggml-soft-max (ggml-add s amask))]                                             ; over d
           [o (ggml-mul-mat vw (ggml-reshape-4d w a-window 1 m a-heads))]                     ; [128, 1, i, head]
           [o (ggml-reshape-2d (ggml-cont (ggml-permute o 0 3 2 1)) a-hidden m)])
      (c-linear o (string-append p "attn_out")))))

(define (light-conv x p l)
  (let* ([h (a-norm x (string-append p "conv_norm"))]                                    ; (the mmproj calls the norm before the conv conv_norm, and the one after it norm_conv)
         [h (c-linear h (string-append p "conv_pw1"))]                                       ; [2048, m]
         [u (ggml-mul (ggml-cont (tensor:slice h 0 0 a-hidden))
                      (ggml-sigmoid (ggml-cont (tensor:slice h 0 a-hidden a-hidden))))]
         [ctx (ggml-concat (state (format "ac.~a" l)) u 1)]                                  ; [1024, 4 + m]
         [taps (ggml-cont (ggml-transpose (tensor:as-f32 (weight (string-append p "conv_dw.weight")))))])    ; [1024, 5]
    (effect (ggml-cpy (tensor:slice ctx 1 a-chunk 4) (state (format "ac.~a" l))))
    (let* ([y (let loop ([j 0] [acc #f])
                (if (= j 5)
                    acc
                    (let ([term (ggml-mul (tensor:slice ctx 1 j a-chunk) (tensor:slice taps 1 j 1))])
                      (loop (+ j 1) (if acc (ggml-add acc term) term)))))]
           [y (a-norm y (string-append p "norm_conv"))]
           [y (c-linear (ggml-silu y) (string-append p "conv_pw2"))])
      (ggml-add x y))))

(define (audio-layer x l winidx posemb amask)
  (let* ([p (format "a.blk.~a." l)]
         [x (ffn x p "")]
         [x (ggml-add x (a-norm (attention x p l winidx posemb amask) (string-append p "attn_post_norm")))]
         [x (light-conv x p l)]
         [x (ffn x p "_1")])
    (a-norm x (string-append p "ln2"))))

(model audio-chunk (inputs [mel f32 (128 51)] [mask1 f32 (25)] [winidx i32 (144)] [posemb f32 (1024 12)] [amask f32 (12 12)])
  (define x0 (subsample mel mask1))
  (define x (let loop ([l 0] [x x0]) (if (= l a-layers) x (loop (+ l 1) (audio-layer x l winidx posemb amask)))))
  (define tower (nn:linear x "a.pre_encode.out"))                                              ; [1536, 12]
  (define emb (nn:linear (ggml-rms-norm tower 1e-6) "mm.a.input_projection"))
  (outputs [sub x0 2] [tower tower 2] [emb emb 2]))

;; --- host side: what each call needs besides the mel window

(define (sinusoid-table)
  (let ([inv (lambda (j) (exp (* (- j) (/ (log 10000.0) 511.0))))])
    (list->array
      (let loop ([d 0] [acc '()])
        (if (= d a-window)
            (reverse acc)
            (loop (+ d 1)
                  (append (reverse (append (map (lambda (j) (sin (* d (inv j)))) (iota 512))
                                           (map (lambda (j) (cos (* d (inv j)))) (iota 512))))
                          acc)))))))
(define audio-posemb (array-reshape (sinusoid-table) a-window a-hidden))
(define audio-winidx
  (list->array (let loop ([i 0] [acc '()])
                 (if (= i a-chunk) (reverse acc)
                     (loop (+ i 1) (append (reverse (map (lambda (d) (+ i a-past (- d))) (iota a-window))) acc))))))
;; Chunk c's masks for `frames` mel frames in all: mask1[i] says the first conv's frame 24c - 1 + i exists
;; (it has a mel frame at its centre), amask[i][d] says token 12c + i may look d tokens back (token valid = 4k < frames).
(define (chunk-masks c frames)
  (let ([mask1 (map (lambda (i) (let ([j (+ (* 24 c) -1 i)]) (if (and (>= j 0) (< (* 2 j) frames)) 1.0 0.0))) (iota 25))]
        [amask (let loop ([i 0] [acc '()])
                 (if (= i a-chunk)
                     (reverse acc)
                     (loop (+ i 1)
                           (append (reverse (map (lambda (d)
                                                   (let ([g (- (+ (* a-chunk c) i) d)])
                                                     (if (and (>= g 0) (< (* 4 g) frames)) 0.0 -1e9)))
                                                 (iota a-window)))
                                   acc))))])
    (values (list->array mask1) (array-reshape (list->array amask) a-chunk a-window))))

;; Test pipeline: the whole utterance's mel features [frames, 128] with 3 zero rows in front and zeros behind
;; (enough for the last chunk), run chunk by chunk; returns the embeddings of every chunk as e0, e1, ... and
;; the tower outputs as t0, t1, ...
(pipeline audio-test ([feats array] [frames string])
  (let* ([frames (string->number frames)]
         [chunks (quotient (+ frames 47) 48)])
    (let loop ([c 0] [out '()])
      (if (= c chunks)
          (reverse out)
          (let-values ([(mask1 amask) (chunk-masks c frames)])
            (let ([r (run audio-chunk [mel (array-slice feats (* 48 c) (+ (* 48 c) 51))] [mask1 mask1]
                                      [winidx audio-winidx] [posemb audio-posemb] [amask amask])])
              (loop (+ c 1)
                    (cons (cons (string->symbol (format "e~a" c)) (output r 'emb))
                          (cons (cons (string->symbol (format "t~a" c)) (output r 'tower)) out)))))))))

;; The front end alone: all log-mel frames of an utterance, [frames, 128].
(pipeline mel-test ([audio audio])
  (let* ([ms (usm-mel-new)]
         [frames (usm-mel-push! ms audio)])
    (results [mel (usm-mel-window ms 0 frames)])))
