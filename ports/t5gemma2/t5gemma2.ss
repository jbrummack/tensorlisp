;; T5Gemma 2 (google/t5gemma-2-270m-270m): a Gemma 3 encoder-decoder with a
;; SigLIP vision tower, ported from transformers' modeling_t5gemma2.py.
;;
;; Entries (shapes in numpy order; the batch is 1):
;;   encode        ids [1, L] i32, gather [1, L] i32, pos [1, L] i32, mask [1, L] f32
;;                 -> memory [1, L, 640]                      text-only encoder
;;   encode-image  pixels [1, 3, 896, 896] + the above        encoder with one image
;;                 -> memory [1, L, 640], image [1, 256, 640]
;;   decode        tokens [1, T] i32, pos [1, T] i32, memory [1, M, 640],
;;                 memory-mask [1, M] f32, at [1, 1] i32
;;                 -> logits [1, 262144] at position `at`     (tap "all_logits": every position)
;;                 the whole decoder, without cache (reference / teacher forcing)
;;   cross         memory [1, M, 640] -> (none): fills the cross-attention K/V cache
;;   decode-step   token [1, 1] i32, pos [1, 1] i32, self-mask [1, 64] f32,
;;                 memory-mask [1, 1024] f32 -> logits [1, 262144], next [1] (argmax)
;;                 one token: writes its K/V into the self-attention cache at pos
;; Pipelines (raw inputs):
;;   caption  image + prompt (with "<start_of_image>") -> text, tokens
;;   generate prompt -> text, tokens
;;
;; gather places the image features and the end-of-image embedding: position
;; i takes row gather[i] of [token embeddings (L); image features (256); eoi]
;; (computed by the pipelines from the token ids).
;;
;; Decoding uses a KV cache in state tensors: `cross` stores every layer's
;; keys and values of the encoder output once, `decode-step` then runs one
;; token per call against both caches. Shapes are fixed (cache capacities
;; below), so each entry's graph is built once.
;;
;; Shapes in comments are ggml order (innermost first).

(define hidden 640)
(define heads 4)
(define head-dim 256)
(define layers 18)
(define eps 1e-6)
(define sliding-window 512)
(define (sliding? i) (not (= (remainder (+ i 1) 6) 0)))   ; every 6th layer is global
(define vocab 262144)

(define vision-width 1152)
(define vision-heads 16)
(define vision-layers 27)
(define image-size 896)
(define patch 14)
(define image-tokens 256)

;; Decoder length: prompt-less generation starts from <bos>, so at most
;; decode-length - 1 new tokens.
(define decode-length 64)
;; Longest encoder output the cross-attention cache holds (an image prompt is ~270).
(define memory-capacity 1024)
;; #f decodes with the `decode` entry instead (recomputes every position per token).
(define use-kv-cache #t)

;; Fused attention kernel for the vision tower (K/V in f16): needed for 4096
;; tokens on the GPU; #f computes exact f32 attention (1 GiB of scores per layer).
(define flash-attention #t)
;; ggml's GELU uses an f16 lookup table on the CPU (~1e-3); #t computes the
;; tanh approximation exactly from ops (for comparisons with PyTorch).
(define exact-gelu #f)

;; --- helpers

(define (dim t i)
  (let ([s (shape t)])
    (if (< i (length s)) (list-ref s i) 1)))

(define (linear x prefix)
  (ggml-add (ggml-mul-mat (weight (string-append prefix "weight")) x)
            (weight (string-append prefix "bias"))))

(define (project x name) (ggml-mul-mat (weight name) x))

;; Weights added to activations must be f32 (e.g. in an f16 file).
(define (as-f32 t) (if (eq? (dtype t) 'f32) t (ggml-cast t GGML_TYPE_F32)))

(define (layer-norm x prefix)
  (ggml-add (ggml-mul (ggml-norm x eps) (weight (string-append prefix "weight")))
            (weight (string-append prefix "bias"))))

;; Gemma's RMSNorm scales by (1 + weight); the file stores 1 + weight
;; (converted with --offset, see README), which saves an op per norm and lets
;; backends fuse the norm with the multiplication.
(define (rms-norm x name)
  (ggml-mul (ggml-rms-norm x eps) (weight name)))

;; gelu_pytorch_tanh
(define (gelu x)
  (if exact-gelu
      (let* ([inner (ggml-scale (ggml-add x (ggml-scale (ggml-mul x (ggml-mul x x)) 0.044715))
                                (sqrt (/ 2.0 3.141592653589793)))])
        (ggml-mul (ggml-scale x 0.5) (ggml-scale-bias (ggml-tanh inner) 1.0 1.0)))
      (ggml-gelu x)))

;; Additive attention mask [n-k, n-q] from positions 0..n-1: 0 where
;; (allowed? dist) with dist = q - k, a large negative number elsewhere.
;; allowed is a list of (scale . bias): allowed where every scale * dist + bias > 0.
(define big 1e30)
(define (distance-mask n-k n-q conditions)
  (let* ([q (ggml-repeat-4d (ggml-reshape-2d (ggml-arange 0.0 (inexact n-q) 1.0) 1 n-q) n-k n-q 1 1)]
         [dist (ggml-sub q (ggml-reshape-2d (ggml-arange 0.0 (inexact n-k) 1.0) n-k 1))]
         [allowed (fold-left (lambda (acc c) (ggml-mul acc (ggml-step (ggml-scale-bias dist (car c) (cdr c)))))
                             (ggml-step (ggml-scale-bias dist (caar conditions) (cdar conditions)))
                             (cdr conditions))])
    (ggml-scale-bias allowed big (- big))))

;; 0 for keys with mask 1, a large negative number for mask 0: [n-k] (broadcasts over queries).
(define (padding-mask mask) (ggml-scale-bias mask big (- big)))

;; Multi-query attention: q [hd, heads, Tq]; k, v [hd, Tk] (one kv head); mask [Tk, Tq].
(define (attend q k v mask)
  (let* ([tq (dim q 2)]
         [q (ggml-cont (ggml-permute q 0 2 1 3))]                                      ; [hd, Tq, heads]
         [probs (ggml-soft-max-ext (ggml-mul-mat k q) mask (/ 1.0 (sqrt (inexact head-dim))) 0.0)]  ; [Tk, Tq, heads]
         [out (ggml-mul-mat (ggml-cont (ggml-transpose v)) probs)])                   ; [hd, Tq, heads]
    (ggml-reshape-2d (ggml-cont (ggml-permute out 0 2 1 3)) (* head-dim heads) tq)))

;; GGML_ROPE_TYPE_NEOX (a #define, not exported): rotate halves, like rotate_half.
(define rope-neox 2)

(define (rope x pos layer)
  (if (sliding? layer)
      (ggml-rope-ext x pos #f head-dim rope-neox 0 10000.0 1.0 0.0 1.0 32.0 1.0)
      ;; Global layers: theta 1e6, linear scaling by 8.
      (ggml-rope-ext x pos #f head-dim rope-neox 0 1000000.0 0.125 0.0 1.0 32.0 1.0)))

(define (mlp x p)
  (project (ggml-mul (gelu (project x (p "mlp.gate_proj.weight"))) (project x (p "mlp.up_proj.weight")))
           (p "mlp.down_proj.weight")))

;; Pre/post-normed residual block around f.
(define (sandwich x p norm-pre norm-post f)
  (ggml-add x (rms-norm (f (rms-norm x (p norm-pre))) (p norm-post))))

(define (prefix-fn stack i)
  (let ([prefix (string-append stack ".layers." (number->string i) ".")])
    (lambda (name) (string-append prefix name))))

;; --- text embeddings

;; Token embeddings are scaled by sqrt(640). transformers (5.x) keeps the
;; encoder's factor in the checkpoint's bf16 (25.25) but the decoder's in f32;
;; the port does the same.
(define embed-scale 25.25)
(define decoder-embed-scale (sqrt (inexact hidden)))

;; Token embeddings [D, L], with image features [D, 256] (or #f) and the
;; end-of-image embedding placed by gather.
(define (embed ids gather image-features)
  (let* ([tokens (ggml-scale (ggml-get-rows (weight "encoder.embed_tokens.weight") ids) embed-scale)]
         [eoi (ggml-reshape-2d (weight "encoder.embed_tokens.eoi_embedding") hidden 1)]
         [table (if image-features
                    (ggml-concat (ggml-concat tokens image-features 1) eoi 1)
                    (ggml-concat tokens eoi 1))])
    (ggml-get-rows table gather)))

;; --- encoder

(define (encoder-layer x i pos masks)
  (let ([p (prefix-fn "encoder" i)])
    (let* ([x (sandwich x p "pre_self_attn_layernorm.weight" "post_self_attn_layernorm.weight"
                (lambda (h)
                  (let* ([n (dim h 1)]
                         [q (rope (rms-norm (ggml-reshape-3d (project h (p "self_attn.q_proj.weight")) head-dim heads n)
                                            (p "self_attn.q_norm.weight"))
                                  pos i)]
                         [k (rope (rms-norm (ggml-reshape-3d (project h (p "self_attn.k_proj.weight")) head-dim 1 n)
                                            (p "self_attn.k_norm.weight"))
                                  pos i)]
                         [v (project h (p "self_attn.v_proj.weight"))])
                    (project (attend q (ggml-reshape-2d k head-dim n) v (if (sliding? i) (car masks) (cdr masks)))
                             (p "self_attn.o_proj.weight")))))])
      (sandwich x p "pre_feedforward_layernorm.weight" "post_feedforward_layernorm.weight"
                (lambda (h) (mlp h p))))))

;; Bidirectional; sliding layers see -257 < q - k < 256 (window (w+1)/2 left, w/2 + 1 right).
(define (encode x pos mask)
  (let* ([n (dim x 1)]
         [pad (padding-mask mask)]
         [global (ggml-add (distance-mask n n '((0.0 . 1.0))) pad)]
         [local (ggml-add (distance-mask n n (list (cons -1.0 (inexact (quotient (+ sliding-window 1) 2)))
                                                   (cons 1.0 (inexact (+ (quotient sliding-window 2) 1)))))
                          pad)])
    (let loop ([i 0] [x x])
      (if (= i layers)
          (rms-norm x "encoder.norm.weight")
          (loop (+ i 1) (tap (format "encoder.~2,'0d" i) (encoder-layer x i pos (cons local global)) 2))))))

;; --- vision tower (SigLIP) and projector

(define (patch-embed image prefix p)
  (let* ([kernel (weight (string-append prefix "weight"))]                  ; [p, p, C, D]
         [cols (ggml-im2col kernel image p p 0 0 1 1 #t GGML_TYPE_F32)]    ; [p*p*C, gw, gh, 1]
         [k (dim cols 0)] [n (* (dim cols 1) (dim cols 2))]
         [out (ggml-mul-mat (ggml-reshape-2d kernel k (dim kernel 3)) (ggml-reshape-2d cols k n))])
    (ggml-add out (weight (string-append prefix "bias")))))

(define (vision-attention x p)
  (let* ([n (dim x 1)] [hd (quotient vision-width vision-heads)]
         [split (lambda (name) (ggml-permute (ggml-reshape-3d (linear x (p name)) hd vision-heads n) 0 2 1 3))]  ; [hd, N, heads]
         [q (split "self_attn.q_proj.")] [k (split "self_attn.k_proj.")] [v (split "self_attn.v_proj.")]
         [scale (/ 1.0 (sqrt (inexact hd)))])
    (linear
      (if flash-attention
          (ggml-reshape-2d (ggml-flash-attn-ext (ggml-cont q) (ggml-cast k GGML_TYPE_F16) (ggml-cast v GGML_TYPE_F16)
                                                #f scale 0.0 0.0)
                           vision-width n)
          (let* ([probs (ggml-soft-max-ext (ggml-mul-mat (ggml-cont k) (ggml-cont q)) #f scale 0.0)]  ; [N_k, N_q, heads]
                 [out (ggml-mul-mat (ggml-cont (ggml-permute v 1 0 2 3)) probs)])                   ; [hd, N_q, heads]
            (ggml-reshape-2d (ggml-cont (ggml-permute out 0 2 1 3)) vision-width n)))
      (p "self_attn.out_proj."))))

(define (vision-layer x i)
  (let* ([p (prefix-fn "vision.encoder" i)]
         [x (ggml-add x (vision-attention (layer-norm x (p "layer_norm1.")) p))])
    (ggml-add x (linear (gelu (linear (layer-norm x (p "layer_norm2.")) (p "mlp.fc1."))) (p "mlp.fc2.")))))

;; pixels [896, 896, 3, 1] -> image features [640, 256]
(define (image-features pixels)
  (let* ([grid (quotient image-size patch)]                                         ; 64
         [x (ggml-add (patch-embed pixels "vision.embeddings.patch_embedding." patch)
                      (as-f32 (weight "vision.embeddings.position_embedding.weight")))]  ; [1152, 4096]
         [x (let loop ([i 0] [x x])
              (if (= i vision-layers) x (loop (+ i 1) (vision-layer x i))))]
         [x (tap "vision" (layer-norm x "vision.post_layernorm.") 2)]
         ;; 4 x 4 average pooling of the 64 x 64 grid, row-major like the tokens.
         [pooled-side (exact (round (sqrt image-tokens)))]
         [kernel (quotient grid pooled-side)]
         [spatial (ggml-cont (ggml-permute (ggml-reshape-3d x vision-width grid grid) 2 0 1 3))]  ; [64, 64, 1152]
         [pooled (ggml-pool-2d spatial GGML_OP_POOL_AVG kernel kernel kernel kernel 0.0 0.0)]    ; [16, 16, 1152]
         [tokens (ggml-reshape-2d (ggml-cont (ggml-permute pooled 1 2 0 3)) vision-width image-tokens)]  ; [1152, 256]
         [normed (rms-norm tokens "mm.mm_soft_emb_norm.weight")])
    ;; mm_input_projection_weight is (in 1152, out 640) in torch: [640, 1152] here.
    (ggml-mul-mat (ggml-cont (ggml-transpose (weight "mm.mm_input_projection_weight"))) normed)))

(define (flat t) (ggml-reshape-1d t (dim t 0)))

(model encode (inputs [ids i32 (_ 1)] [gather i32 (_ 1)] [pos i32 (_ 1)] [mask f32 (_ 1)])
  (define x (tap "embeddings" (embed (flat ids) (flat gather) #f) 2))
  (outputs [memory (encode x (flat pos) (flat mask)) 3]))

(model encode-image (inputs [pixels f32 (896 896 3 1)] [ids i32 (_ 1)] [gather i32 (_ 1)] [pos i32 (_ 1)] [mask f32 (_ 1)])
  (define features (tap "image_features" (image-features pixels) 2))
  (define x (tap "embeddings" (embed (flat ids) (flat gather) features) 2))
  (outputs [memory (encode x (flat pos) (flat mask)) 3]
           [image features 3]))

;; --- decoder: merged self + cross attention

(define (decoder-layer x i pos memory masks)
  (let ([p (prefix-fn "decoder" i)])
    (let* ([x (sandwich x p "pre_self_attn_layernorm.weight" "post_self_attn_layernorm.weight"
                (lambda (h)
                  (let* ([n (dim h 1)] [m (dim memory 1)]
                         [q (rope (rms-norm (ggml-reshape-3d (project h (p "self_attn.q_proj.weight")) head-dim heads n)
                                            (p "self_attn.q_norm.weight"))
                                  pos i)]
                         [k-self (rope (rms-norm (ggml-reshape-3d (project h (p "self_attn.k_proj.weight")) head-dim 1 n)
                                                 (p "self_attn.k_norm.weight"))
                                       pos i)]
                         ;; Cross keys: the same projection and norm, no rotary embedding.
                         [k-cross (rms-norm (project memory (p "self_attn.k_proj.weight")) (p "self_attn.k_norm.weight"))]
                         [k (ggml-concat (ggml-reshape-2d k-self head-dim n) k-cross 1)]           ; [hd, T + M]
                         [v (ggml-concat (project h (p "self_attn.v_proj.weight"))
                                         (project memory (p "self_attn.v_proj.weight")) 1)])
                    (project (attend q k v (if (sliding? i) (car masks) (cdr masks)))
                             (p "self_attn.o_proj.weight")))))])
      (sandwich x p "pre_feedforward_layernorm.weight" "post_feedforward_layernorm.weight"
                (lambda (h) (mlp h p))))))

(model decode (inputs [tokens i32 (_ 1)] [pos i32 (_ 1)] [memory f32 (640 _ 1)] [memory-mask f32 (_ 1)] [at i32 (1 1)])
  (define n (dim tokens 0))
  (define m (dim memory 1))
  (define memory2 (ggml-reshape-2d memory hidden m))
  (define x (ggml-scale (ggml-get-rows (weight "encoder.embed_tokens.weight") (flat tokens)) decoder-embed-scale))
  ;; Keys [self (T); memory (M)]: causal (and a 512 window on sliding layers)
  ;; over the decoder tokens, the memory's padding mask over the rest.
  (define cross (ggml-repeat-4d (ggml-reshape-2d (padding-mask (flat memory-mask)) m 1) m n 1 1))
  (define causal (distance-mask n n '((1.0 . 1.0))))
  (define windowed (distance-mask n n (list '(1.0 . 1.0) (cons -1.0 (inexact sliding-window)))))
  (define masks (cons (ggml-concat windowed cross 0) (ggml-concat causal cross 0)))
  (define hidden-states
    (tap "hidden"
      (rms-norm (let loop ([i 0] [x x])
                  (if (= i layers)
                      x
                      (loop (+ i 1) (tap (format "decoder.~2,'0d" i) (decoder-layer x i (flat pos) memory2 masks) 2))))
                "decoder.norm.weight")
      2))
  (define embeddings (weight "encoder.embed_tokens.weight"))   ; tied output projection
  (define all-logits (tap "all_logits" (ggml-mul-mat embeddings hidden-states) 2))
  (outputs [logits (ggml-mul-mat embeddings (ggml-get-rows hidden-states (flat at))) 2]))

;; --- KV cache
;;
;; One f16 key and one value cache per layer, [hd, 64 + 1024]: rows 0..63
;; hold the decoder's own tokens (written by decode-step at pos), rows 64..
;; the encoder output's (written once by cross). Attention then reads one
;; contiguous tensor, masked by [self-mask; memory-mask].

(define cache-rows (+ decode-length memory-capacity))
(define (cache kind i) (state (string-append kind "." (number->string i))))

(for-each (lambda (i)
            (define-state (string-append "k." (number->string i)) f16 head-dim cache-rows)
            (define-state (string-append "v." (number->string i)) f16 head-dim cache-rows))
          (iota layers))

;; m rows of a cache starting at row `from`: [hd, m].
(define (cache-view t from m)
  (let ([row (list-ref (strides t) 1)])
    (ggml-view-2d t head-dim m row (* from row))))

;; Cross-attention keys (k_norm, no rotary embedding) and values of the memory, per layer.
(model cross (inputs [memory f32 (640 _ 1)])
  (define m (dim memory 1))
  (define memory2 (ggml-reshape-2d memory hidden m))
  (for-each
    (lambda (i)
      (let ([p (prefix-fn "decoder" i)])
        (effect (ggml-cpy (rms-norm (project memory2 (p "self_attn.k_proj.weight")) (p "self_attn.k_norm.weight"))
                          (cache-view (cache "k" i) decode-length m)))
        (effect (ggml-cpy (project memory2 (p "self_attn.v_proj.weight"))
                          (cache-view (cache "v" i) decode-length m)))))
    (iota layers))
  (outputs))

(define (decoder-step-layer x i pos mask)
  (let ([p (prefix-fn "decoder" i)])
    (let* ([x (sandwich x p "pre_self_attn_layernorm.weight" "post_self_attn_layernorm.weight"
                (lambda (h)
                  (let* ([q (rope (rms-norm (ggml-reshape-3d (project h (p "self_attn.q_proj.weight")) head-dim heads 1)
                                            (p "self_attn.q_norm.weight"))
                                  pos i)]
                         [k (rope (rms-norm (ggml-reshape-3d (project h (p "self_attn.k_proj.weight")) head-dim 1 1)
                                            (p "self_attn.k_norm.weight"))
                                  pos i)]
                         [v (project h (p "self_attn.v_proj.weight"))]
                         [k-cache (cache "k" i)]
                         [v-cache (cache "v" i)])
                    ;; This token's K/V go into the cache at pos before attention reads it.
                    (effect (ggml-set-rows k-cache (ggml-reshape-2d k head-dim 1) pos))
                    (effect (ggml-set-rows v-cache v pos))
                    ;; One query: [hd, heads, 1] is [hd, 1, heads] in memory.
                    (project (ggml-reshape-2d
                               (ggml-flash-attn-ext (ggml-reshape-3d q head-dim 1 heads)
                                                    (ggml-reshape-3d k-cache head-dim cache-rows 1)
                                                    (ggml-reshape-3d v-cache head-dim cache-rows 1)
                                                    mask (/ 1.0 (sqrt (inexact head-dim))) 0.0 0.0)
                               (* head-dim heads) 1)
                             (p "self_attn.o_proj.weight")))))])
      (sandwich x p "pre_feedforward_layernorm.weight" "post_feedforward_layernorm.weight"
                (lambda (h) (mlp h p))))))

;; self-mask: 1 for cached positions <= pos (0 after); memory-mask: 1 for
;; the memory's valid rows, 0 for padding and unused cache rows. The self
;; part is at most 64 long, so the 512 window never applies.
(model decode-step (inputs [token i32 (1 1)] [pos i32 (1 1)] [self-mask f32 (64 1)] [memory-mask f32 (1024 1)])
  (define checked
    (unless (and (= (dim self-mask 0) decode-length) (= (dim memory-mask 0) memory-capacity))
      (error 'decode-step "mask sizes must match the cache capacities" (list decode-length memory-capacity))))
  (define x (ggml-scale (ggml-get-rows (weight "encoder.embed_tokens.weight") (flat token)) decoder-embed-scale))
  (define mask (ggml-cast (ggml-reshape-2d (ggml-concat (padding-mask (flat self-mask)) (padding-mask (flat memory-mask)) 0)
                                           cache-rows 1)
                          GGML_TYPE_F16))
  (define hidden-states
    (tap "hidden"
      (rms-norm (let loop ([i 0] [x x])
                  (if (= i layers) x (loop (+ i 1) (decoder-step-layer x i (flat pos) mask))))
                "decoder.norm.weight")
      2))
  (define logits (ggml-mul-mat (weight "encoder.embed_tokens.weight") hidden-states))
  ;; The greedy choice computed on the device: one number to read back per token.
  (outputs [logits logits 2] [next (ggml-argmax logits) 1]))

;; --- pipelines: prompt expansion, tokenization and greedy decoding

(define tok (tokenizer (asset "tokenizer.json")))
(define bos (token-id tok "<bos>"))
(define eos (token-id tok "<eos>"))
(define image-token (token-id tok "<image_soft_token>"))
(define eoi-token (token-id tok "<end_of_image>"))

(define (string-repeat s n) (if (= n 0) "" (string-append s (string-repeat s (- n 1)))))

;; Every occurrence of `from` in s replaced by `to`.
(define (string-replace s from to)
  (let ([n (string-length s)] [m (string-length from)])
    (let loop ([i 0] [start 0] [parts '()])
      (cond
        [(> (+ i m) n) (apply string-append (reverse (cons (substring s start n) parts)))]
        [(string=? (substring s i (+ i m)) from)
         (loop (+ i m) (+ i m) (cons to (cons (substring s start i) parts)))]
        [else (loop (+ i 1) start parts)]))))

;; Gemma3Processor: "<start_of_image>" becomes the image's token sequence.
(define (expand-image prompt)
  (string-replace prompt "<start_of_image>"
                  (string-append "\n\n<start_of_image>" (string-repeat "<image_soft_token>" image-tokens)
                                 "<end_of_image>\n\n")))

(define (whole xs) (map (lambda (x) (exact (round x))) xs))

;; Encoder inputs for a prompt: ids, gather, pos, mask (host arrays).
(define (encoder-inputs prompt n-image)
  (let-values ([(ids mask) (tokenize tok prompt)])
    (let* ([id-list (whole (array->list ids))]
           [n (length id-list)]
           [gather (let loop ([ids id-list] [i 0] [k 0] [acc '()])
                     (cond
                       [(null? ids) (reverse acc)]
                       [(and (= (car ids) image-token) (< k n-image))
                        (loop (cdr ids) (+ i 1) (+ k 1) (cons (+ n k) acc))]
                       [(= (car ids) eoi-token) (loop (cdr ids) (+ i 1) k (cons (+ n n-image) acc))]
                       [else (loop (cdr ids) (+ i 1) k (cons i acc))]))])
      (values ids (list->array gather) (list->array (iota n)) mask))))

(define (pad-to xs n value) (append xs (make-list (- n (length xs)) value)))
(define (make-list n x) (if (<= n 0) '() (cons x (make-list (- n 1) x))))

;; Greedy decoding from <bos> until <eos> or max-new tokens.
(define (greedy memory memory-mask max-new)
  (if use-kv-cache
      (greedy-cached memory memory-mask max-new)
      (greedy-uncached memory memory-mask max-new)))

(define (argmax-token r) (car (whole (array->list (array-argmax (output r 'logits))))))

;; One decode-step per token after filling the cross cache.
(define (greedy-cached memory memory-mask max-new)
  (let ([m (car (array-shape memory-mask))]
        [limit (min decode-length (+ max-new 1))])
    (when (> m memory-capacity)
      (error 'generate (format "the prompt is ~a tokens, the cache holds ~a" m memory-capacity)))
    (run cross [memory memory])
    (let ([memory-mask (list->array (pad-to (array->list memory-mask) memory-capacity 0))])
      (let loop ([tokens (list bos)] [pos 0])
        (if (or (= (length tokens) limit) (= (car (last-pair* tokens)) eos))
            tokens
            (let ([r (run decode-step [token (list->array (list (car (last-pair* tokens))))]
                                      [pos (list->array (list pos))]
                                      [self-mask (list->array (pad-to (make-list (+ pos 1) 1) decode-length 0))]
                                      [memory-mask memory-mask])])
              (loop (append tokens (list (car (whole (array->list (output r 'next)))))) (+ pos 1))))))))

;; The whole decoder per token (no cache).
(define (greedy-uncached memory memory-mask max-new)
  (let ([limit (min decode-length (+ max-new 1))]
        [pos (list->array (iota decode-length))])
    (let loop ([tokens (list bos)])
      (if (or (= (length tokens) limit) (= (car (last-pair* tokens)) eos))
          tokens
          (let* ([r (run decode [tokens (list->array (pad-to tokens decode-length 0))] [pos pos]
                              [memory memory] [memory-mask memory-mask] [at (list->array (list (- (length tokens) 1)))])]
                 [next (argmax-token r)])
            (loop (append tokens (list next))))))))

(define (last-pair* xs) (if (null? (cdr xs)) xs (last-pair* (cdr xs))))

(define (generation-results tokens)
  (results [text (detokenize tok tokens)] [tokens tokens]))

(pipeline caption ([image image] [prompt string])
  (let-values ([(ids gather pos mask) (encoder-inputs (expand-image prompt) image-tokens)])
    (let* ([pixels (image->array (image-resize image image-size image-size 'bilinear)
                                 'mean '(0.5 0.5 0.5) 'std '(0.5 0.5 0.5))]
           [memory (output (run encode-image [pixels pixels] [ids ids] [gather gather] [pos pos] [mask mask]) 'memory)])
      (generation-results (greedy memory mask 32)))))

(pipeline generate ([prompt string])
  (let-values ([(ids gather pos mask) (encoder-inputs prompt 0)])
    (let ([memory (output (run encode [ids ids] [gather gather] [pos pos] [mask mask]) 'memory)])
      (generation-results (greedy memory mask 32)))))
