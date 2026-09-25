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

(import (tl tensor) (tl nn) (tl attn) (tl vision) (tl util))

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

(define dim tensor:dim)

;; Gemma's RMSNorm scales by (1 + weight); the file stores 1 + weight
;; (converted with --offset, see README), which saves an op per norm and lets
;; backends fuse the norm with the multiplication.
(define (rms-norm x prefix) (nn:rms-norm x prefix 'eps eps))

;; gelu_pytorch_tanh
(define (gelu x) (if exact-gelu (nn:gelu-tanh-exact x) (nn:gelu-tanh x)))

;; Local layers: theta 1e4. Global layers: theta 1e6, linear scaling by 8.
(define (rope x pos layer)
  (if (sliding? layer)
      (attn:rope x pos 'base 10000.0)
      (attn:rope x pos 'base 1000000.0 'scale 0.125)))

;; Query heads [hd, heads, T] (after RoPE) -> [hd, T, heads] for attn:sdpa;
;; single keys/values [hd, T] (one K/V head, multi-query) -> [hd, T, 1, 1].
(define (query-heads q) (ggml-permute q 0 2 1 3))
(define (one-head t) (ggml-reshape-4d t head-dim (dim t 1) 1 1))

(define (mlp x p)
  (nn:linear (ggml-mul (gelu (nn:linear x (p "mlp.gate_proj"))) (nn:linear x (p "mlp.up_proj")))
             (p "mlp.down_proj")))

;; Gemma 2/3 blocks: norms before and after each sublayer, then the residual.
(define (sandwich x p norm-pre norm-post f)
  (ggml-add x (rms-norm (f (rms-norm x (p norm-pre))) (p norm-post))))

(define (prefix-fn stack i)
  (lambda (name) (format "~a.layers.~a.~a" stack i name)))

;; q [hd, heads, T] and k [hd, 1, T] of a layer, normed and rotated.
(define (query+key h p pos i)
  (let ([n (dim h 1)])
    (values (rope (rms-norm (ggml-reshape-3d (nn:linear h (p "self_attn.q_proj")) head-dim heads n) (p "self_attn.q_norm")) pos i)
            (rope (rms-norm (ggml-reshape-3d (nn:linear h (p "self_attn.k_proj")) head-dim 1 n) (p "self_attn.k_norm")) pos i))))

;; --- text embeddings

;; Token embeddings are scaled by sqrt(640). transformers (5.x) keeps the
;; encoder's factor in the checkpoint's bf16 (25.25) but the decoder's in f32;
;; the port does the same.
(define embed-scale 25.25)
(define decoder-embed-scale (sqrt (inexact hidden)))

;; Token embeddings [D, L], with image features [D, 256] (or #f) and the
;; end-of-image embedding placed by gather.
(define (embed ids gather image-features)
  (let* ([tokens (ggml-reshape-2d (ggml-scale (nn:embedding "encoder.embed_tokens.weight" ids) embed-scale) hidden (dim ids 0))]
         [eoi (ggml-reshape-2d (weight "encoder.embed_tokens.eoi_embedding") hidden 1)])
    (ggml-get-rows (tensor:concat (if image-features (list tokens image-features eoi) (list tokens eoi)) 1) gather)))

;; --- encoder

(define (encoder-layer x i pos masks)
  (let ([p (prefix-fn "encoder" i)])
    (let ([x (sandwich x p "pre_self_attn_layernorm" "post_self_attn_layernorm"
               (lambda (h)
                 (let-values ([(q k) (query+key h p pos i)])
                   (nn:linear (attn:sdpa (query-heads q) (one-head (ggml-reshape-2d k head-dim (dim h 1)))
                                         (one-head (nn:linear h (p "self_attn.v_proj")))
                                         'mask (if (sliding? i) (car masks) (cdr masks)))
                              (p "self_attn.o_proj")))))])
      (sandwich x p "pre_feedforward_layernorm" "post_feedforward_layernorm" (lambda (h) (mlp h p))))))

;; Bidirectional; sliding layers see keys 255 before to 256 after the query
;; (HF: window (w+1)/2 left, w/2 + 1 right).
(define (encode x pos mask)
  (let* ([n (dim x 1)]
         [pad (attn:padding-mask mask n)]
         [local (ggml-add (attn:window-mask n 'left (quotient (+ sliding-window 1) 2)
                                              'right (+ (quotient sliding-window 2) 1))
                          pad)])
    (let loop ([i 0] [x x])
      (if (= i layers)
          (rms-norm x "encoder.norm")
          (loop (+ i 1) (tap (format "encoder.~2,'0d" i) (encoder-layer x i pos (cons local pad)) 2))))))

;; --- vision tower (SigLIP) and projector

(define (vision-layer x i)
  (let* ([p (prefix-fn "vision.encoder" i)]
         [heads (lambda (name) (attn:split-heads (nn:linear (nn:layer-norm x (p "layer_norm1") 'eps eps) (p name)) vision-heads))]
         [attention (attn:sdpa (heads "self_attn.q_proj") (heads "self_attn.k_proj") (heads "self_attn.v_proj")
                               'flash flash-attention)]
         [x (ggml-add x (nn:linear attention (p "self_attn.out_proj")))])
    (ggml-add x (nn:linear (gelu (nn:linear (nn:layer-norm x (p "layer_norm2") 'eps eps) (p "mlp.fc1"))) (p "mlp.fc2")))))

;; pixels [896, 896, 3, 1] -> image features [640, 256]
(define (image-features pixels)
  (let* ([grid (quotient image-size patch)]                                         ; 64
         [x (ggml-add (nn:patch-embed pixels "vision.embeddings.patch_embedding" patch)
                      (tensor:as-f32 (weight "vision.embeddings.position_embedding.weight")))]  ; [1152, 4096]
         [x (let loop ([i 0] [x x])
              (if (= i vision-layers) x (loop (+ i 1) (vision-layer x i))))]
         [x (tap "vision" (ggml-reshape-2d (nn:layer-norm x "vision.post_layernorm" 'eps eps) vision-width (* grid grid)) 2)]
         ;; 4 x 4 average pooling of the 64 x 64 grid, row-major like the tokens.
         [side (exact (round (sqrt image-tokens)))]
         [pooled (vision:grid->tokens (nn:avg-pool (vision:tokens->grid x grid grid) (quotient grid side)))]  ; [1152, 256]
         [normed (rms-norm pooled "mm.mm_soft_emb_norm")])
    ;; mm_input_projection_weight is (in 1152, out 640) in torch: [640, 1152] here.
    (ggml-mul-mat (ggml-cont (ggml-transpose (weight "mm.mm_input_projection_weight"))) normed)))

(define flat tensor:flatten)

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
    (let ([x (sandwich x p "pre_self_attn_layernorm" "post_self_attn_layernorm"
               (lambda (h)
                 (let*-values ([(n) (dim h 1)]
                               [(q k-self) (query+key h p pos i)])
                   ;; Keys: the decoder's own (rotated), then the memory's (same
                   ;; projection and norm, no rotary embedding).
                   (let ([k (tensor:concat (list (ggml-reshape-2d k-self head-dim n)
                                                 (rms-norm (nn:linear memory (p "self_attn.k_proj")) (p "self_attn.k_norm")))
                                           1)]
                         [v (tensor:concat (list (nn:linear h (p "self_attn.v_proj")) (nn:linear memory (p "self_attn.v_proj"))) 1)])
                     (nn:linear (attn:sdpa (query-heads q) (one-head k) (one-head v) 'mask (if (sliding? i) (car masks) (cdr masks)))
                                (p "self_attn.o_proj"))))))])
      (sandwich x p "pre_feedforward_layernorm" "post_feedforward_layernorm" (lambda (h) (mlp h p))))))

(define (decoder-embed tokens)
  (ggml-reshape-2d (ggml-scale (nn:embedding "encoder.embed_tokens.weight" tokens) decoder-embed-scale) hidden (dim tokens 0)))

(model decode (inputs [tokens i32 (_ 1)] [pos i32 (_ 1)] [memory f32 (640 _ 1)] [memory-mask f32 (_ 1)] [at i32 (1 1)])
  (define n (dim tokens 0))
  (define m (dim memory 1))
  (define memory2 (ggml-reshape-2d memory hidden m))
  (define x (decoder-embed (flat tokens)))
  ;; Keys [self (T); memory (M)]: causal (and a 512 window on sliding layers)
  ;; over the decoder tokens, the memory's padding mask over the rest.
  (define cross (attn:padding-mask (flat memory-mask) n))
  (define masks (cons (tensor:concat (list (attn:causal-mask n 'window sliding-window) cross) 0)
                      (tensor:concat (list (attn:causal-mask n) cross) 0)))
  (define hidden-states
    (tap "hidden"
      (rms-norm (let loop ([i 0] [x x])
                  (if (= i layers)
                      x
                      (loop (+ i 1) (tap (format "decoder.~2,'0d" i) (decoder-layer x i (flat pos) memory2 masks) 2))))
                "decoder.norm")
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
(define (cache kind i) (state (format "~a.~a" kind i)))

(for-each (lambda (i)
            (define-state (format "k.~a" i) f16 head-dim cache-rows)
            (define-state (format "v.~a" i) f16 head-dim cache-rows))
          (iota layers))

;; Cross-attention keys (k_norm, no rotary embedding) and values of the memory, per layer.
(model cross (inputs [memory f32 (640 _ 1)])
  (define m (dim memory 1))
  (define memory2 (ggml-reshape-2d memory hidden m))
  (for-each
    (lambda (i)
      (let ([p (prefix-fn "decoder" i)])
        (effect (ggml-cpy (rms-norm (nn:linear memory2 (p "self_attn.k_proj")) (p "self_attn.k_norm"))
                          (tensor:slice (cache "k" i) 1 decode-length m)))
        (effect (ggml-cpy (nn:linear memory2 (p "self_attn.v_proj"))
                          (tensor:slice (cache "v" i) 1 decode-length m)))))
    (iota layers))
  (outputs))

(define (decoder-step-layer x i pos mask)
  (let ([p (prefix-fn "decoder" i)])
    (let ([x (sandwich x p "pre_self_attn_layernorm" "post_self_attn_layernorm"
               (lambda (h)
                 (let-values ([(q k) (query+key h p pos i)])
                   (let ([k-cache (cache "k" i)] [v-cache (cache "v" i)])
                     ;; This token's K/V go into the cache at pos before attention reads it.
                     (effect (ggml-set-rows k-cache (ggml-reshape-2d k head-dim 1) pos))
                     (effect (ggml-set-rows v-cache (nn:linear h (p "self_attn.v_proj")) pos))
                     ;; One query: [hd, heads, 1] is [hd, 1, heads] in memory.
                     (nn:linear (attn:sdpa (ggml-reshape-4d q head-dim 1 heads 1) (one-head k-cache) (one-head v-cache)
                                           'mask mask 'flash #t)
                                (p "self_attn.o_proj"))))))])
      (sandwich x p "pre_feedforward_layernorm" "post_feedforward_layernorm" (lambda (h) (mlp h p))))))

;; self-mask: 1 for cached positions <= pos (0 after); memory-mask: 1 for
;; the memory's valid rows, 0 for padding and unused cache rows. The self
;; part is at most 64 long, so the 512 window never applies.
(model decode-step (inputs [token i32 (1 1)] [pos i32 (1 1)] [self-mask f32 (64 1)] [memory-mask f32 (1024 1)])
  (define checked
    (unless (and (= (dim self-mask 0) decode-length) (= (dim memory-mask 0) memory-capacity))
      (error 'decode-step "mask sizes must match the cache capacities" (list decode-length memory-capacity))))
  (define x (decoder-embed (flat token)))
  (define mask (tensor:concat (list (attn:padding-mask (flat self-mask) 1) (attn:padding-mask (flat memory-mask) 1)) 0))
  (define hidden-states
    (tap "hidden"
      (rms-norm (let loop ([i 0] [x x])
                  (if (= i layers) x (loop (+ i 1) (decoder-step-layer x i (flat pos) mask))))
                "decoder.norm")
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

;; Gemma3Processor: "<start_of_image>" becomes the image's token sequence.
(define (expand-image prompt)
  (util:string-replace prompt "<start_of_image>"
                       (string-append "\n\n<start_of_image>" (util:string-repeat "<image_soft_token>" image-tokens)
                                      "<end_of_image>\n\n")))

;; Encoder inputs for a prompt: ids, gather, pos, mask (host arrays).
(define (encoder-inputs prompt n-image)
  (let-values ([(ids mask) (tokenize tok prompt)])
    (let* ([id-list (util:->integers (array->list ids))]
           [n (length id-list)]
           [gather (let loop ([ids id-list] [i 0] [k 0] [acc '()])
                     (cond
                       [(null? ids) (reverse acc)]
                       [(and (= (car ids) image-token) (< k n-image))
                        (loop (cdr ids) (+ i 1) (+ k 1) (cons (+ n k) acc))]
                       [(= (car ids) eoi-token) (loop (cdr ids) (+ i 1) k (cons (+ n n-image) acc))]
                       [else (loop (cdr ids) (+ i 1) k (cons i acc))]))])
      (values ids (list->array gather) (list->array (iota n)) mask))))

;; Greedy decoding from <bos> until <eos> or max-new tokens.
(define (greedy memory memory-mask max-new)
  (if use-kv-cache
      (greedy-cached memory memory-mask max-new)
      (greedy-uncached memory memory-mask max-new)))

(define (done? tokens limit) (or (= (length tokens) limit) (= (util:last tokens) eos)))

;; One decode-step per token after filling the cross cache.
(define (greedy-cached memory memory-mask max-new)
  (let ([m (car (array-shape memory-mask))]
        [limit (min decode-length (+ max-new 1))])
    (when (> m memory-capacity)
      (error 'generate (format "the prompt is ~a tokens, the cache holds ~a" m memory-capacity)))
    (run cross [memory memory])
    (let ([memory-mask (list->array (util:pad-list (array->list memory-mask) memory-capacity 0))])
      (let loop ([tokens (list bos)] [pos 0])
        (if (done? tokens limit)
            tokens
            (let ([r (run decode-step [token (list->array (list (util:last tokens)))]
                                      [pos (list->array (list pos))]
                                      [self-mask (list->array (util:pad-list (util:make-list (+ pos 1) 1) decode-length 0))]
                                      [memory-mask memory-mask])])
              (loop (append tokens (util:->integers (array->list (output r 'next)))) (+ pos 1))))))))

;; The whole decoder per token (no cache).
(define (greedy-uncached memory memory-mask max-new)
  (let ([limit (min decode-length (+ max-new 1))]
        [pos (list->array (iota decode-length))])
    (let loop ([tokens (list bos)])
      (if (done? tokens limit)
          tokens
          (let ([r (run decode [tokens (list->array (util:pad-list tokens decode-length 0))] [pos pos]
                              [memory memory] [memory-mask memory-mask] [at (list->array (list (- (length tokens) 1)))])])
            (loop (append tokens (util:->integers (array->list (array-argmax (output r 'logits)))))))))))

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
