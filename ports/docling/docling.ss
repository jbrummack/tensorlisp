;; Granite Docling 258M (ibm-granite/granite-docling-258M): an Idefics3 VLM --
;; a SigLIP-style vision tower, a pixel-shuffle connector, and a plain
;; Llama-style (RoPE, GQA, SwiGLU, RMSNorm) causal decoder -- ported from
;; transformers' modeling_idefics3.py.
;;
;; Unlike T5Gemma2 this decoder has NO cross-attention: image features are
;; spliced into the token embedding sequence (like T5Gemma2's gather trick),
;; then the whole sequence runs through one plain causal decoder stack.
;;
;; Entries (shapes in numpy order; batch 1):
;;   prefill-image  pixels [1, N, 3, 512, 512] (N tiles, row-major grid then
;;                  the global view -- see autopro's image-tile), ids [1, L]
;;                  i32, gather [1, L] i32, pos [1, L] i32
;;                  -> logits [1, vocab] at the last position
;;                  Fills the self-attention KV cache for all L positions.
;;   prefill-text   ids [1, L] i32, pos [1, L] i32 -> logits [1, vocab]
;;                  Text-only (no image), same cache-filling behavior.
;;   decode-step    token [1, 1] i32, pos [1, 1] i32, mask [1, cache-rows] f32
;;                  -> logits [1, vocab], next [1] (argmax)
;; Pipelines (raw inputs):
;;   caption  image + prompt (with "<image>") -> text, tokens
;;   generate prompt -> text, tokens
;;
;; gather places image-feature rows (N * 64, tile grid row-major then the
;; global view) at the prompt's "<image>" placeholder positions; every other
;; token (including the row/col and fake-image-token markers) is a normal
;; vocabulary embedding.
;;
;; KV cache: one f16 key and one value tensor per (layer, kv-head), written
;; a whole prefill's worth of positions at once (prefill-image/-text) or one
;; position at a time (decode-step), read back over the whole cache each
;; step. Shapes are fixed by decode-length, so decode-step's graph builds once.
;;
;; Shapes in comments are ggml order (innermost first).

(import (tl tensor) (tl nn) (tl attn) (tl vision) (tl util))

;; --- vision tower (SigLIP-style, no CLS/register token) ---
(define vision-hidden 768)
(define vision-heads 12)
(define vision-layers 12)
(define image-size 512)
(define patch 16)
(define vision-grid (quotient image-size patch))          ; 32
(define vision-eps 1e-6)

;; --- pixel-shuffle connector ---
(define scale-factor 4)
(define tile-tokens (quotient (* vision-grid vision-grid) (* scale-factor scale-factor)))  ; 64

;; --- decoder (plain Llama: RoPE, GQA, SwiGLU, RMSNorm) ---
(define hidden 576)
(define heads 9)
(define kv-heads 3)
(define head-dim 64)
(define layers 30)
(define rms-eps 1e-5)
(define rope-base 100000.0)
(define vocab 100352)

;; ggml's GELU uses an f16 lookup table on the CPU (~1e-3); #t computes the
;; tanh approximation exactly from ops (for comparisons with PyTorch).
(define exact-gelu #t)
;; Fused attention kernel (f16 K/V): faster on GPUs, slightly less exact.
(define flash-attention #t)

;; Decoder length: the KV cache holds this many total positions (image tokens
;; + prompt + generated). A tiled page can have many image tokens (17 tiles x
;; 64 = 1088 for a typical page), so this needs headroom.
(define decode-length 2048)

(define dim tensor:dim)
(define flat tensor:flatten)

(define (gelu x) (if exact-gelu (nn:gelu-tanh-exact x) (nn:gelu-tanh x)))

(define (prefix-fn stack i) (lambda (name) (format "~a.layers.~a.~a" stack i name)))

;; --- vision tower ---

(define (vision-layer x i)
  (let* ([p (prefix-fn "vision.encoder" i)]
         [heads (lambda (name) (attn:split-heads (nn:linear (nn:layer-norm x (p "layer_norm1") 'eps vision-eps) (p name)) vision-heads))]
         [attention (attn:sdpa (heads "self_attn.q_proj") (heads "self_attn.k_proj") (heads "self_attn.v_proj")
                               'flash flash-attention)]
         [x (ggml-add x (nn:linear attention (p "self_attn.out_proj")))])
    (ggml-add x (nn:linear (gelu (nn:linear (nn:layer-norm x (p "layer_norm2") 'eps vision-eps) (p "mlp.fc1"))) (p "mlp.fc2")))))

;; Idefics3Connector.pixel_shuffle, translated to ggml's innermost-first axes.
;; Torch (row-major, batch n): view(n,h,w,C) -> view(n,h,w/sf,C*sf) ->
;; permute(0,2,1,3) -> reshape(n,w/sf,h/sf,C*sf*sf) -> permute(0,2,1,3) ->
;; reshape(n,seq/sf/sf,C*sf*sf). ggml's axis i = torch's axis (rank-1-i), and
;; ggml_permute(a,ax0,ax1,ax2,ax3) moves original axis i to position axi -- so
;; each torch permute(0,2,1,3) (swap axes 1 and 2) is (ggml-permute _ 0 2 1 3)
;; here too (axis 0 = embed and axis 3 = batch never move).
(define (pixel-shuffle grid w h n)
  (let* ([x (ggml-reshape-4d grid (* vision-hidden scale-factor) (quotient w scale-factor) h n)]
         [x (ggml-cont (ggml-permute x 0 2 1 3))]
         [x (ggml-reshape-4d x (* vision-hidden scale-factor scale-factor) (quotient h scale-factor) (quotient w scale-factor) n)]
         [x (ggml-cont (ggml-permute x 0 2 1 3))])
    (ggml-reshape-2d x (* vision-hidden scale-factor scale-factor) (quotient (* w h n) (* scale-factor scale-factor)))))

;; pixels [512, 512, 3, n] (n tiles, row-major grid then the global view) ->
;; image features [576, 64 * n], in the same tile order (matches the prompt).
(define (image-features pixels)
  (let* ([n (dim pixels 3)]
         [x (ggml-add (nn:patch-embed pixels "vision.embeddings.patch_embedding" patch)
                      (tensor:as-f32 (weight "vision.embeddings.position_embedding.weight")))]
         [x (let loop ([i 0] [x x]) (if (= i vision-layers) x (loop (+ i 1) (vision-layer x i))))]
         [x (nn:layer-norm x "vision.post_layernorm" 'eps vision-eps)]
         [grid (ggml-reshape-4d x vision-hidden vision-grid vision-grid n)]
         [shuffled (tap "pixel_shuffle" (pixel-shuffle grid vision-grid vision-grid n) 2)])
    (nn:linear shuffled "mm.proj")))

;; --- text embeddings: token embeddings with image features spliced in ---

;; With image features, gather splices them in at the "<image>" placeholder
;; positions (index >= the token count). Without, ids directly index the
;; embedding table -- no gather indirection needed.
(define (embed ids gather image-feats)
  (let ([tokens (ggml-reshape-2d (nn:embedding "decoder.embed_tokens.weight" ids) hidden (dim ids 0))])
    (if image-feats (ggml-get-rows (tensor:concat (list tokens image-feats) 1) gather) tokens)))

;; --- decoder: plain pre-norm Llama block ---

(define (rms-norm x prefix) (nn:rms-norm x prefix 'eps rms-eps))

(define (mlp x p)
  (nn:linear (ggml-mul (ggml-silu (nn:linear x (p "mlp.gate_proj"))) (nn:linear x (p "mlp.up_proj")))
             (p "mlp.down_proj")))

;; q/k [hd, heads-or-kv-heads, L] (before split-heads), roped.
(define (query+key h p pos)
  (let ([n (dim h 1)])
    (values (attn:rope (ggml-reshape-3d (nn:linear h (p "self_attn.q_proj")) head-dim heads n) pos 'base rope-base)
            (attn:rope (ggml-reshape-3d (nn:linear h (p "self_attn.k_proj")) head-dim kv-heads n) pos 'base rope-base))))

;; [hd, X, L] (after rope, or v straight from its projection) -> [hd, L, X, 1],
;; attn:sdpa's (and ggml-set-rows') expected layout -- also directly usable as
;; the cache-write source, since the cache itself is [hd, decode-length, X].
(define (as-heads x count n) (ggml-cont (ggml-permute (ggml-reshape-4d x head-dim count n 1) 0 2 1 3)))

;; --- KV cache: one (layer, k-or-v) state tensor covering all kv-heads at
;; once, [hd, decode-length, kv-heads]. attn:sdpa's native GQA support (q has
;; `heads`, k/v have `kv-heads`, consecutive query heads share one kv-head
;; like HF repeat_kv) means one attn:sdpa call handles every head -- no
;; per-head loop, no concatenating partial outputs back together.

(define (cache kind i) (state (format "~a.~a" kind i)))

(for-each (lambda (i)
            (define-state (format "k.~a" i) f16 head-dim decode-length kv-heads)
            (define-state (format "v.~a" i) f16 head-dim decode-length kv-heads))
          (iota layers))

;; Writes k/v [hd, n, kv-heads, 1] (this call's new rows, every kv-head at
;; once) into the cache at positions `pos` (length n).
(define (write-cache! i k v pos)
  (effect (ggml-set-rows (cache "k" i) k pos))
  (effect (ggml-set-rows (cache "v" i) v pos)))

(define (qkv h p pos)
  (let-values ([(q k) (query+key h p pos)])
    (let* ([n (dim h 1)]
           [v (ggml-reshape-3d (nn:linear h (p "self_attn.v_proj")) head-dim kv-heads n)])
      (values (as-heads q heads n) (as-heads k kv-heads n) (as-heads v kv-heads n)))))

;; Prefill: full causal attention over the freshly-computed positions
;; (doesn't need the persistent cache back yet), and writes them into it for
;; decode-step's benefit.
(define (prefill-attention x p pos mask i)
  (let* ([h (rms-norm x (p "input_layernorm"))])
    (let-values ([(q k v) (qkv h p pos)])
      (write-cache! i k v pos)
      (nn:linear (attn:sdpa q k v 'mask mask 'flash flash-attention) (p "self_attn.o_proj")))))

;; decode-step: writes this token's K/V at `pos`, then reads the whole
;; persistent cache back (masked to the valid prefix) for attention.
(define (decode-attention x p pos mask i)
  (let* ([h (rms-norm x (p "input_layernorm"))])
    (let-values ([(q k v) (qkv h p pos)])
      (write-cache! i k v pos)
      (let ([k-cache (ggml-reshape-4d (cache "k" i) head-dim decode-length kv-heads 1)]
            [v-cache (ggml-reshape-4d (cache "v" i) head-dim decode-length kv-heads 1)])
        (nn:linear (attn:sdpa q k-cache v-cache 'mask mask 'flash flash-attention) (p "self_attn.o_proj"))))))

(define (decoder-layer x i pos mask attention)
  (let* ([p (prefix-fn "decoder" i)]
         [x (ggml-add x (attention x p pos mask i))])
    (ggml-add x (mlp (rms-norm x (p "post_attention_layernorm")) p))))

;; ids [L] with image features already gathered in; pos [L]; mask [L, L]
;; (causal). Returns hidden states [hidden, L].
(define (prefill x pos mask)
  (rms-norm
    (let loop ([i 0] [x x]) (if (= i layers) x (loop (+ i 1) (decoder-layer x i pos mask prefill-attention))))
    "decoder.norm"))

(model prefill-image (inputs [pixels f32 (512 512 3 _)] [ids i32 (_ 1)] [gather i32 (_ 1)] [pos i32 (_ 1)] [at i32 (1 1)])
  (define feats (tap "image_features" (image-features pixels) 2))
  (define x (tap "embeddings" (embed (flat ids) (flat gather) feats) 2))
  (define n (dim x 1))
  (define hidden-states (tap "hidden" (prefill x (flat pos) (attn:causal-mask n)) 2))
  (define embeddings (weight "decoder.embed_tokens.weight"))
  (outputs [logits (ggml-mul-mat embeddings (ggml-get-rows hidden-states (flat at))) 2]))

(model prefill-text (inputs [ids i32 (_ 1)] [pos i32 (_ 1)] [at i32 (1 1)])
  (define x (tap "embeddings" (embed (flat ids) #f #f) 2))
  (define n (dim x 1))
  (define hidden-states (prefill x (flat pos) (attn:causal-mask n)))
  (define embeddings (weight "decoder.embed_tokens.weight"))
  (outputs [logits (ggml-mul-mat embeddings (ggml-get-rows hidden-states (flat at))) 2]))

;; --- decode-step: one token, reading the whole cache ---

(model decode-step (inputs [token i32 (1 1)] [pos i32 (1 1)] [mask f32 (2048 1)])
  (define checked (unless (= (dim mask 0) decode-length) (error 'decode-step "mask size must match decode-length" decode-length)))
  (define x (ggml-reshape-2d (nn:embedding "decoder.embed_tokens.weight" (flat token)) hidden 1))
  (define pos1 (flat pos))
  (define mask1 (attn:padding-mask (flat mask) 1))
  (define hidden-states
    (rms-norm (let loop ([i 0] [x x]) (if (= i layers) x (loop (+ i 1) (decoder-layer x i pos1 mask1 decode-attention))))
              "decoder.norm"))
  (define logits (ggml-mul-mat (weight "decoder.embed_tokens.weight") hidden-states))
  (outputs [logits logits 2] [next (ggml-argmax logits) 1]))

;; --- pipelines: chat template, tile-prompt expansion, tokenization, greedy decode ---

(define tok (tokenizer (asset "tokenizer.json")))
(define bos (token-id tok "<|start_of_role|>"))
(define eos (token-id tok "<|end_of_text|>"))
(define image-token (token-id tok "<image>"))
(define fake-token (token-id tok "<fake_token_around_image>"))
(define global-img (token-id tok "<global-img>"))

;; The single-turn chat template: <|start_of_role|>user<|end_of_role|><image>
;; PROMPT<|end_of_text|><|start_of_role|>assistant<|end_of_role|>.
(define (apply-chat-template prompt)
  (string-append "<|start_of_role|>user<|end_of_role|><image>" prompt "<|end_of_text|><|start_of_role|>assistant<|end_of_role|>"))

;; One tile's block: <fake_token_around_image><row_R_col_C> + 64x<image>, or,
;; for the trailing global view, <fake_token_around_image><global-img> +
;; 64x<image><fake_token_around_image>.
(define (expand-image prompt rows cols)
  (define (tile-block r c) (format "<fake_token_around_image><row_~a_col_~a>~a" r c (util:string-repeat "<image>" tile-tokens)))
  (define grid
    (if (> rows 0)
        (string-append
          (util:string-join (map (lambda (r) (util:string-join (map (lambda (c) (tile-block (+ r 1) (+ c 1))) (iota cols)) "")) (iota rows)) "\n")
          "\n")
        ""))
  (define global-block (format "<fake_token_around_image><global-img>~a<fake_token_around_image>" (util:string-repeat "<image>" tile-tokens)))
  (util:string-replace prompt "<image>" (string-append grid global-block)))

;; ids, gather (image features 0..n-image-tokens-1 then token embeddings),
;; pos, for a fully-built prompt string.
(define (prompt-inputs prompt n-image-tokens)
  (let-values ([(ids mask) (tokenize tok prompt)])
    (let* ([id-list (util:->integers (array->list ids))]
           [n (length id-list)]
           [gather (let loop ([ids id-list] [i 0] [k 0] [acc '()])
                     (cond
                       [(null? ids) (reverse acc)]
                       [(and (= (car ids) image-token) (< k n-image-tokens)) (loop (cdr ids) (+ i 1) (+ k 1) (cons (+ n k) acc))]
                       [else (loop (cdr ids) (+ i 1) k (cons i acc))]))])
      (values ids (list->array gather) (list->array (iota n)) n))))

(define (done? tokens limit) (or (= (length tokens) limit) (and (pair? tokens) (= (util:last tokens) eos))))

(define (greedy first-pos max-new)
  (let ([limit (min decode-length (+ first-pos max-new))])
    (let loop ([tokens '()] [pos first-pos])
      (if (done? tokens (- limit first-pos))
          tokens
          (let* ([self-mask (list->array (util:pad-list (util:make-list (+ pos 1) 1) decode-length 0))]
                 [r (run decode-step [token (list->array (list (if (null? tokens) bos (util:last tokens))))]
                                     [pos (list->array (list pos))] [mask self-mask])])
            (loop (append tokens (util:->integers (array->list (output r 'next)))) (+ pos 1)))))))

(define (generation-results tokens) (results [text (detokenize tok tokens)] [tokens tokens]))

(pipeline caption ([image image] [prompt string])
  (let-values ([(tiles rows cols) (image-tile image 2048 512 'mean '(0.5 0.5 0.5) 'std '(0.5 0.5 0.5))])
    (let* ([rows (car (util:->integers (array->list rows)))]
           [cols (car (util:->integers (array->list cols)))]
           [n-tiles (car (array-shape tiles))]
           [n-image-tokens (* tile-tokens n-tiles)]
           [expanded (expand-image (apply-chat-template prompt) rows cols)])
      (let-values ([(ids gather pos n) (prompt-inputs expanded n-image-tokens)])
        (let* ([at (list->array (list (- n 1)))]
               [r (run prefill-image [pixels tiles] [ids ids] [gather gather] [pos pos] [at at])]
               [first (car (array->list (array-argmax (output r 'logits))))])
          (generation-results (cons first (greedy n 512))))))))

(pipeline generate ([prompt string])
  (let-values ([(ids gather pos n) (prompt-inputs (apply-chat-template prompt) 0)])
    (let* ([at (list->array (list (- n 1)))]
           [r (run prefill-text [ids ids] [pos pos] [at at])]
           [first (car (array->list (array-argmax (output r 'logits))))])
      (generation-results (cons first (greedy n 256))))))
