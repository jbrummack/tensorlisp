;; LoRA fine-tuning of T5Gemma 2 (text). Not a standalone program: append it to
;; t5gemma2.ss (cat t5gemma2.ss train.ss > t5gemma2-train.ss), then train with the
;; `train` entry (see docs/design/training.md; crates/tensorlisp/tests/train_t5gemma2.rs).
;;
;; Adapters (PEFT semantics, y = W x + (alpha / r) B A x, A kaiming-uniform, B zeros) on the
;; linear layers listed in lora-targets of every encoder and decoder layer. The base weights
;; stay frozen (f16 in the file). They are `(define-param ...)`s named
;; lora.<module>.A / .B, which every other entry (encode, decode, decode-step, ...) also
;; applies, so the same program generates with a trained adapter.
;;
;; Pipeline train-example (input string, output string) -> the inputs of `train`, padded to
;; train-src-len encoder and train-tgt-len decoder tokens (one graph shape for the whole run):
;; the output text is tokenized and closed with <eos>, the decoder sees <bos> and the output
;; shifted right, every output token (and <eos>) weighs 1 / its count.
;;
;; Entry train (teacher forcing; numpy order, batch 1, L encoder tokens, T decoder tokens):
;;   ids, gather, pos [1, L] i32, mask [1, L] f32   as `encode` (gather = 0..L-1 without image)
;;   tokens, dpos [1, T] i32                        decoder input ([bos, y_0 .. y_{T-2}]), positions
;;   targets [1, T] f32                             y_0 .. y_{T-1} (token ids, whole numbers)
;;   weights [1, T] f32                             per-token loss weight, normalized by the host
;;                                                  (mask / sum(mask)): 0 for padding and prompt
;;   -> loss [1]: sum_t weights_t * -log softmax(logits_t)[targets_t]; nll [T]: -log p per token

(define lora-rank 8)
(define lora-alpha 16.0)
;; (module suffix, in features, out features)
(define lora-targets
  '(("self_attn.q_proj" 640 1024) ("self_attn.k_proj" 640 256) ("self_attn.v_proj" 640 256)
    ("self_attn.o_proj" 1024 640)
    ("mlp.gate_proj" 640 2048) ("mlp.up_proj" 640 2048) ("mlp.down_proj" 2048 640)))

(lora-config! lora-rank lora-alpha)
(for-each
  (lambda (stack)
    (for-each
      (lambda (i)
        (for-each (lambda (m) (lora-attach! (format "~a.layers.~a.~a" stack i (car m)) (cadr m) (caddr m)))
                  lora-targets))
      (iota layers)))
  '("encoder" "decoder"))

(model train (inputs [ids i32 (_ 1)] [gather i32 (_ 1)] [pos i32 (_ 1)] [mask f32 (_ 1)]
                     [tokens i32 (_ 1)] [dpos i32 (_ 1)] [targets f32 (_ 1)] [weights f32 (_ 1)])
  ;; Only ops with a backward pass: the exact tanh GELU, no fused attention.
  (define exact (set! exact-gelu #t))
  (define x (embed (flat ids) (flat gather) #f))
  (define memory (encode x (flat pos) (flat mask)))
  (define n (dim tokens 0))
  (define cross (attn:padding-mask (flat mask) n))
  (define masks (cons (tensor:concat (list (attn:causal-mask n 'window sliding-window) cross) 0)
                      (tensor:concat (list (attn:causal-mask n) cross) 0)))
  (define hidden-states
    (rms-norm (let loop ([i 0] [x (decoder-embed (flat tokens))])
                (if (= i layers) x (loop (+ i 1) (decoder-layer x i (flat dpos) memory masks))))
              "decoder.norm"))
  (define logits (ggml-mul-mat (weight "encoder.embed_tokens.weight") hidden-states))   ; [vocab, T]
  (define probs (ggml-soft-max logits))
  ;; one-hot of the targets [vocab, T], built from ids (no gradient flows into it)
  (define onehot
    (ggml-step (ggml-scale-bias (ggml-abs (ggml-sub (ggml-repeat-4d (ggml-reshape-2d (ggml-arange 0.0 (inexact vocab) 1.0) vocab 1) vocab n 1 1)
                                                    (ggml-reshape-2d (flat targets) 1 n)))
                                -1.0 0.5)))
  (define target-prob (ggml-sum-rows (ggml-mul probs onehot)))                           ; [1, T]
  (define nll (ggml-scale (ggml-log (ggml-scale-bias target-prob 1.0 1e-20)) -1.0))
  (outputs [loss (ggml-sum (ggml-mul nll (ggml-reshape-2d (flat weights) 1 n)))]
           [nll (ggml-reshape-1d nll n)]))

;; --- data

(define train-src-len 64)
(define train-tgt-len 32)

(pipeline train-example ([input string] [output string])
  (let-values ([(ids gather pos mask) (encoder-inputs input 0)])
    (let ([src (util:->integers (array->list ids))])
      (when (> (length src) train-src-len)
        (error 'train-example (format "the input is ~a tokens, train-src-len is ~a" (length src) train-src-len)))
      (let-values ([(out-ids out-mask) (tokenize tok output)])
        (let* ([raw (util:->integers (array->list out-ids))]
               [y (append (if (and (pair? raw) (= (car raw) bos)) (cdr raw) raw) (list eos))]
               [m (length y)])
          (when (> m train-tgt-len)
            (error 'train-example (format "the output is ~a tokens with <eos>, train-tgt-len is ~a" m train-tgt-len)))
          (results
            [ids (list->array (util:pad-list src train-src-len 0))]
            [gather (list->array (iota train-src-len))]
            [pos (list->array (iota train-src-len))]
            [mask (list->array (util:pad-list (util:make-list (length src) 1) train-src-len 0))]
            [tokens (list->array (util:pad-list (cons bos (reverse (cdr (reverse y)))) train-tgt-len 0))]
            [dpos (list->array (iota train-tgt-len))]
            [targets (list->array (util:pad-list y train-tgt-len 0))]
            [weights (list->array (util:pad-list (util:make-list m (/ 1.0 m)) train-tgt-len 0.0))]))))))
