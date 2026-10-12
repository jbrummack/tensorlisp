;; Extra pipelines for serving T5Gemma 2 over RPC (crates/tensorlisp-serve, docs/design/serve.md). Appended to
;; t5gemma2.ss, which defines everything they use:
;;   cat t5gemma2.ss rpc.ss > t5gemma2-rpc.ss && tl pack t5gemma2-270m-q8_0.gguf --program t5gemma2-rpc.ss -o t5gemma2-rpc-q8.gguf

;; --- RPC demo pipelines: token streaming, a client-side sampler, and a prompt stream

;; Encodes a text prompt: (values memory mask).
(define (prompt-memory prompt)
  (let-values ([(ids gather pos mask) (encoder-inputs prompt 0)])
    (values (output (run encode [ids ids] [gather gather] [pos pos] [mask mask]) 'memory) mask)))

(define (step-token r) (car (util:->integers (array->list (output r 'next)))))

;; Like greedy-cached, but `choose` picks each token from the decode step's result and the tokens so
;; far, and every new token is emitted as it is chosen ('id: the id, 'token: its text).
(define (decode-loop memory memory-mask max-new choose)
  (let ([limit (min decode-length (+ max-new 1))])
    (run cross [memory memory])
    (let ([memory-mask (list->array (util:pad-list (array->list memory-mask) memory-capacity 0))])
      (let loop ([tokens (list bos)] [pos 0])
        (if (done? tokens limit)
            tokens
            (let* ([r (run decode-step [token (list->array (list (util:last tokens)))]
                                       [pos (list->array (list pos))]
                                       [self-mask (list->array (util:pad-list (util:make-list (+ pos 1) 1) decode-length 0))]
                                       [memory-mask memory-mask])]
                   [next (choose r tokens)])
              (emit 'id (list->array (list next)))
              (emit 'token (detokenize tok (list next)))
              (loop (append tokens (list next)) (+ pos 1))))))))

(define (choose-greedy r tokens) (step-token r))
;; The client picks among the 20 best candidates.
(define (choose-client r tokens) (client-sample (output r 'logits) tokens 20))

(pipeline generate-stream ([prompt string])
  (let-values ([(memory mask) (prompt-memory prompt)])
    (generation-results (decode-loop memory mask 32 choose-greedy))))

(pipeline generate-sampled ([prompt string])
  (let-values ([(memory mask) (prompt-memory prompt)])
    (generation-results (decode-loop memory mask 32 choose-client))))

;; One reply per prompt received, until the prompt stream ends.
(pipeline chat ([prompts stream])
  (let loop ([turns 0])
    (let ([prompt (stream-recv prompts)])
      (cond
        [prompt
         (let-values ([(memory mask) (prompt-memory prompt)])
           (emit 'reply (detokenize tok (decode-loop memory mask 32 choose-greedy)))
           (loop (+ turns 1)))]
        [else (results [turns turns])]))))
