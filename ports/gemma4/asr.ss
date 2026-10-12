;; Gemma 4 with audio: the language model takes audio tokens (the audio tower's embeddings) in place of
;; text embeddings, in chunks as they are produced. Append after gemma4.ss, audio_clips.ss and audio.ss.
;;
;; In the model, a position holding an audio token has the audio embedding as its input (not scaled; the
;; per-layer inputs are projected from it like from any embedding, with the padding token's identity part),
;; and attention is plain causal, so the tokens before the audio, the audio chunks as they arrive and
;; the tokens after it can be fed to the decoder one block after the other, filling the same KV cache.
;;
;;   fill-audio   token, pos [n], audio [n, 1536], amask [n, 1] (1 where the position is an audio token)
;;                -> h [1536] (nothing else: the cache is what it is for)
;;   step-audio   the same plus at [1] (which of the n positions is the last real one; the rest is
;;                padding that only writes cache rows later tokens overwrite) -> logits, next

(define pad-id 0)

;; (embeds . ple-embeds) for ids with audio rows where amask is 1. The per-layer inputs are projected from the
;; merged embeddings (audio features included), their token-identity part looks up the padding id at
;; audio positions (the caller passes token 0 there).
(define (audio-embeds ids audio amask)
  (let* ([text (token-embeds ids)]
         [keep (ggml-scale-bias amask -1.0 1.0)]
         [embeds (ggml-add (ggml-mul text keep) (ggml-mul audio amask))])
    (cons embeds embeds)))

(model fill-audio (inputs [token i32 (_ 1)] [pos i32 (_ 1)] [audio f32 (1536 _)] [amask f32 (1 _)])
  (define ids (flat token))
  (define n (dim ids 0))
  (define mixed (audio-embeds ids audio amask))
  (define x (decode-stack ids (flat pos) (car mixed) (cdr mixed)))
  (outputs [h (tensor:slice x 1 (- n 1) 1) 2]))

(model step-audio (inputs [token i32 (_ 1)] [pos i32 (_ 1)] [audio f32 (1536 _)] [amask f32 (1 _)] [at i32 (1 1)])
  (define ids (flat token))
  (define mixed (audio-embeds ids audio amask))
  (define x (decode-stack ids (flat pos) (car mixed) (cdr mixed)))
  (define raw (logits-of (ggml-get-rows x (flat at))))
  (outputs [logits (softcapped raw) 2] [next (ggml-argmax raw) 1]))

;; --- generation helpers

(define end-of-turn (token-id tok "<turn|>"))

(define (stop? token) (or (= token eos) (= token end-of-turn)))

(define (ints->array xs) (list->array xs))
(define (repeat-list x n) (util:make-list n x))
(define (range-from start n) (map (lambda (i) (+ start i)) (iota n)))

;; Greedy decoding from the first sampled token at position `pos`, at most `limit` tokens.
(define (decode-from first pos limit)
  (let loop ([tokens (list first)] [pos pos])
    (if (or (stop? (util:last tokens)) (>= (length tokens) limit))
        tokens
        (let ([r (run step [token (list->array (list (util:last tokens)))] [pos (list->array (list pos))])])
          (loop (append tokens (util:->integers (array->list (output r 'next)))) (+ pos 1))))))

;; The fixed-shape call after the audio: the audio tokens that did not fill a chunk (rows of `left`,
;; [r, 1536]), then the text tokens `post`, padded to `size` positions. Returns the first generated token.
(define final-size 24)
(define (final-block pos0 left post-ids)
  (let* ([r (if left (car (array-shape left)) 0)]
         [n (+ r (length post-ids))]
         [pad (- final-size n)])
    (when (< pad 0) (error 'final-block "too many tokens after the audio" n))
    (let* ([zeros (lambda (k) (repeat-list 0 k))]
           [tokens (append (zeros r) post-ids (zeros pad))]
           [audio (let ([rows (if left left #f)])
                    (cond [(and rows (> (- final-size r) 0)) (array-concat rows (array-zeros (- final-size r) 1536))]
                          [rows rows]
                          [else (array-zeros final-size 1536)]))]
           [amask (array-reshape (list->array (append (repeat-list 1.0 r) (zeros (- final-size r)))) final-size 1)]
           [r* (run step-audio [token (list->array tokens)] [pos (list->array (range-from pos0 final-size))]
                               [audio audio] [amask amask] [at (list->array (list (- n 1)))])])
      (values (car (util:->integers (array->list (output r* 'next)))) (+ pos0 n)))))

;; Reference pipeline: audio embeddings computed elsewhere (emb [tokens, 1536]), the prompt around them as text.
(pipeline transcribe-ref ([emb array] [pre string] [post string])
  (let-values ([(pre-ids m1) (tokenize tok pre)]
               [(post-ids m2) (tokenize tok post 'special-tokens #f)])
    (let* ([pre-list (util:->integers (array->list pre-ids))]
           [post-list (util:->integers (array->list post-ids))]
           [np (length pre-list)]
           [total (car (array-shape emb))]
           [full (quotient total a-chunk)])
      (run step [token pre-ids] [pos (list->array (iota np))])
      (let loop ([c 0])
        (when (< c full)
          (run fill-audio [token (list->array (repeat-list 0 a-chunk))]
                          [pos (list->array (range-from (+ np (* c a-chunk)) a-chunk))]
                          [audio (array-slice emb (* c a-chunk) (* (+ c 1) a-chunk))]
                          [amask (array-reshape (list->array (repeat-list 1.0 a-chunk)) a-chunk 1)])
          (loop (+ c 1))))
      (let-values ([(first pos) (final-block (+ np (* full a-chunk))
                                             (and (> total (* full a-chunk)) (array-slice emb (* full a-chunk) total))
                                             post-list)])
        (generation-results (decode-from first pos 96))))))

;; The same prompt in one block (no chunking, no padding), to compare with the incremental pipeline.
(pipeline transcribe-oneshot ([emb array] [pre string] [post string])
  (let-values ([(pre-ids m1) (tokenize tok pre)]
               [(post-ids m2) (tokenize tok post 'special-tokens #f)])
    (let* ([pre-list (util:->integers (array->list pre-ids))]
           [post-list (util:->integers (array->list post-ids))]
           [np (length pre-list)] [na (car (array-shape emb))] [ns (length post-list)]
           [n (+ np na ns)]
           [audio (array-concat (array-concat (array-zeros np 1536) emb) (array-zeros ns 1536))]
           [amask (array-reshape (list->array (append (repeat-list 0.0 np) (repeat-list 1.0 na) (repeat-list 0.0 ns))) n 1)]
           [r (run step-audio [token (list->array (append pre-list (repeat-list 0 na) post-list))]
                              [pos (list->array (iota n))] [audio audio] [amask amask] [at (list->array (list (- n 1)))])])
      (generation-results (decode-from (car (util:->integers (array->list (output r 'next)))) n 96)))))

;; --- streaming transcription

;; The audio tower's states back to zero (the start of an utterance).
(model audio-reset (inputs [dummy f32 (1)])
  (for-each (lambda (l)
              (for-each (lambda (kind)
                          (let ([s (state (format "~a.~a" kind l))])
                            (effect (ggml-cpy (ggml-scale s 0.0) s))))
                        '("ak" "av" "ac")))
            (iota a-layers))
  (outputs [ok (ggml-scale dummy 1.0) 1]))

;; Like decode-from, emitting each token's text as 'token and its id as 'id.
(define (decode-emit first pos limit)
  (emit 'id (list->array (list first)))
  (emit 'token (detokenize tok (list first)))
  (let loop ([tokens (list first)] [pos pos])
    (if (or (stop? (util:last tokens)) (>= (length tokens) limit))
        tokens
        (let* ([r (run step [token (list->array (list (util:last tokens)))] [pos (list->array (list pos))])]
               [next (car (util:->integers (array->list (output r 'next))))])
          (emit 'id (list->array (list next)))
          (emit 'token (detokenize tok (list next)))
          (loop (append tokens (list next)) (+ pos 1))))))

;; One conformer chunk c over `frames` mel frames so far: the 12 tokens' embeddings [12, 1536].
(define (conformer-chunk ms c frames)
  (let-values ([(mask1 amask) (chunk-masks c frames)])
    (output (run audio-chunk [mel (usm-mel-window ms (- (* 48 c) 3) a-mel-window)] [mask1 mask1]
                             [winidx audio-winidx] [posemb audio-posemb] [amask amask])
            'emb)))

;; Feeds a full chunk's embeddings to the language model.
(define (fill-chunk np c emb)
  (run fill-audio [token (list->array (repeat-list 0 a-chunk))]
                  [pos (list->array (range-from (+ np (* c a-chunk)) a-chunk))]
                  [audio emb]
                  [amask (array-reshape (list->array (repeat-list 1.0 a-chunk)) a-chunk 1)]))

;; The audio arrives as a stream of 16 kHz audio chunks. The prompt before the audio is prefilled at once,
;; every 480 ms of audio is run through the conformer and prefilled as soon as the front end has the 51 mel
;; frames it needs, and when the stream ends the last partial chunk, the text after the audio and the first
;; token are one call away. Tokens are emitted as 'token / 'id while the answer is generated.
(pipeline transcribe ([audio stream] [pre string] [post string])
  (let-values ([(pre-ids m1) (tokenize tok pre)]
               [(post-ids m2) (tokenize tok post 'special-tokens #f)])
    (run audio-reset [dummy (list->array (list 0.0))])
    (let* ([np (car (array-shape pre-ids))]
           [post-list (util:->integers (array->list post-ids))]
           [ms (usm-mel-new)])
      (run step [token pre-ids] [pos (list->array (iota np))])
      ;; c: chunks run so far
      (let loop ([c 0])
        (let ([a (stream-recv audio)])
          (cond
            [a
             (let ([frames (usm-mel-push! ms a)])
               (let ready ([c c])
                 (if (>= frames (+ (* 48 c) 48))
                     (begin (fill-chunk np c (conformer-chunk ms c frames))
                            (ready (+ c 1)))
                     (loop c))))]
            [else
             (emit 'stage "end of audio")
             ;; the end: T mel frames make N tokens (token k exists while 4k < T)
             (let* ([frames (usm-mel-frames ms)]
                    [n (quotient (+ frames 3) 4)]
                    [full (quotient n a-chunk)]
                    [left-tokens (- n (* full a-chunk))])
               (let finish ([c c])
                 (cond
                   [(< c full)
                    (fill-chunk np c (conformer-chunk ms c frames))
                    (finish (+ c 1))]
                   [else
                    (let ([left (and (> left-tokens 0)
                                     (array-slice (conformer-chunk ms full frames) 0 left-tokens))])
                      (emit 'stage "audio tower done")
                      (let-values ([(first pos) (final-block (+ np (* full a-chunk)) left post-list)])
                        (emit 'stage "first token chosen")
                        (generation-results (decode-emit first pos 128))))])))]))))))
