;; VoiceChat as a live stream: audio in as it arrives, the agent's speech out frame by frame.
;; Append to backbone.ss audio.ss tts.ss.
;;
;; The FastConformer runs one 80 ms frame at a time with caches (NeMo's cache-aware streaming):
;;   enc-sub   a window of mel frames -> the subsampled frames. Frame o of the subsampling reads mel frames
;;             8o-14 .. 8o, so the last output of the window [8(o-3), 8o] is exact (a window starting at 0 pads like the
;;             full utterance); at the end of the audio the same window gives the frames the right zero padding makes.
;;   enc-step  one subsampled frame -> emb, proj; every layer keeps the last 70 keys and values of its attention and
;;             the last 8 inputs of its depthwise conv (zeros before the start, as the offline model pads).
;; Pipeline `listen` ([audio stream]: 16 kHz chunks of any size): per frame mel -> encoder -> duplex language model ->
;; speech decoder -> codec, emitting 'audio (1764 samples at 22.05 kHz per frame; the first is 1758 and 6 follow at the end), 'token / 'id for the agent's text. Results: text, tokens, audio (everything emitted).

(define enc-keys (+ enc-left 1))

(for-each
 (lambda (l)
   (define-state (format "ek.~a" l) f32 enc-d enc-left)
   (define-state (format "ev.~a" l) f32 enc-d enc-left)
   (define-state (format "ec.~a" l) f32 (- enc-conv-kernel 1) enc-d))
 (iota enc-layers))

(model enc-sub (inputs [mel f32 (128 _)])
  (outputs [sub (subsample mel) 2]))

;; one query against the 70 cached keys and its own: scores (q+u) k^T + (q+v) p^T, positions 70 .. 0
(define (stream-attention x l pe mask)
  (let* ([p (string-append (e-prefix l) ".self_attn")]
         [split (lambda (name) (ggml-reshape-3d (nn:linear x (string-append p name)) enc-hd enc-heads 1))]
         [q (split ".linear_q")]
         [knew (ggml-reshape-2d (nn:linear x (string-append p ".linear_k")) enc-d 1)]
         [vnew (ggml-reshape-2d (nn:linear x (string-append p ".linear_v")) enc-d 1)]
         [kfull (ggml-concat (state (format "ek.~a" l)) knew 1)]                              ; [enc-d, 71]
         [vfull (ggml-concat (state (format "ev.~a" l)) vnew 1)]
         [_ (effect (ggml-cpy (ggml-view-2d kfull enc-d enc-left (* 4 enc-d) (* 4 enc-d)) (state (format "ek.~a" l))))]
         [_ (effect (ggml-cpy (ggml-view-2d vfull enc-d enc-left (* 4 enc-d) (* 4 enc-d)) (state (format "ev.~a" l))))]
         [bias (lambda (name) (ggml-reshape-3d (weight (string-append p name)) enc-hd enc-heads 1))]
         [qu (heads-first (ggml-add q (bias ".pos_bias_u")) 1)]
         [qv (heads-first (ggml-add q (bias ".pos_bias_v")) 1)]
         [k (heads-first (ggml-reshape-3d kfull enc-hd enc-heads enc-keys) enc-keys)]
         [pp (heads-first (ggml-reshape-3d (nn:linear pe (string-append p ".linear_pos")) enc-hd enc-heads enc-keys) enc-keys)]
         [scores (ggml-add (ggml-mul-mat k qu) (ggml-mul-mat pp qv))]                          ; [key, 1, heads]
         [probs (ggml-soft-max-ext scores mask (/ 1.0 (sqrt (inexact enc-hd))) 0.0)]
         [vt (ggml-cont (ggml-permute (ggml-reshape-3d vfull enc-hd enc-heads enc-keys) 1 2 0 3))]
         [ctx (ggml-cont (ggml-permute (ggml-mul-mat vt probs) 0 2 1 3))])
    (nn:linear (ggml-reshape-2d ctx enc-d 1) (string-append p ".linear_out"))))

(define (stream-conv-module x l)
  (let* ([p (string-append (e-prefix l) ".conv")]
         [pointwise (lambda (name n-out x) (ggml-mul-mat (ggml-reshape-2d (weight (string-append p name ".weight")) (dim x 0) n-out) x))]
         [h (pointwise ".pointwise_conv1" (* 2 enc-d) x)]
         [h (ggml-mul (ggml-cont (tensor:slice h 0 0 enc-d)) (ggml-sigmoid (ggml-cont (tensor:slice h 0 enc-d enc-d))))]
         [hp (ggml-concat (state (format "ec.~a" l)) (ggml-cont (ggml-transpose h)) 0)]       ; [9, enc-d]
         [_ (effect (ggml-cpy (ggml-view-2d hp (- enc-conv-kernel 1) enc-d (* 4 enc-conv-kernel) 4) (state (format "ec.~a" l))))]
         [kernel (ggml-cast (ggml-reshape-2d (weight (string-append p ".depthwise_conv.weight")) enc-conv-kernel enc-d) GGML_TYPE_F32)]
         [h (ggml-ssm-conv (ggml-reshape-3d hp enc-conv-kernel enc-d 1) kernel)]
         [h (ggml-silu (nn:layer-norm (ggml-reshape-2d h enc-d 1) (string-append p ".batch_norm") 'eps enc-eps))])
    (pointwise ".pointwise_conv2" enc-d h)))

(define (stream-layer x l pe mask)
  (let* ([p (e-prefix l)]
         [norm (lambda (x name) (nn:layer-norm x (string-append p "." name) 'eps enc-eps))]
         [feed-forward (lambda (x name)
                         (nn:linear (ggml-silu (nn:linear x (string-append p "." name ".linear1"))) (string-append p "." name ".linear2")))]
         [x (ggml-add x (ggml-scale (feed-forward (norm x "norm_feed_forward1") "feed_forward1") 0.5))]
         [x (ggml-add x (stream-attention (norm x "norm_self_att") l pe mask))]
         [x (ggml-add x (stream-conv-module (norm x "norm_conv") l))]
         [x (ggml-add x (ggml-scale (feed-forward (norm x "norm_feed_forward2") "feed_forward2") 0.5))])
    (norm x "norm_out")))

(model enc-step (inputs [x f32 (1024 1)] [mask f32 (71 1)])
  (define pe (pos-emb-at (ggml-scale-bias (ggml-arange 0.0 (inexact enc-keys) 1.0) -1.0 (inexact enc-left)) enc-keys))
  (define out (let loop ([l 0] [x x]) (if (= l enc-layers) x (loop (+ l 1) (stream-layer x l pe mask)))))
  (outputs [emb out 2] [proj (nn:linear out "perc.proj") 2]))

;; Every state of the live path back to zero: a new conversation.
(model stream-reset (inputs [dummy f32 (1)])
  (define (zero! s) (effect (ggml-cpy (ggml-scale s 0.0) s)))
  (for-each (lambda (l) (for-each (lambda (kind) (zero! (state (format "~a.~a" kind l)))) '("ek" "ev" "ec"))) (iota enc-layers))
  (for-each (lambda (i) (zero! (state (format "cd.~a" i)))) (apply append (map (lambda (s) (map (lambda (b) (+ (* 4 s) 1 b)) (iota codec-blocks))) '(0 1 2))))
  (zero! (state "cd.tail"))
  (zero! (state "cd.etail"))
  (for-each (lambda (i) (when (eq? (layer-kind i) 'mamba) (zero! (state (format "conv.~a" i))) (zero! (state (format "ssm.~a" i)))))
            (iota layer-count))
  (outputs [ok (ggml-scale dummy 1.0) 1]))

;; --- the loop

;; mel frames [from, end) of the stream -> the subsampled frames of encoder frame `o` on (see the top)
(define (sub-rows ms o end)
  (let* ([s (max 0 (* 8 (- o 3)))]
         [r (output (run enc-sub [mel (nemo-mel-window ms s (- end s))]) 'sub)])
    (array-slice r (- o (quotient s 8)) (car (array-shape r)))))

;; encoder frames the offline model makes of `t` mel frames: three stride-2 convs (floor (t/2) + 1 each)
(define (encoder-frames t) (let ([f (lambda (n) (+ (quotient n 2) 1))]) (f (f (f t)))))

(define (attention-mask frames)
  (let ([cached (min frames enc-left)])
    (tts-array-of (map (lambda (j) (if (>= j (- enc-left cached)) 0.0 -1e9)) (iota enc-keys)) 1 enc-keys)))

;; The waveform a code completes (codec-step): 1764 samples, the first token's without the 6 padding samples at the start.
(define last-edge #f)     ; the 6 samples after the last token's, for the end of the stream
(define (codec-token code first? idft window)
  (let* ([r (run codec-step [codes (tts-array-of code 1 31)] [idft idft] [window window])]
         [wav (output r 'wav)])
    (set! last-edge (output r 'edge))
    (if first? (array-slice wav 6 codec-samples) wav)))

;; Runs the encoder over a stream of 16 kHz audio chunks, calling (on-frame emb proj) for every 80 ms frame as soon as
;; it exists.
(define (stream-encoder audio on-frame)
  (let ([ms (nemo-mel-new)] [o 0])
    (letrec
        ([rows (lambda (r)
                 (do ([i 0 (+ i 1)]) ((= i (car (array-shape r))))
                   (let ([e (run enc-step [x (array-slice r i (+ i 1))] [mask (attention-mask o)])])
                     (on-frame o (output e 'emb) (output e 'proj))
                     (set! o (+ o 1)))))])
      (let loop ()
        (let ([a (stream-recv audio)])
          (cond
            [a (unless (= (audio-rate a) 16000) (error 'stream-encoder "the audio must be 16 kHz"))
               (nemo-mel-push! ms a)
               (let ready ()
                 (when (>= (nemo-mel-frames ms) (+ (* 8 o) 1))
                   (rows (sub-rows ms o (+ (* 8 o) 1)))
                   (ready)))
               (loop)]
            [else
             (let ([total (nemo-mel-finish! ms)])
               (when (< o (encoder-frames total)) (rows (sub-rows ms o total))))])))
      o)))

;; The encoder alone (a check against `encode`): emb [T', 1024], proj [T', 4480].
(pipeline stream-encode ([audio stream])
  (run stream-reset [dummy (list->array (list 0.0))])
  (let ([embs '()])
    (stream-encoder audio (lambda (o emb proj) (set! embs (cons emb embs))))
    (results [emb (let ([all (reverse embs)]) (fold-left array-concat (car all) (cdr all)))])))

;; --- the live conversation, with a system prompt and tool calls
;;
;; The function channel (a second head on the language model) is mostly pad. When it emits <SPECIAL_20> (start of a
;; tool call) the model is writing a call, one token per frame, ending with <SPECIAL_21>; that is NeMo's function-calling
;; layout: while the call is written and answered, the user's audio is replaced by the encoded silence (the audio that
;; arrives meanwhile waits and is read afterwards) and the agent's speech for these frames is cut out. The call
;; `<TOOLCALL>[{"name": ..., "arguments": {...}}]</TOOLCALL>` is handed to the client (`(client-call "tool" call)`,
;; which answers with the response as JSON text; the pipeline argument `reply` stands in for it when it is not empty),
;; the response `<TOOL_RESPONSE>[...]</TOOL_RESPONSE>` is fed in on the function channel one token per frame, and one more
;; frame has the model close it with <SPECIAL_22>. With `tools` (a JSON array of {name, description, parameters}, or of OpenAI-style {function: {...}}, in the order the model was trained on: the call's properties come in the order of the schema) the call
;; tokens are constrained (outlines-core) to a valid call of one of the tools.

(define sotc 20)
(define eotc 21)
(define eotr 22)
(define max-call-frames 400)

(define mel-silence -16.63553233343869)       ; ln 2^-24: the log-mel of digital silence
(define silence-count 129)
(define silence-cache #f)

;; The encoder's output for silence, frame k (the last one repeats): what NeMo inserts for the frames of a tool call.
(define (silence-row k)
  (unless silence-cache
    (let ([mel (tts-array-of (util:make-list (* 128 (+ (* 8 (- silence-count 1)) 1)) mel-silence) (+ (* 8 (- silence-count 1)) 1) 128)])
      (set! silence-cache (output (run encode [mel mel]) 'proj))))
  (let ([k (min k (- (car (array-shape silence-cache)) 1))])
    (array-slice silence-cache k (+ k 1))))

(define (system-prompt system tools)
  (cond
    [(> (string-length tools) 0)
     (string-append (if (> (string-length system) 0) system default-system-message) fc-prompt-before-tools (tools-prompt tools) fc-prompt-after-tools)]
    [else system]))

(define (token-ids text)
  (let-values ([(ids mask) (tokenize tok text 'special-tokens #f)])
    (util:->integers (array->list ids))))

(define (tool-response call reply)
  (if (> (string-length reply) 0) reply (client-call "tool" call)))

(define (listen-core audio system tools reply brain?)
  (run stream-reset [dummy (list->array (list 0.0))])
  (random-seed! 1234)
  (let* ([idft (and (not brain?) (codec-idft))] [window (and (not brain?) (codec-window))] [silence (and (not brain?) (tts-silence-codes))]
         [guide (and (> (string-length tools) 0) (guide-new (asset "tokenizer.json") (tool-call-regex tools 3) eotc))]
         [text text-pad] [func text-pad] [prev tts-pad-code] [texts '()] [speech #f] [calls '()]
         [lp 0] [frames 0] [spoken? #f] [queue '()] [in-call #f])
    ;; the system prompt fills the first positions of the language model
    (let ([prompt (system-prompt system tools)])
      (when (> (string-length prompt) 0)
        (let ([ids (append '(1) (token-ids prompt) '(2))])
          (let chunk ([from 0])
            (when (< from (length ids))
              (let* ([n (min 256 (- (length ids) from))] [part (list-head (list-tail ids from) n)])
                (run prefill [token (list->array part)] [pos (list->array (iota-from from n))] [pad (list->array (list text-pad))])
                (chunk (+ from n)))))
          (set! lp (length ids)))))
    (unless brain? (tts-warmup silence))
    (letrec
        ([put-audio (lambda (a) (emit 'audio a) (set! speech (if speech (array-concat speech a) a)))]
         ;; One language-model frame with the given audio row; the speech decoder follows it. gap?: inside a tool call
         ;; (its speech is not played). force: a function-channel token to use instead of the model's.
         [frame!
          (lambda (row gap? force)
            (let* ([d (run duplex [audio row] [text (list->array (list text))] [func (list->array (list func))]
                                  [pos (list->array (list lp))]
                                  [fmask (if (and guide in-call) (guide-mask guide) no-mask)])]
                   [next-text (->int (output d 'next_text))]
                   [next-func (->int (output d 'next_func))])
              (set! lp (+ lp 1))
              (set! text next-text)
              (set! func (or force next-func))
              (set! texts (cons next-text texts))
              (unless (memv next-text '(1 2 12))
                (emit 'token (detokenize tok (list next-text))))
              (if brain?
                  (emit 'frame (list->array (list (inexact next-text) (if gap? 1.0 0.0))))
                  (let ([code (if (= frames 0)
                                  (util:make-list 31 0)
                                  (tts-frame next-text prev (+ tts-prompt-frames (- frames 1)) silence))])
                    (set! frames (+ frames 1))
                    (set! prev code)
                    (unless gap?
                      (put-audio (codec-token code (not spoken?) idft window))
                      (set! spoken? #t))))
              next-func))]
         ;; The model started a call: write it on silent frames, run the tool, feed the answer back.
         [call!
          (lambda ()
            (set! in-call #t)
            (when guide (guide-reset! guide))
            (let loop ([k 0] [ids '()])
              (let ([nf (if (< k max-call-frames) (frame! (silence-row k) #t #f) eotc)])
                (cond
                  [(= nf eotc)
                   (set! in-call #f)
                   (let* ([call (detokenize tok (reverse ids) 'skip-special #f)]
                          [_1 (emit 'call call)]
                          [answer (tool-response call reply)]
                          [response (string-append "<TOOL_RESPONSE>[" answer "]</TOOL_RESPONSE>")]
                          [_2 (emit 'response response)])
                     (set! calls (cons call calls))
                     (let feed ([k 0] [rest (token-ids response)])
                       (unless (null? rest)
                         (frame! (silence-row k) #t (car rest))
                         (feed (+ k 1) (cdr rest))))
                     (frame! (silence-row 0) #t eotr)
                     (set! func eotr))]
                  [else
                   (when guide (guide-advance! guide nf))
                   (loop (+ k 1) (cons nf ids))]))))]
         [normal-frame!
          (lambda (row)
            (let ([nf (frame! row #f #f)])
              (when (and guide (= nf sotc)) (call!))))]
         [drain!
          (lambda ()
            (unless (null? queue)
              (let ([row (car queue)])
                (set! queue (cdr queue))
                (normal-frame! row)
                (drain!))))])
      (stream-encoder audio (lambda (o emb proj) (set! queue (append queue (list proj))) (drain!)))
      (when spoken? (put-audio last-edge))
      (let ([texts (reverse texts)])
        (results [text (detokenize tok (filter (lambda (id) (not (memv id '(1 2 12)))) texts))] [tokens texts]
                 [calls (apply string-append (map (lambda (c) (string-append c "\n")) (reverse calls)))]
                 [audio (or speech (list->array (list 0.0)))])))))

(define (iota-from from n) (map (lambda (i) (+ from i)) (iota n)))

(pipeline listen ([audio stream])
  (listen-core audio "" "" "" #f))

;; With a system message (empty: the default) and tools (a JSON array); `reply`, if not empty, is the tool response (JSON)
;; used when the client does not answer `(client-call "tool" call)`.
(pipeline listen-tools ([audio stream] [system string] [tools string] [reply string])
  (listen-core audio system tools reply #f))
;; --- the same conversation split over two machines
;;
;; `brain` (the encoder and the language model, no speech weights) takes the user's audio and emits, per 80 ms frame, a
;; 2-element array [agent text token, 1 if the frame is inside a tool call] on `frame` (plus the `token`, `call` and
;; `response` of listen-tools). `voice` (the speech decoder and the codec, no language model) takes that stream of
;; frames and emits the agent's audio. Nothing flows back from `voice` to `brain`, so a client (or a relay) only
;; forwards the frames; per frame that is 8 bytes one way, and 44 KB/s of audio the other.
(pipeline brain ([audio stream] [system string] [tools string] [reply string])
  (listen-core audio system tools reply #t))

(pipeline voice ([frames stream])
  (let* ([idft (codec-idft)] [window (codec-window)] [silence (tts-silence-codes)]
         [prev tts-pad-code] [n 0] [spoken? #f] [speech #f])
    (tts-warmup silence)
    (let loop ()
      (let ([f (stream-recv frames)])
        (when f
          (let* ([v (array->list f)] [text (inexact->exact (round (car v)))] [gap? (> (cadr v) 0.5)]
                 [code (if (= n 0) (util:make-list 31 0) (tts-frame text prev (+ tts-prompt-frames (- n 1)) silence))])
            (set! n (+ n 1))
            (set! prev code)
            (unless gap?
              (let ([a (codec-token code (not spoken?) idft window)])
                (emit 'audio a)
                (set! speech (if speech (array-concat speech a) a))
                (set! spoken? #t))))
          (loop))))
    (when spoken?
      (emit 'audio last-edge)
      (set! speech (array-concat speech last-edge)))
    (results [audio (or speech (list->array (list 0.0)))])))
