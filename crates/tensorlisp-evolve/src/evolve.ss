;; The Scheme the harness runs for agents: sessions (a REPL that keeps its definitions between calls)
;; and the candidate evaluator. Evaluated once, after the Tree libraries are loaded.
;;
;; Sessions and candidate code run in a copy of an environment that has R6RS without file access,
;; the Tree libraries and a few Chez printing helpers. That keeps honest mistakes from touching the
;; disk or the process; it is not a security boundary.

(define ($evo-env-spec kernels)
  (list
    '(except (rnrs) open-file-input-port open-file-output-port open-file-input/output-port
             open-input-file open-output-file call-with-input-file call-with-output-file
             with-input-from-file with-output-to-file file-exists? delete-file)
    '(only (chezscheme) pretty-print format printf iota list-head)
    '(tl tree) '(tl tree opt) kernels '(tl tree ptx) '(tl tree msl)))

(define $evo-base-env (apply environment ($evo-env-spec '(tl tree kernels))))

(define $evo-optimize (eval 'optimize $evo-base-env))
(define $evo-default-passes (eval 'default-passes $evo-base-env))
(define $evo-has-cmat? (eval 'has-cmat? $evo-base-env))
(define $evo-compile-spirv (eval 'compile-kernel $evo-base-env))
(define $evo-compile-ptx (eval 'compile-kernel-ptx $evo-base-env))
(define $evo-compile-msl (eval 'compile-kernel-msl $evo-base-env))

;; Helpers every session (and every candidate) can call.
(define $evo-helpers
  '((define (tree-size t) (cond [(pair? t) (+ 1 (tree-size (car t)) (tree-size (cdr t)))] [else 0]))
    (define (show x) (pretty-print x))
    (define (spirv-words k) (quotient (bytevector-length (compile-kernel k)) 4))
    ;; (stats k): size of the kernel tree and of its SPIR-V
    (define (stats k) (list (cons 'nodes (tree-size k)) (cons 'spirv-words (spirv-words k))))
    ;; (compare k '(fold cse) '(fold licm cse dce)): how two pipelines do on one kernel
    (define (compare k a b)
      (list (cons a (stats (optimize k a))) (cons b (stats (optimize k b)))))
    ;; (diff-stmts a b): the top-level statements that differ between two kernels
    (define (diff-stmts a b)
      (let ([sa (cddr a)] [sb (cddr b)])
        (list (cons 'only-in-a (filter (lambda (s) (not (member s sb))) sa))
              (cons 'only-in-b (filter (lambda (s) (not (member s sa))) sb)))))))

;; A mutable copy of `base` (default: the original generators) with the helpers defined.
(define ($evo-new-env . base)
  (let ([env (copy-environment (if (null? base) $evo-base-env (car base)) #t)])
    (for-each (lambda (f) (eval f env)) $evo-helpers)
    env))

(define $evo-sessions (make-hashtable string-hash string=?))

(define ($evo-env name)
  (or (hashtable-ref $evo-sessions name #f)
      (let ([e ($evo-new-env)]) (hashtable-set! $evo-sessions name e) e)))

(define ($evo-reset name) (hashtable-delete! $evo-sessions name))

;; Runs thunk, raising an error once `ms` of real time have passed. Chez counts procedure calls
;; with a timer and calls the handler every so often.
(define ($evo-timed ms thunk)
  (let ([deadline (+ (real-time) ms)] [old (timer-interrupt-handler)])
    (dynamic-wind
      (lambda ()
        (timer-interrupt-handler
          (lambda ()
            (if (> (real-time) deadline)
                (begin (set-timer 0) (error 'evolve "time limit exceeded"))
                (set-timer 20000))))
        (set-timer 20000))
      thunk
      (lambda () (set-timer 0) (timer-interrupt-handler old)))))

(define ($evo-read-all code)
  (let ([p (open-input-string code)])
    (let loop ([acc '()])
      (let ([x (read p)])
        (if (eof-object? x) (reverse acc) (loop (cons x acc)))))))

;; The REPL tool: output printed by the code, then "=> value" of the last form.
(define ($evo-run name code ms)
  (let-values ([(port get) (open-string-output-port)])
    (let* ([env ($evo-env name)]
           [failure #f]
           [last
             (guard (e [#t (set! failure ($tl-error-message e)) (void)])
               ($evo-timed ms
                 (lambda ()
                   (parameterize ([current-output-port port])
                     (let loop ([forms ($evo-read-all code)] [v (void)])
                       (cond
                         [(null? forms) v]
                         ;; the libraries are imported already; a model's habitual (import ...) is not an error
                         [(and (pair? (car forms)) (eq? (caar forms) 'import))
                          (display "(import ignored: rnrs, (tl tree), (tl tree opt), (tl tree kernels), (tl tree ptx) and (tl tree msl) are already imported)")
                          (newline)
                          (loop (cdr forms) v)]
                         [else (loop (cdr forms) (eval (car forms) env))]))))))])
      (string-append
        (get)
        (cond [failure (string-append "error: " failure)]
              [(eq? last (void)) ""]
              [else (string-append "=> " (call-with-string-output-port (lambda (o) (write last o))))])))))

;; ---- the evaluator --------------------------------------------------------------------------------

(define ($evo-compile target k)
  (case target
    [(spirv) ($evo-compile-spirv k)]
    [(ptx) ($evo-compile-ptx k)]
    [(msl) ($evo-compile-msl k)]
    [else (error 'evaluate "unknown target" target)]))

;; SPIR-V words, or characters of PTX / MSL text.
(define ($evo-size out)
  (if (bytevector? out) (quotient (bytevector-length out) 4) (string-length (car out))))

(define ($evo-header k)
  (filter (lambda (f) (and (pair? f) (memq (car f) '(workgroup buffers push shared offsets)) #t)) (cddr k)))

(define ($evo-unique l)
  (let loop ([l l] [acc '()])
    (cond [(null? l) (reverse acc)]
          [(member (car l) acc) (loop (cdr l) acc)]
          [else (loop (cdr l) (cons (car l) acc))])))

;; What a kernel does to the outside: the buffers it stores to and whether it synchronizes.
(define ($evo-effects k)
  (let ([stores '()] [barriers 0])
    (let walk ([t (cddr k)])
      (when (pair? t)
        (case (car t)
          [(store) (when (pair? (cdr t)) (set! stores (cons (if (symbol? (cadr t)) (cadr t) 'expr) stores)))]
          [(cm-store) (when (and (pair? (cdr t)) (pair? (cddr t)))
                        (set! stores (cons (if (symbol? (caddr t)) (caddr t) 'expr) stores)))]
          [(barrier) (set! barriers (+ barriers 1))]
          [else (void)])
        (let loop ([x t]) (when (pair? x) (walk (car x)) (loop (cdr x))))))
    (cons (list-sort (lambda (a b) (string<? (symbol->string a) (symbol->string b))) ($evo-unique stores))
          (> barriers 0))))

;; #f when the candidate kept the kernel's interface and effects, else a sentence saying what changed.
(define ($evo-structure-problem base cand)
  (cond
    [(not (and (pair? cand) (eq? (car cand) 'kernel))) "the pipeline did not return a kernel form"]
    [(not (equal? ($evo-header base) ($evo-header cand))) "the buffers / push / workgroup / shared clauses changed"]
    [else
     (let ([a ($evo-effects base)] [b ($evo-effects cand)])
       (cond [(not (equal? (car a) (car b)))
              (format "the buffers stored to changed from ~s to ~s" (car a) (car b))]
             [(not (eq? (cdr a) (cdr b))) "barrier presence changed"]
             [else #f]))]))

;; One row per generator expression: (expr status note base-size candidate-size bytes-or-#f base-bytes-or-#f cand-reply-or-#f base-reply-or-#f).
;; The baseline is built by the original generators in `clean-env` and the default passes; the candidate by the
;; generators visible in `env` (the candidate's, when it brought its own) and its pipeline.
;; `skip`: the baseline itself does not build for this target, so there is nothing to compare with.
(define ($evo-row clean-env env src pipeline target ms)
  (let* ([k0 #f] [base #f] [bo #f]
         [setup (guard (e [#t ($tl-error-message e)])
                  (set! k0 (eval (car ($evo-read-all src)) clean-env))
                  (set! base ($evo-optimize k0 $evo-default-passes))
                  (set! bo ($evo-compile target base))
                  #f)])
    (if setup
        (list src 'skip setup 0 0 #f #f #f #f)
        (guard (e [#t (list src 'error ($tl-error-message e) 0 0 #f #f #f #f)])
          (let* ([kc ($evo-timed ms (lambda () (eval (car ($evo-read-all src)) env)))]
                 [cand ($evo-timed ms (lambda () ($evo-optimize kc pipeline)))]
                 [problem ($evo-structure-problem base cand)])
            (if problem
                (list src 'structure problem 0 0 #f #f #f #f)
                (let ([co ($evo-timed ms (lambda () ($evo-compile target cand)))])
                  ;; `cmat`: optimize leaves kernels with cooperative matrices alone, so no pass can change this one
                  (list src 'ok (if ($evo-has-cmat? base) "cmat" "") ($evo-size bo) ($evo-size co) (and (bytevector? co) co) (and (bytevector? bo) bo)
                        ;; PTX / MSL: the whole compile reply (text entry accesses push-bytes workgroup), for running it
                        (and (pair? co) co) (and (pair? bo) bo)))))))))

;; (load-error message) or (rows row ...). Candidate code is evaluated once in a fresh environment; the pass registry
;; is restored afterwards so one candidate cannot leak into the next. `gensrc` is the text of a candidate
;; `(library (tl tree kernels-candidate) ...)`, a modified copy of the generators ("" for none): the candidate's
;; expressions are then built by it, the baseline's by the original.
(define ($evo-evaluate code pipeline target exprs ms gensrc)
  (let ([saved (eval '(pass-registry) $evo-base-env)])
    (dynamic-wind
      void
      (lambda ()
        (let ([gen-error
               (and (not (string=? gensrc ""))
                    (guard (e [#t (string-append "[generators] " ($tl-error-message e))])
                      ($evo-timed ms (lambda () (for-each eval ($evo-read-all gensrc))))
                      #f))])
          (if gen-error
              (list 'load-error gen-error)
              (guard (e [#t (list 'load-error ($tl-error-message e))])
                (let* ([clean-env ($evo-new-env)]
                       [cand-base (if (string=? gensrc "")
                                      $evo-base-env
                                      (apply environment ($evo-env-spec '(tl tree kernels-candidate))))]
                       [env ($evo-new-env cand-base)])
                  ($evo-timed ms (lambda () (for-each (lambda (f) (eval f env)) ($evo-read-all code))))
                  (let ([known (eval '(pass-names) $evo-base-env)])
                    (for-each (lambda (s) (unless (memq s known) (error 'evaluate "no pass named" s))) pipeline))
                  (cons 'rows (map (lambda (src) ($evo-row clean-env env src pipeline target ms)) exprs)))))))
      (lambda () (eval (list 'restore-passes! (list 'quote saved)) $evo-base-env)))))

;; Which of these generator expressions build kernels that passes cannot change (cooperative matrices)?
(define ($evo-locked exprs)
  (let ([env ($evo-new-env)])
    (map (lambda (src)
           (guard (e [#t #f])
             ($evo-has-cmat? (eval (car ($evo-read-all src)) env))))
         exprs)))
