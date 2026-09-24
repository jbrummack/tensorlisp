;; Helpers the Rust side calls into. Each returns (#t . value) on success or
;; (#f . message) when a condition was raised, so Scheme errors never unwind
;; through Rust frames.
(begin
  (define ($tl-error-message e)
    (if (condition? e)
        (call-with-string-output-port (lambda (p) (display-condition e p)))
        (format "non-condition raised: ~s" e)))

  (define ($tl-protect thunk)
    (guard (e [#t (cons #f ($tl-error-message e))])
      (cons #t (thunk))))

  ;; Reads and evaluates every form in s, returning the last value.
  (define ($tl-eval-string s)
    ($tl-protect
      (lambda ()
        (let ([p (open-input-string s)])
          (let loop ([result (void)])
            (let ([x (read p)])
              (if (eof-object? x)
                  result
                  (loop (eval x)))))))))

  (define ($tl-call name args)
    ($tl-protect (lambda () (apply (top-level-value name) args)))))
