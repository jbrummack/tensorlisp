;; (tl util), imported as util:  -- plain Scheme helpers for pipelines and
;; program code (strings, lists); no tensors.
(library (tl util)
  (export string-replace string-repeat string-join make-list pad-list last ->integers)
  (import (rnrs) (only (chezscheme) format quotient remainder))

  ;; s with every occurrence of `from` replaced by `to`.
  (define (string-replace s from to)
    (when (= (string-length from) 0) (error 'util:string-replace "empty search string"))
    (let ([n (string-length s)] [m (string-length from)])
      (let loop ([i 0] [start 0] [parts '()])
        (cond
          [(> (+ i m) n) (apply string-append (reverse (cons (substring s start n) parts)))]
          [(string=? (substring s i (+ i m)) from)
           (loop (+ i m) (+ i m) (cons to (cons (substring s start i) parts)))]
          [else (loop (+ i 1) start parts)]))))

  ;; s repeated n times.
  (define (string-repeat s n)
    (let loop ([n n] [acc '()]) (if (<= n 0) (apply string-append acc) (loop (- n 1) (cons s acc)))))

  ;; Strings joined with a separator.
  (define (string-join strings separator)
    (if (null? strings)
        ""
        (fold-left (lambda (acc s) (string-append acc separator s)) (car strings) (cdr strings))))

  (define (make-list n x) (let loop ([n n] [acc '()]) (if (<= n 0) acc (loop (- n 1) (cons x acc)))))

  ;; xs padded with `value` to length n (xs longer than n is an error).
  (define (pad-list xs n value)
    (when (> (length xs) n) (error 'util:pad-list (format "~a elements don't fit in ~a" (length xs) n)))
    (append xs (make-list (- n (length xs)) value)))

  (define (last xs)
    (when (null? xs) (error 'util:last "empty list"))
    (if (null? (cdr xs)) (car xs) (last (cdr xs))))

  ;; Numbers (e.g. from array->list, which returns floats) as exact integers.
  (define (->integers xs) (map (lambda (x) (exact (round x))) xs)))
