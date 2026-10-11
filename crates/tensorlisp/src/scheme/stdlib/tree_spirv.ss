;; (tl tree spirv) -- the SPIR-V half of the Tree compiler: a module under
;; construction (ids, interned types and constants, instruction sections) and
;; the binary encoder. Instructions are lists (opcode operand ...), operands
;; being unsigned integers (words) or strings (literal strings, NUL padded).
;; The language front end is (tl tree).
(library (tl tree spirv)
  (export new-module fresh! capability! extension! import-ext! annotate! global! intern! cached! memo-ref
          body! prelude! local-var! interface! module-bytes f32-bits)
  (import (rnrs))

  (define-record-type module
    (fields (mutable next) (mutable caps) (mutable exts) (mutable imports) (mutable annot) (mutable globals)
            (mutable body) (mutable prelude) (mutable locals) (mutable interface)
            interned memo)
    (protocol
      (lambda (new)
        (lambda ()
          (new 1 '(1) '() '() '() '() '() '() '() '()
               (make-hashtable equal-hash equal?) (make-hashtable equal-hash equal?))))))

  (define (new-module) (make-module))

  (define (fresh! m)
    (let ([id (module-next m)])
      (module-next-set! m (+ id 1))
      id))

  (define (extension! m name)
    (unless (member name (module-exts m))
      (module-exts-set! m (cons name (module-exts m)))))

  (define (capability! m cap)
    (unless (memv cap (module-caps m))
      (module-caps-set! m (cons cap (module-caps m)))))

  ;; The id of an extended instruction set, imported on first use.
  (define (import-ext! m name)
    (cond
      [(assoc name (module-imports m)) => cdr]
      [else
       (let ([id (fresh! m)])
         (module-imports-set! m (cons (cons name id) (module-imports m)))
         id)]))

  (define (annotate! m . inst) (module-annot-set! m (cons inst (module-annot m))))
  (define (global! m . inst) (module-globals-set! m (cons inst (module-globals m))))
  (define (body! m . inst) (module-body-set! m (cons inst (module-body m))))
  (define (prelude! m . inst) (module-prelude-set! m (cons inst (module-prelude m))))
  (define (local-var! m . inst) (module-locals-set! m (cons inst (module-locals m))))
  (define (interface! m id)
    (unless (memv id (module-interface m))
      (module-interface-set! m (cons id (module-interface m)))))

  ;; The id registered under `key`; `build` is called with a fresh id the first time
  ;; and emits its declaration (types, constants, global variables).
  (define (intern! m key build)
    (or (hashtable-ref (module-interned m) key #f)
        (let ([id (fresh! m)])
          (hashtable-set! (module-interned m) key id)
          (build id)
          id)))

  (define (memo-ref m key) (hashtable-ref (module-memo m) key #f))

  ;; Front-end memo table (values computed once per kernel).
  (define (cached! m key build)
    (or (hashtable-ref (module-memo m) key #f)
        (let ([v (build)])
          (hashtable-set! (module-memo m) key v)
          v)))

  (define (f32-bits x)
    (let ([bv (make-bytevector 4 0)])
      (bytevector-ieee-single-set! bv 0 (inexact x) (endianness little))
      (bytevector-u32-ref bv 0 (endianness little))))

  ;; "main" -> little-endian words of its UTF-8 bytes plus a NUL, padded to four.
  (define (string-words s)
    (let* ([u (string->utf8 s)]
           [n (bytevector-length u)]
           [padded (* 4 (div (+ n 4) 4))]
           [bv (make-bytevector padded 0)])
      (bytevector-copy! u 0 bv 0 n)
      (let loop ([i 0] [acc '()])
        (if (= i padded)
            (reverse acc)
            (loop (+ i 4) (cons (bytevector-u32-ref bv i (endianness little)) acc))))))

  (define (operand-words ops)
    (let loop ([ops ops] [acc '()])
      (cond
        [(null? ops) (reverse acc)]
        [(string? (car ops)) (loop (cdr ops) (append (reverse (string-words (car ops))) acc))]
        [(and (integer? (car ops)) (<= 0 (car ops) #xffffffff)) (loop (cdr ops) (cons (car ops) acc))]
        [else (error 'tree-spirv "bad instruction operand" (car ops))])))

  (define (encode inst)
    (let* ([ops (operand-words (cdr inst))]
           [count (+ 1 (length ops))])
      (when (> count #xffff) (error 'tree-spirv "instruction too long" (car inst)))
      (cons (bitwise-ior (bitwise-arithmetic-shift-left count 16) (car inst)) ops)))

  ;; The finished module as a bytevector of little-endian words. `fn` is the entry
  ;; function's id, `fn-type` its (void) function type, `void` the void type.
  (define (module-bytes m name fn void fn-type entry-label workgroup)
    (let* ([ext-insts (map (lambda (p) (list 11 (cdr p) (car p))) (reverse (module-imports m)))]
           [caps (map (lambda (c) (list 17 c)) (list-sort < (module-caps m)))]
           [exts (map (lambda (e) (list 10 e)) (reverse (module-exts m)))]
           [entry (list 15 5 fn name)]
           [entry (append entry (reverse (module-interface m)))]
           [mode (list 16 fn 17 (car workgroup) (cadr workgroup) (caddr workgroup))]
           [insts (append
                    caps
                    exts
                    ext-insts
                    (list (list 14 0 (if (memv 5345 (module-caps m)) 3 1)) entry mode)
                    (reverse (module-annot m))
                    (reverse (module-globals m))
                    (list (list 54 void fn 0 fn-type) (list 248 entry-label))
                    (reverse (module-locals m))
                    (reverse (module-prelude m))
                    (reverse (module-body m))
                    (list (list 253) (list 56)))]
           [words (apply append (map encode insts))]
           [header (list #x07230203 #x00010300 0 (module-next m) 0)]
           [all (append header words)]
           [bv (make-bytevector (* 4 (length all)) 0)])
      (let loop ([ws all] [i 0])
        (unless (null? ws)
          (bytevector-u32-set! bv i (car ws) (endianness little))
          (loop (cdr ws) (+ i 4))))
      bv)))
