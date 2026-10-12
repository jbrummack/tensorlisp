;; (tl tree msl) -- the Metal Shading Language target of Tree (docs/design/tree.md): the same kernel
;; language as (tl tree), compiled to MSL source text for the Metal runtime compiler.
;;
;;   (compile-kernel-msl '(kernel ...)) => (msl-text entry-name bindings push-bytes workgroup)
;;
;; bindings is the access (ro | wo | rw) of every buffer in order, push-bytes the size of the
;; kernel's own push words (four bytes per member), workgroup the threadgroup shape. The calling
;; convention is `OffsetMode::Prefix` of the scheduler (kernels/src/metal/backend.rs): buffer i is
;; bound at [[buffer(i)]] rounded down to the offset alignment, and a `constant uint*` at
;; [[buffer(n)]] (n = number of buffers) holds one residual byte offset per buffer followed by the
;; push words. A kernel's pointer for buffer i is its base plus that residual.
;;
;; Every value is a C++ local named t<N>, assigned once (`var` locals v<N> are the mutable ones),
;; so the emitter keeps no register state: the Metal compiler allocates. The library is compiled
;; without fast math (the PTX target builds exp/tanh/... from approximate instructions with
;; corrections; here the precise library functions do the same job and -inf / NaN behave).
;; Subgroups are SIMD-groups (32 lanes on Apple GPUs).
(library (tl tree msl)
  (export compile-kernel-msl)
  (import (rnrs) (tl tree) (only (tl tree spirv) f32-bits) (only (chezscheme) format))

  (define (fmt . args) (apply format #f args))

  ;; ---- emitter state -------------------------------------------------------------------------

  (define-record-type ms
    (fields (mutable lines) (mutable prelude) (mutable counter) memo nbuf)
    (protocol (lambda (new) (lambda (nbuf) (new '() '() 0 (make-hashtable equal-hash equal?) nbuf)))))

  (define (ins! st . parts) (ms-lines-set! st (cons (apply fmt parts) (ms-lines st))))
  (define (pre! st . parts) (ms-prelude-set! st (cons (apply fmt parts) (ms-prelude st))))

  (define (fresh! st prefix)
    (let ([n (ms-counter st)])
      (ms-counter-set! st (+ n 1))
      (string-append prefix (number->string n))))

  (define (memo! st key build)
    (or (hashtable-ref (ms-memo st) key #f)
        (let ([v (build)]) (hashtable-set! (ms-memo st) key v) v)))

  (define (ctype ty)
    (case ty
      [(f32) "float"] [(i32) "int"] [(u32) "uint"] [(bool) "bool"] [(u16) "ushort"] [(f16) "half"]
      [else (error 'tree-msl "no C type" ty)]))

  (define (esize ty) (if (memq ty '(u16 f16)) 2 4))

  ;; `name = expr` as a fresh typed local; returns the (name . type) value.
  (define (def! st ty expr)
    (let ([n (fresh! st "t")])
      (ins! st "~a ~a = ~a;" (ctype ty) n expr)
      (cons n ty)))

  (define (hex n) (string-append "0x" (number->string n 16) "u"))
  (define (f32-lit x) (fmt "as_type<float>(~a)" (hex (f32-bits x))))

  (define (const! st ty v)
    (case ty
      [(f32) (def! st ty (f32-lit v))]
      [(u32) (def! st ty (hex (bitwise-and v #xffffffff)))]
      [(i32) (def! st ty (fmt "as_type<int>(~a)" (hex (bitwise-and v #xffffffff))))]
      [(bool) (def! st ty (if v "true" "false"))]
      [else (error 'tree-msl "no constants of this type" ty)]))

  (define (join strings sep)
    (if (null? strings) "" (fold-left (lambda (acc s) (string-append acc sep s)) (car strings) (cdr strings))))

  (define (ident s)
    (list->string (map (lambda (c) (if (or (char-alphabetic? c) (char-numeric? c) (char=? c #\_)) c #\_)) (string->list s))))

  ;; ---- parameters and builtins ---------------------------------------------------------------

  ;; The push word `idx` (after the residual offsets), loaded once.
  (define (push-value st idx ty)
    (memo! st (list 'push idx)
      (lambda ()
        (let ([n (fresh! st "p")] [word (fmt "pc[~a]" (+ (ms-nbuf st) idx))])
          (pre! st "~a ~a = ~a;" (ctype ty)
                n (case ty [(f32) (fmt "as_type<float>(~a)" word)] [(i32) (fmt "as_type<int>(~a)" word)] [else word]))
          n))))

  ;; The typed pointer of buffer `i`: base plus the residual byte offset.
  (define (buffer-ptr st i elem access)
    (memo! st (list 'ptr i)
      (lambda ()
        (let* ([n (fresh! st "q")] [c (if (eq? access 'ro) "const " "")])
          (pre! st "~adevice ~a* ~a = (~adevice ~a*)(b~a + pc[~a]);" c (ctype elem) n c (ctype elem) i i)
          n))))

  ;; ---- expressions -------------------------------------------------------------------------------

  (define (buffer-like form env name)
    (let ([b (lookup form env name)])
      (case (vector-ref b 0)
        [(buf shared) b]
        [else (fail form "~a is not a buffer or shared array" name)])))

  (define (cm-lookup form env name)
    (let ([b (lookup form env name)])
      (unless (eq? (vector-ref b 0) 'cmat) (fail form "~a is not a cooperative matrix" name))
      b))

  (define (cx-as st e env t form)
    (let ([v (cx st e env t)])
      (unless (eq? (cdr v) t) (fail form "expected ~a, got ~a for ~s" t (cdr v) e))
      v))

  (define (arith-op h t form)
    (let ([f (eq? t 'f32)])
      (case h
        [(+) "+"] [(-) "-"] [(*) "*"] [(/) "/"]
        [(rem) (if f (fail form "rem on f32") "%")]
        [(bit-and) "&"] [(bit-or) "|"] [(bit-xor) "^"] [(shl) "<<"] [(shr) ">>"]
        [else (fail form "no operator for ~a" h)])))

  (define (cmp-op h form)
    (case h [(<) "<"] [(<=) "<="] [(>) ">"] [(>=) ">="] [(=) "=="] [(!=) "!="]
      [else (fail form "no comparison ~a" h)]))

  (define (cast st form v t)
    (let ([s (cdr v)])
      (cond
        [(eq? s t) v]
        [(and (memq s '(f32 i32 u32)) (memq t '(f32 i32 u32))) (def! st t (fmt "(~a)(~a)" (ctype t) (car v)))]
        [(and (eq? s 'u16) (memq t '(u32 i32))) (def! st t (fmt "(~a)(~a)" (ctype t) (car v)))]
        [(and (memq s '(u32 i32)) (eq? t 'u16)) (def! st t (fmt "(ushort)(~a)" (car v)))]
        [(and (eq? s 'f32) (eq? t 'f16)) (def! st t (fmt "half(~a)" (car v)))]
        [(and (eq? s 'f16) (eq? t 'f32)) (def! st t (fmt "float(~a)" (car v)))]
        [else (fail form "cannot convert ~a to ~a" s t)])))

  ;; (name . type) of element `idx` of a buffer or shared array, as an lvalue string.
  (define (element st form env name idx)
    (let* ([b (buffer-like form env name)]
           [elem (vector-ref b 2)]
           [i (cx st idx env 'u32)])
      (unless (memq (cdr i) '(u32 i32)) (fail form "index must be an integer, got ~a" (cdr i)))
      (if (eq? (vector-ref b 0) 'buf)
          (values (fmt "~a[~a]" (buffer-ptr st (vector-ref b 4) elem (vector-ref b 3)) (car i)) elem (vector-ref b 3)
                  (fmt "~a + ~a" (buffer-ptr st (vector-ref b 4) elem (vector-ref b 3)) (car i)))
          (values (fmt "~a[~a]" (vector-ref b 1) (car i)) elem 'rw (fmt "~a + ~a" (vector-ref b 1) (car i))))))

  (define (dim-component form d)
    (unless (and (integer? d) (<= 0 d 2)) (fail form "dimension must be 0, 1 or 2"))
    (vector-ref '#("x" "y" "z") d))

  (define (cx st e env expected)
    (cond
      [(boolean? e) (const! st 'bool e)]
      [(and (number? e) (exact? e))
       (const! st (if (memq expected '(f32 i32 u32)) expected 'i32) e)]
      [(number? e) (const! st 'f32 e)]
      [(symbol? e)
       (let ([b (lookup e env e)])
         (case (vector-ref b 0)
           ;; a var is read as a snapshot: a later set! must not change a value already taken
           [(val) (cons (vector-ref b 1) (vector-ref b 2))]
           [(var) (def! st (vector-ref b 2) (vector-ref b 1))]
           [(push) (cons (push-value st (vector-ref b 1) (vector-ref b 2)) (vector-ref b 2))]
           [else (fail e "~a is a buffer, not a value" e)]))]
      [(pair? e) (cx-form st e env expected)]
      [else (fail e "cannot compile this form")]))

  (define (cx-form st e env expected)
    (let ([h (car e)] [args (cdr e)])
      (cond
        [(memq h (append arith-ops bit-ops))
         (when (null? args) (fail e "~a needs operands" h))
         (let* ([t (resolve (common e (map (lambda (a) (ty-of a env)) args)) expected)]
                [vals (map (lambda (a) (cx-as st a env t e)) args)])
           (when (eq? t 'bool) (fail e "~a on bool" h))
           (when (eq? t 'u16) (fail e "~a on u16: convert with (u32 ...) first" h))
           (when (eq? t 'f16) (fail e "~a on f16: f16 is for storage and matrices only" h))
           (when (and (memq h bit-ops) (eq? t 'f32)) (fail e "~a on f32" h))
           (cond
             [(and (eq? h '-) (null? (cdr vals))) (def! st t (fmt "-~a" (car (car vals))))]
             [(memq h '(min max))
              (let ([fn (if (eq? t 'f32) (if (eq? h 'min) "fmin" "fmax") (symbol->string h))])
                (fold-left (lambda (acc v) (def! st t (fmt "~a(~a, ~a)" fn (car acc) (car v)))) (car vals) (cdr vals)))]
             [(eq? h 'pow)
              (unless (eq? t 'f32) (fail e "pow needs f32"))
              (fold-left (lambda (acc v) (def! st t (fmt "pow(~a, ~a)" (car acc) (car v)))) (car vals) (cdr vals))]
             [else
              (let ([op (arith-op h t e)])
                (fold-left (lambda (acc v) (def! st t (fmt "~a ~a ~a" (car acc) op (car v)))) (car vals) (cdr vals)))]))]
        [(memq h cmp-ops)
         (unless (= (length args) 2) (fail e "~a takes two operands" h))
         (let* ([t (resolve (common e (map (lambda (a) (ty-of a env)) args)) #f)]
                [a (cx-as st (car args) env t e)]
                [b (cx-as st (cadr args) env t e)])
           (def! st 'bool (fmt "~a ~a ~a" (car a) (cmp-op h e) (car b))))]
        [(memq h '(and or))
         (let ([vals (map (lambda (a) (cx-as st a env 'bool e)) args)])
           (fold-left (lambda (acc v) (def! st 'bool (fmt "~a ~a ~a" (car acc) (if (eq? h 'and) "&&" "||") (car v)))) (car vals) (cdr vals)))]
        [(eq? h 'not) (def! st 'bool (fmt "!~a" (car (cx-as st (car args) env 'bool e))))]
        [(eq? h 'load)
         (let-values ([(lvalue elem access ptr) (element st e env (car args) (cadr args))])
           (def! st elem lvalue))]
        [(memq h '(f32 i32 u32 u16 f16))
         (let* ([t0 (ty-of (car args) env)]
                [v (cx st (car args) env (if (memq t0 '(ilit flit)) (if (eq? t0 'ilit) h 'f32) #f))])
           (cast st e v h))]
        [(eq? h 'guard)
         ;; (guard c expr default): expr is only evaluated (and so only loads) when c holds
         (let* ([t (resolve (common e (map (lambda (a) (ty-of a env)) (cdr args))) expected)]
                [c (cx-as st (car args) env 'bool e)]
                [d (cx-as st (caddr args) env t e)]
                [out (def! st t (car d))])
           (ins! st "if (~a) {" (car c))
           (let ([v (cx-as st (cadr args) env t e)])
             (ins! st "~a = ~a;" (car out) (car v)))
           (ins! st "}")
           out)]
        [(eq? h 'select)
         (let* ([t (resolve (common e (map (lambda (a) (ty-of a env)) (cdr args))) expected)]
                [c (cx-as st (car args) env 'bool e)]
                [a (cx-as st (cadr args) env t e)]
                [b (cx-as st (caddr args) env t e)])
           (def! st t (fmt "~a ? ~a : ~a" (car c) (car a) (car b))))]
        [(memq h float-unary)
         (let ([v (car (cx-as st (car args) env 'f32 e))])
           (def! st 'f32 (fmt "~a(~a)" (case h [(inv-sqrt) "rsqrt"] [else (symbol->string h)]) v)))]
        [(eq? h 'abs)
         (let* ([t (resolve (ty-of (car args) env) expected)] [v (cx-as st (car args) env t e)])
           (case t
             [(f32) (def! st t (fmt "fabs(~a)" (car v)))]
             [(i32) (def! st t (fmt "(int)abs(~a)" (car v)))]
             [else v]))]
        [(eq? h 'fma)
         (let ([vs (map (lambda (a) (car (cx-as st a env 'f32 e))) args)])
           (def! st 'f32 (fmt "fma(~a, ~a, ~a)" (car vs) (cadr vs) (caddr vs))))]
        [(eq? h 'dp4a)
         ;; (dp4a a b acc): acc + the dot product of the four signed bytes of a and of b
         (let* ([a (cast st e (cx st (car args) env 'u32) 'u32)]
                [b (cast st e (cx st (cadr args) env 'u32) 'u32)]
                [c (cx-as st (caddr args) env 'i32 e)]
                [x (fresh! st "c")] [y (fresh! st "c")])
           (ins! st "char4 ~a = as_type<char4>(~a);" x (car a))
           (ins! st "char4 ~a = as_type<char4>(~a);" y (car b))
           (def! st 'i32 (fmt "~a + int(~a.x) * int(~a.x) + int(~a.y) * int(~a.y) + int(~a.z) * int(~a.z) + int(~a.w) * int(~a.w)"
                              (car c) x y x y x y x y)))]
        [(eq? h 'shuffle-xor)
         ;; (shuffle-xor v mask): the value of v in lane (lane xor mask) of the SIMD-group
         (let* ([t (resolve (ty-of (car args) env) expected)] [v (cx-as st (car args) env t e)])
           (unless (and (integer? (cadr args)) (exact? (cadr args))) (fail e "the shuffle mask must be a literal"))
           (def! st t (fmt "simd_shuffle_xor(~a, (ushort)~a)" (car v) (cadr args))))]
        [(eq? h 'half->f32)
         (let* ([v (cx st (car args) env 'u32)])
           (def! st 'f32 (if (eq? (cdr v) 'f16) (fmt "float(~a)" (car v)) (fmt "float(as_type<half>((ushort)~a))" (car v)))))]
        [(eq? h 'f32->half)
         (let* ([v (cx-as st (car args) env 'f32 e)])
           (def! st 'u32 (fmt "(uint)as_type<ushort>(half(~a))" (car v))))]
        [(memq h '(sbits ubits))
         (let* ([t (if (eq? h 'sbits) 'i32 'u32)]
                [x (cast st e (cx st (car args) env t) t)]
                [off (cx-as st (cadr args) env 'u32 e)]
                [cnt (cx-as st (caddr args) env 'u32 e)])
           (def! st t (fmt "extract_bits(~a, ~a, ~a)" (car x) (car off) (car cnt))))]
        [(memq h '(local-id group-id num-groups global-id))
         (let ([c (dim-component e (car args))])
           (def! st 'u32 (case h
                           [(local-id) (fmt "lid.~a" c)]
                           [(group-id) (fmt "grp.~a" c)]
                           [(num-groups) (fmt "ngr.~a" c)]
                           [else (fmt "(grp.~a * tgs.~a + lid.~a)" c c c)])))]
        [(eq? h 'lane) (def! st 'u32 "lane")]
        [(eq? h 'subgroup-size) (const! st 'u32 32)]
        [(eq? h 'subgroup-id) (def! st 'u32 "sgi")]
        [(eq? h 'num-subgroups) (def! st 'u32 "nsg")]
        [(memq h '(subgroup-add subgroup-max subgroup-min))
         (let* ([t (resolve (ty-of (car args) env) expected)] [v (cx-as st (car args) env t e)])
           (when (memq t '(bool u16)) (fail e "~a on ~a" h t))
           (def! st t (fmt "~a(~a)" (case h [(subgroup-add) "simd_sum"] [(subgroup-max) "simd_max"] [else "simd_min"]) (car v))))]
        [(eq? h 'let)
         (let loop ([bs (cadr e)] [env env])
           (if (null? bs)
               (cx st (caddr e) env expected)
               (let* ([t (resolve (ty-of (cadar bs) env) #f)] [v (cx-as st (cadar bs) env t e)])
                 (loop (cdr bs) (cons (cons (caar bs) (vector 'val (car v) t)) env)))))]
        [else (fail e "unknown operator ~a" h)])))

  ;; ---- statements --------------------------------------------------------------------------------

  (define (cstmts st stmts env)
    (let loop ([stmts stmts] [env env])
      (if (null? stmts) env (loop (cdr stmts) (cs st (car stmts) env)))))

  ;; A statement list in braces: locals declared inside are not visible after it.
  (define (block st stmts env)
    (ins! st "{")
    (cstmts st stmts env)
    (ins! st "}"))

  (define (cs st s env)
    (unless (pair? s) (fail s "not a statement"))
    (let ([h (car s)] [args (cdr s)])
      (case h
        [(let)
         (let loop ([bs (car args)] [inner env])
           (if (null? bs)
               (cstmts st (cdr args) inner)
               (let* ([t (resolve (ty-of (cadar bs) inner) #f)] [v (cx-as st (cadar bs) inner t s)])
                 (loop (cdr bs) (cons (cons (caar bs) (vector 'val (car v) t)) inner)))))
         env]
        [(var)
         (let* ([name (car args)] [t (cadr args)])
           (unless (value-type? t) (fail s "unknown type ~a" t))
           (let* ([init (cx-as st (caddr args) env t s)] [n (fresh! st "v")])
             (ins! st "~a ~a = ~a;" (ctype t) n (car init))
             (cons (cons name (vector 'var n t)) env)))]
        [(set!)
         (let ([b (lookup s env (car args))])
           (unless (eq? (vector-ref b 0) 'var) (fail s "~a is not a var" (car args)))
           (let ([v (cx-as st (cadr args) env (vector-ref b 2) s)])
             (ins! st "~a = ~a;" (vector-ref b 1) (car v))
             env))]
        [(store)
         (let-values ([(lvalue elem access ptr) (element st s env (car args) (cadr args))])
           (let ([v (cx-as st (caddr args) env elem s)])
             (ins! st "~a = ~a;" lvalue (car v))
             env))]
        [(vload)
         ;; (vload (n0 n1 [n2 n3]) buf idx): one 2- or 4-wide load; idx * element size must be aligned
         ;; to the vector width (the lowering that picked this kernel guarantees it).
         (let* ([names (car args)] [n (length names)]
                [b (buffer-like s env (cadr args))]
                [elem (vector-ref b 2)])
           (unless (and (memv n '(2 4)) (memq elem '(f32 u32 i32)) (eq? (vector-ref b 0) 'buf)) (fail s "vload needs 2 or 4 words from a global buffer"))
           (let-values ([(lvalue elem access ptr) (element st s env (cadr args) (caddr args))])
             (let ([v (fresh! st "w")] [vt (fmt "~a~a" (ctype elem) n)])
               (ins! st "~a ~a = *(~adevice ~a*)(~a);" vt v (if (eq? access 'ro) "const " "") vt ptr)
               (let loop ([names names] [comps '("x" "y" "z" "w")] [env env])
                 (if (null? names)
                     env
                     (let ([d (def! st elem (fmt "~a.~a" v (car comps)))])
                       (loop (cdr names) (cdr comps) (cons (cons (car names) (vector 'val (car d) elem)) env))))))))]
        [(begin) (block st args env) env]
        [(when unless)
         (let ([c (cx-as st (car args) env 'bool s)])
           (ins! st (if (eq? h 'when) "if (~a) {" "if (!~a) {") (car c))
           (cstmts st (cdr args) env)
           (ins! st "}")
           env)]
        [(if)
         (let ([c (cx-as st (car args) env 'bool s)])
           (ins! st "if (~a) {" (car c))
           (cs st (cadr args) env)
           (when (= (length args) 3)
             (ins! st "} else {")
             (cs st (caddr args) env))
           (ins! st "}")
           env)]
        [(for) (cfor st s env) env]
        [(barrier) (ins! st "threadgroup_barrier(mem_flags::mem_threadgroup);") env]
        [(cmat)
         ;; (cmat name use type): a 16x16 matrix held as a 2x2 grid of 8x8 simdgroup matrices m[2 * row + col]
         (let* ([name (car args)] [use (cadr args)] [ty (caddr args)]
                [ok (or (and (memq use '(a b)) (eq? ty 'f16)) (and (eq? use 'c) (eq? ty 'f32)))])
           (unless ok (fail s "the Metal target has a and b matrices of f16 and c matrices of f32"))
           (let ([n (fresh! st "m")])
             (ins! st "simdgroup_~a8x8 ~a[4];" (if (eq? ty 'f16) "half" "float") n)
             (cons (cons name (vector 'cmat n use ty)) env)))]
        [(cm-zero)
         (let ([n (vector-ref (cm-lookup s env (car args)) 1)])
           (do ([i 0 (+ i 1)]) ((= i 4) env)
             (ins! st "~a[~a] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);" n i)))]
        [(cm-load cm-store)
         ;; (cm-load M buf idx stride row|col), (cm-store ...): the 16x16 matrix whose corner is element
         ;; idx of buf; row: element (r, c) at idx + r * stride + c, col: at idx + c * stride + r.
         (let* ([b (cm-lookup s env (car args))] [n (vector-ref b 1)]
                [layout (car (cddddr args))]
                [stride (car (cx-as st (cadddr args) env 'u32 s))]
                [load? (eq? h 'cm-load)])
           (unless (memq layout '(row col)) (fail s "layout must be row or col"))
           (when (and (not load?) (not (eq? (vector-ref b 2) 'c))) (fail s "cm-store stores accumulators"))
           (let-values ([(lvalue elem access ptr) (element st s env (cadr args) (caddr args))])
             (let* ([bufb (buffer-like s env (cadr args))]
                    [space (if (eq? (vector-ref bufb 0) 'shared) "threadgroup" "device")]
                    [want (if (eq? (vector-ref b 2) 'c) 'f32 'f16)]
                    )
               (unless (or (eq? elem want) (and (eq? want 'f16) (eq? elem 'u16)))
                 (fail s "cm-~a: ~a elements expected, the buffer holds ~a" (if load? "load" "store") want elem))
               (do ([r 0 (+ r 1)]) ((= r 2) env)
                 (do ([c 0 (+ c 1)]) ((= c 2))
                   (let ([origin (if (eq? layout 'row)
                                     (fmt "~a * 8 * ~a + ~a" r stride (* c 8))
                                     (fmt "~a * 8 * ~a + ~a" c stride (* r 8)))]
                         [tr (if (eq? layout 'row) "false" "true")])
                     (if load?
                         (ins! st "simdgroup_load(~a[~a], (const ~a ~a*)(~a + ~a), (ulong)~a, 0, ~a);"
                               n (+ (* 2 r) c) space (ctype 'f16) ptr origin stride tr)
                         (ins! st "simdgroup_store(~a[~a], (~a float*)(~a + ~a), (ulong)~a, 0, ~a);"
                               n (+ (* 2 r) c) space ptr origin stride tr))))))))]
        [(cm-muladd)
         ;; (cm-muladd dst a b c): dst = a * b + c
         (let ([d (vector-ref (cm-lookup s env (car args)) 1)] [a (vector-ref (cm-lookup s env (cadr args)) 1)]
               [b (vector-ref (cm-lookup s env (caddr args)) 1)] [c (vector-ref (cm-lookup s env (cadddr args)) 1)])
           ;; k outermost: the four accumulator tiles are independent, so consecutive multiplies do not wait
           ;; for each other (the two steps of one tile are four multiplies apart)
           (unless (equal? c d)
             (do ([t 0 (+ t 1)]) ((= t 4))
               (ins! st "~a[~a] = ~a[~a];" d t c t)))
           (do ([k 0 (+ k 1)]) ((= k 2))
             (do ([i 0 (+ i 1)]) ((= i 2))
               (do ([j 0 (+ j 1)]) ((= j 2))
                 (let ([dij (fmt "~a[~a]" d (+ (* 2 i) j))])
                   (ins! st "simdgroup_multiply_accumulate(~a, ~a[~a], ~a[~a], ~a);" dij a (+ (* 2 i) k) b (+ (* 2 k) j) dij)))))
           env)]
        [(mma-s8)
         (fail s "~a: int8 matrix products are not available on the Metal target" h)]
        [else (fail s "unknown statement ~a" h)])))

  ;; Loops with literal bounds that run few times are unrolled by the Metal compiler on request;
  ;; the rest are counted loops.
  (define max-unroll 32)

  (define (cfor st s env)
    (let* ([spec (cadr s)] [name (car spec)] [lo (cadr spec)] [hi (caddr spec)]
           [step (if (pair? (cdddr spec)) (cadddr spec) 1)]
           [t0 (common s (map (lambda (x) (ty-of x env)) (list lo hi step)))]
           [t (case t0 [(ilit) 'u32] [(f32 bool flit) (fail s "loop bounds must be integers")] [else t0])]
           [i (fresh! st "i")]
           [small (and (integer? lo) (exact? lo) (integer? hi) (exact? hi) (integer? step) (exact? step) (> step 0)
                       (<= (div (+ (- hi lo) (- step 1)) step) max-unroll))]
           [lo-v (cx-as st lo env t s)] [hi-v (cx-as st hi env t s)] [step-v (cx-as st step env t s)])
      (when small (ins! st "#pragma unroll"))
      (ins! st "for (~a ~a = ~a; ~a < ~a; ~a += ~a) {" (ctype t) i (car lo-v) i (car hi-v) i (car step-v))
      (cstmts st (cddr s) (cons (cons name (vector 'val i t)) env))
      (ins! st "}")))

  ;; ---- kernels -----------------------------------------------------------------------------------

  (define (clause? f names) (and (pair? f) (memq (car f) names) #t))

  (define (compile-kernel-msl k)
    (unless (and (pair? k) (eq? (car k) 'kernel)) (error 'tree-msl "expected (kernel name clause ... body ...)" k))
    (let* ([forms (cddr k)]
           [clause-names '(workgroup buffers push shared offsets)]
           [clauses (filter (lambda (f) (clause? f clause-names)) forms)]
           [body (filter (lambda (f) (not (clause? f clause-names))) forms)]
           [find (lambda (name) (let ([c (filter (lambda (f) (eq? (car f) name)) clauses)])
                                  (if (null? c) '() (apply append (map cdr c)))))]
           [wg0 (find 'workgroup)]
           [wg (list (if (pair? wg0) (car wg0) 1)
                     (if (and (pair? wg0) (pair? (cdr wg0))) (cadr wg0) 1)
                     (if (and (pair? wg0) (pair? (cdr wg0)) (pair? (cddr wg0))) (caddr wg0) 1))]
           [buffer-specs (find 'buffers)]
           [push-specs (find 'push)]
           [shared-specs (find 'shared)]
           [nbuf (length buffer-specs)]
           [st (make-ms nbuf)]
           [buffers (let loop ([i 0] [bs buffer-specs] [acc '()])
                      (if (null? bs)
                          (reverse acc)
                          (loop (+ i 1) (cdr bs)
                                (cons (cons (caar bs) (vector 'buf (fmt "b~a" i) (caddr (car bs)) (cadr (car bs)) i)) acc))))]
           [pushes (let loop ([i 0] [ps push-specs] [acc '()])
                     (if (null? ps)
                         (reverse acc)
                         (loop (+ i 1) (cdr ps) (cons (cons (caar ps) (vector 'push i (cadar ps))) acc))))]
           [shareds (map (lambda (s) (cons (car s) (vector 'shared (string-append "sh_" (ident (symbol->string (car s)))) (cadr s) (caddr s)))) shared-specs)]
           [env (append buffers pushes shareds)]
           [name (ident (symbol->string (cadr k)))]
           [push-bytes (* 4 (length push-specs))])
      (cstmts st body env)
      (let* ([params (append
                       (let loop ([i 0] [bs buffer-specs] [acc '()])
                         (if (null? bs)
                             (reverse acc)
                             (loop (+ i 1) (cdr bs)
                                   (cons (fmt "    ~adevice uchar* b~a [[buffer(~a)]]" (if (eq? (cadr (car bs)) 'ro) "const " "") i i) acc))))
                       (if (> (+ nbuf (length push-specs)) 0) (list (fmt "    constant uint* pc [[buffer(~a)]]" nbuf)) '())
                       (list "    uint3 lid [[thread_position_in_threadgroup]]"
                             "    uint3 grp [[threadgroup_position_in_grid]]"
                             "    uint3 ngr [[threadgroups_per_grid]]"
                             "    uint3 tgs [[threads_per_threadgroup]]"
                             "    uint lane [[thread_index_in_simdgroup]]"
                             "    uint sgi [[simdgroup_index_in_threadgroup]]"
                             "    uint nsg [[simdgroups_per_threadgroup]]"))]
             [decls (map (lambda (s) (fmt "  threadgroup ~a sh_~a[~a];" (ctype (cadr s)) (ident (symbol->string (car s))) (caddr s))) shared-specs)]
             [text (join (append
                           (list "#include <metal_stdlib>" "using namespace metal;" ""
                                 (fmt "[[max_total_threads_per_threadgroup(~a)]]" (* (car wg) (cadr wg) (caddr wg)))
                                 (fmt "kernel void ~a(" name)
                                 (join params ",\n")
                                 ") {")
                           decls
                           (map (lambda (l) (string-append "  " l)) (reverse (ms-prelude st)))
                           (map (lambda (l) (string-append "  " l)) (reverse (ms-lines st)))
                           (list "}" ""))
                         "\n")])
        (list text name (map cadr buffer-specs) push-bytes wg))))
)
