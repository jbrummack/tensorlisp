;; (tl tree ptx) -- the PTX target of Tree (docs/design/tree.md): the same kernel language as
;; (tl tree), compiled to PTX text for the CUDA driver's JIT (cuModuleLoadDataEx).
;;
;;   (compile-kernel-ptx '(kernel ...)) => (ptx-text entry-name bindings push-bytes workgroup)
;;
;; bindings is the access (ro | wo | rw) of every buffer in order, push-bytes the size of the
;; by-value parameter struct (four bytes per push member, `(offsets prefix)` words excluded: a
;; CUDA launch binds exact addresses), workgroup the block shape. The kernel takes one 64-bit
;; pointer per buffer and then the struct, the calling convention of kernels/src/cuda/backend.rs.
;;
;; Registers are typed and mutable (`var` is a register, `set!` a move); PTX is not SSA and ptxas
;; does the allocation. Subgroups are warps (32 lanes); a subgroup reduction is a butterfly of
;; shuffles. exp/log/sin/cos/tanh/pow are built from the approximate ex2/lg2/sin/cos instructions
;; with the corrections that keep them within a few ulp (sin/cos reduce their argument first).
(library (tl tree ptx)
  (export compile-kernel-ptx)
  (import (rnrs) (tl tree) (only (tl tree spirv) f32-bits) (only (chezscheme) format))

  (define (fmt . args) (apply format #f args))

  ;; ---- emitter state -------------------------------------------------------------------------

  (define-record-type ps
    (fields (mutable lines) (mutable prelude) counts memo (mutable labels))
    (protocol (lambda (new) (lambda () (new '() '() (make-eq-hashtable) (make-hashtable equal-hash equal?) 0)))))

  (define (ins! st . parts) (ps-lines-set! st (cons (apply fmt parts) (ps-lines st))))
  (define (pre! st . parts) (ps-prelude-set! st (cons (apply fmt parts) (ps-prelude st))))

  (define (reg! st cls)
    (let ([n (hashtable-ref (ps-counts st) cls 0)])
      (hashtable-set! (ps-counts st) cls (+ n 1))
      (string-append "%" (symbol->string cls) (number->string n))))

  (define (label! st)
    (let ([n (ps-labels st)])
      (ps-labels-set! st (+ n 1))
      (string-append "L" (number->string n))))

  (define (memo! st key build)
    (or (hashtable-ref (ps-memo st) key #f)
        (let ([v (build)]) (hashtable-set! (ps-memo st) key v) v)))

  (define (cls ty) (case ty [(f32) 'f] [(i32 u32) 'r] [(bool) 'p] [(u16 f16) 'h] [else (error 'tree-ptx "no register class" ty)]))
  (define (tsfx ty) (case ty [(f32) "f32"] [(u32) "u32"] [(i32) "s32"] [(u16) "u16"] [(f16) "b16"] [else (error 'tree-ptx "no type suffix" ty)]))
  (define (esize ty) (if (memq ty '(u16 f16)) 2 4))

  (define (hex8 n)
    (let ([s (string-upcase (number->string n 16))])
      (string-append (make-string (- 8 (string-length s)) #\0) s)))
  (define (f32-lit x) (string-append "0f" (hex8 (f32-bits x))))

  (define (mov! st ty dst src)
    (case ty
      [(f32) (ins! st "mov.f32 ~a, ~a;" dst src)]
      [(i32 u32) (ins! st "mov.b32 ~a, ~a;" dst src)]
      [(u16 f16) (ins! st "mov.b16 ~a, ~a;" dst src)]
      [(bool) (ins! st "mov.pred ~a, ~a;" dst src)]))

  (define (new-val st ty) (cons (reg! st (cls ty)) ty))

  (define (const! st ty v)
    (let ([r (new-val st ty)])
      (case ty
        [(f32) (ins! st "mov.f32 ~a, ~a;" (car r) (f32-lit v))]
        [(i32 u32) (ins! st "mov.b32 ~a, ~a;" (car r) (bitwise-and v #xffffffff))]
        [(bool) (ins! st (if v "setp.eq.u32 ~a, 0, 0;" "setp.ne.u32 ~a, 0, 0;") (car r))]
        [else (error 'tree-ptx "no constants of this type" ty)])
      r))

  ;; ---- builtins and parameters (loaded once, in the prelude) -------------------------------------------

  (define (special st name)
    (memo! st (list 'special name)
      (lambda ()
        (let ([r (reg! st 'r)]) (pre! st "mov.u32 ~a, ~a;" r name) r))))

  (define (dim-name d) (vector-ref '#("x" "y" "z") d))

  (define (global-id st d)
    (memo! st (list 'gid d)
      (lambda ()
        (let ([r (reg! st 'r)])
          (pre! st "mad.lo.u32 ~a, ~a, ~a, ~a;" r (special st (string-append "%ctaid." (dim-name d)))
                (special st (string-append "%ntid." (dim-name d))) (special st (string-append "%tid." (dim-name d))))
          r))))

  (define (push-value st idx ty)
    (memo! st (list 'push idx)
      (lambda ()
        (let ([r (reg! st (cls ty))])
          (pre! st "ld.param.~a ~a, [pc+~a];" (if (eq? ty 'f32) "f32" "u32") r (* 4 idx))
          r))))

  (define (buffer-base st pname)
    (memo! st (list 'base pname)
      (lambda ()
        (let ([r (reg! st 'rd)]) (pre! st "ld.param.u64 ~a, [~a];" r pname) r))))

  (define (shared-base st sname)
    (memo! st (list 'sbase sname)
      (lambda ()
        (let ([r (reg! st 'r)]) (pre! st "mov.u32 ~a, ~a;" r sname) r))))

  ;; ---- expressions -------------------------------------------------------------------------------

  (define (buffer-like form env name)
    (let ([b (lookup form env name)])
      (case (vector-ref b 0)
        [(buf shared) b]
        [else (fail form "~a is not a buffer or shared array" name)])))

  (define (cx-as st e env t form)
    (let ([v (cx st e env t)])
      (unless (eq? (cdr v) t) (fail form "expected ~a, got ~a for ~s" t (cdr v) e))
      v))

  (define (unop st fmt-string ty a)
    (let ([d (new-val st ty)])
      (ins! st fmt-string (car d) a)
      d))

  (define (binop st op ty a b)
    (let ([d (new-val st ty)])
      (ins! st "~a ~a, ~a, ~a;" op (car d) a b)
      d))

  (define (arith-op h t form)
    (let ([f (eq? t 'f32)] [u (eq? t 'u32)])
      (case h
        [(+) (cond [f "add.f32"] [u "add.u32"] [else "add.s32"])]
        [(-) (cond [f "sub.f32"] [u "sub.u32"] [else "sub.s32"])]
        [(*) (cond [f "mul.f32"] [u "mul.lo.u32"] [else "mul.lo.s32"])]
        [(/) (cond [f "div.rn.f32"] [u "div.u32"] [else "div.s32"])]
        [(rem) (cond [f (fail form "rem on f32")] [u "rem.u32"] [else "rem.s32"])]
        [(min) (cond [f "min.f32"] [u "min.u32"] [else "min.s32"])]
        [(max) (cond [f "max.f32"] [u "max.u32"] [else "max.s32"])]
        [(bit-and) "and.b32"] [(bit-or) "or.b32"] [(bit-xor) "xor.b32"] [(shl) "shl.b32"]
        [(shr) (if u "shr.u32" "shr.s32")]
        [else (fail form "no instruction for ~a" h)])))

  (define (cmp-op h t form)
    (let ([name (case h [(<) "lt"] [(<=) "le"] [(>) "gt"] [(>=) "ge"] [(=) "eq"] [(!=) "ne"]
                  [else (fail form "no comparison ~a" h)])])
      (string-append "setp." name "." (case t [(f32) "f32"] [(u32) "u32"] [(i32) "s32"] [else "u32"]))))

  (define (cast st form v t)
    (let ([s (cdr v)])
      (cond
        [(eq? s t) v]
        [(and (eq? s 'f32) (eq? t 'i32)) (unop st "cvt.rzi.s32.f32 ~a, ~a;" t (car v))]
        [(and (eq? s 'f32) (eq? t 'u32)) (unop st "cvt.rzi.u32.f32 ~a, ~a;" t (car v))]
        [(and (eq? s 'i32) (eq? t 'f32)) (unop st "cvt.rn.f32.s32 ~a, ~a;" t (car v))]
        [(and (eq? s 'u32) (eq? t 'f32)) (unop st "cvt.rn.f32.u32 ~a, ~a;" t (car v))]
        [(and (memq s '(i32 u32)) (memq t '(i32 u32))) (unop st "mov.b32 ~a, ~a;" t (car v))]
        [(and (eq? s 'u16) (eq? t 'u32)) (unop st "cvt.u32.u16 ~a, ~a;" t (car v))]
        [(and (eq? s 'u16) (eq? t 'i32)) (unop st "cvt.u32.u16 ~a, ~a;" t (car v))]
        [(and (memq s '(u32 i32)) (eq? t 'u16)) (unop st "cvt.u16.u32 ~a, ~a;" t (car v))]
        [(and (eq? s 'f32) (eq? t 'f16)) (unop st "cvt.rn.f16.f32 ~a, ~a;" t (car v))]
        [(and (eq? s 'f16) (eq? t 'f32)) (unop st "cvt.f32.f16 ~a, ~a;" t (car v))]
        [else (fail form "cannot convert ~a to ~a" s t)])))

  ;; Approximate-instruction building blocks.
  (define log2e (f32-lit 1.4426950408889634))
  (define ln2 (f32-lit 0.6931471805599453))

  (define (f-exp st x)
    ;; 2^(x log2e) with the rounding error of the product folded back in.
    (let ([t (new-val st 'f32)] [nt (new-val st 'f32)] [u (new-val st 'f32)] [d (new-val st 'f32)] [e (new-val st 'f32)] [r (new-val st 'f32)])
      (ins! st "mul.f32 ~a, ~a, ~a;" (car t) x log2e)
      (ins! st "neg.f32 ~a, ~a;" (car nt) (car t))
      (ins! st "fma.rn.f32 ~a, ~a, ~a, ~a;" (car u) x log2e (car nt))
      (ins! st "ex2.approx.ftz.f32 ~a, ~a;" (car d) (car t))
      (ins! st "mul.f32 ~a, ~a, ~a;" (car e) (car u) ln2)
      (ins! st "fma.rn.f32 ~a, ~a, ~a, ~a;" (car r) (car d) (car e) (car d))
      ;; The correction term is inf - inf = NaN at the ends of the range: exp(-inf) = 0, exp(+inf) = inf
      ;; (masked attention scores are -inf). NaN inputs stay NaN: both comparisons are false for them.
      (let ([lo (new-val st 'bool)] [hi (new-val st 'bool)] [r0 (new-val st 'f32)] [r1 (new-val st 'f32)])
        (ins! st "setp.lt.f32 ~a, ~a, 0fC2D20000;" (car lo) x)
        (ins! st "selp.f32 ~a, 0f00000000, ~a, ~a;" (car r0) (car r) (car lo))
        (ins! st "setp.gt.f32 ~a, ~a, 0f42D20000;" (car hi) x)
        (ins! st "selp.f32 ~a, 0f7F800000, ~a, ~a;" (car r1) (car r0) (car hi))
        r1)))

  (define (f-tanh st x)
    ;; tanh x = sign(x) (1 - e) / (1 + e), e = exp(-2|x|)
    (let ([a (new-val st 'f32)] [n (new-val st 'f32)] [one (const! st 'f32 1.0)] [num (new-val st 'f32)] [den (new-val st 'f32)] [q (new-val st 'f32)] [d (new-val st 'f32)])
      (ins! st "abs.f32 ~a, ~a;" (car a) x)
      (ins! st "mul.f32 ~a, ~a, ~a;" (car n) (car a) (f32-lit -2.0))
      (let ([e (f-exp st (car n))])
        (ins! st "sub.f32 ~a, ~a, ~a;" (car num) (car one) (car e))
        (ins! st "add.f32 ~a, ~a, ~a;" (car den) (car one) (car e))
        (ins! st "div.rn.f32 ~a, ~a, ~a;" (car q) (car num) (car den))
        (ins! st "copysign.f32 ~a, ~a, ~a;" (car d) x (car q))
        d)))

  (define (f-trig st fn x)
    ;; reduce to [-pi, pi] with the two-part 2 pi, then the hardware sine / cosine
    (let ([t (new-val st 'f32)] [k (new-val st 'f32)] [r1 (new-val st 'f32)] [r2 (new-val st 'f32)] [d (new-val st 'f32)])
      (ins! st "mul.f32 ~a, ~a, ~a;" (car t) x (f32-lit 0.15915494309189535))
      (ins! st "cvt.rni.f32.f32 ~a, ~a;" (car k) (car t))
      (ins! st "fma.rn.f32 ~a, ~a, ~a, ~a;" (car r1) (car k) (f32-lit -6.2831854820251465) x)
      (ins! st "fma.rn.f32 ~a, ~a, ~a, ~a;" (car r2) (car k) (f32-lit 1.7484556e-7) (car r1))
      (ins! st "~a.approx.ftz.f32 ~a, ~a;" (if (eq? fn 'sin) "sin" "cos") (car d) (car r2))
      d))

  ;; Butterfly reduction over the 32 lanes of a warp: every lane ends with the total.
  (define (warp-reduce st h t v)
    (let ([acc (new-val st t)] [tmp (new-val st t)])
      (mov! st t (car acc) (car v))
      (for-each
        (lambda (off)
          (ins! st "shfl.sync.bfly.b32 ~a, ~a, ~a, 31, 0xffffffff;" (car tmp) (car acc) off)
          (ins! st "~a ~a, ~a, ~a;" (arith-op (case h [(subgroup-add) '+] [(subgroup-max) 'max] [else 'min]) t h) (car acc) (car acc) (car tmp)))
        '(16 8 4 2 1))
      acc))

  (define (element-addr st form env name idx)
    (let* ([b (buffer-like form env name)]
           [elem (vector-ref b 2)]
           [i (cx st idx env 'u32)])
      (unless (memq (cdr i) '(u32 i32)) (fail form "index must be an integer, got ~a" (cdr i)))
      (if (eq? (vector-ref b 0) 'buf)
          (let ([off (reg! st 'rd)] [addr (reg! st 'rd)])
            (ins! st "mul.wide.u32 ~a, ~a, ~a;" off (car i) (esize elem))
            (ins! st "add.u64 ~a, ~a, ~a;" addr (buffer-base st (vector-ref b 1)) off)
            (values addr elem (vector-ref b 3) 'global))
          (let ([addr (reg! st 'r)])
            (ins! st "mad.lo.u32 ~a, ~a, ~a, ~a;" addr (car i) (esize elem) (shared-base st (vector-ref b 1)))
            (values addr elem 'rw 'shared)))))

  (define (cx st e env expected)
    (cond
      [(boolean? e) (const! st 'bool e)]
      [(and (number? e) (exact? e))
       (const! st (if (memq expected '(f32 i32 u32)) expected 'i32) e)]
      [(number? e) (const! st 'f32 e)]
      [(symbol? e)
       (let ([b (lookup e env e)])
         (case (vector-ref b 0)
           [(val) (cons (vector-ref b 1) (vector-ref b 2))]
           [(var) (let ([d (new-val st (vector-ref b 2))])
                    (mov! st (vector-ref b 2) (car d) (vector-ref b 1))
                    d)]
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
             [(and (eq? h '-) (null? (cdr vals)))
              (unop st (if (eq? t 'f32) "neg.f32 ~a, ~a;" "neg.s32 ~a, ~a;") t (car (car vals)))]
             [(eq? h 'pow)
              (unless (eq? t 'f32) (fail e "pow needs f32"))
              (fold-left
                (lambda (acc v)
                  (let ([l (new-val st 'f32)] [m (new-val st 'f32)])
                    (ins! st "lg2.approx.ftz.f32 ~a, ~a;" (car l) (car acc))
                    (ins! st "mul.f32 ~a, ~a, ~a;" (car m) (car v) (car l))
                    (let ([d (new-val st 'f32)])
                      (ins! st "ex2.approx.ftz.f32 ~a, ~a;" (car d) (car m))
                      d)))
                (car vals) (cdr vals))]
             [else
              (let ([op (arith-op h t e)])
                (fold-left (lambda (acc v) (binop st op t (car acc) (car v))) (car vals) (cdr vals)))]))]
        [(memq h cmp-ops)
         (unless (= (length args) 2) (fail e "~a takes two operands" h))
         (let* ([t (resolve (common e (map (lambda (a) (ty-of a env)) args)) #f)]
                [a (cx-as st (car args) env t e)]
                [b (cx-as st (cadr args) env t e)]
                [d (new-val st 'bool)])
           (cond
             [(eq? t 'bool)
              (let ([x (new-val st 'bool)])
                (ins! st "xor.pred ~a, ~a, ~a;" (car x) (car a) (car b))
                (if (eq? h '=) (ins! st "not.pred ~a, ~a;" (car d) (car x)) (mov! st 'bool (car d) (car x))))]
             [else (ins! st "~a ~a, ~a, ~a;" (cmp-op h t e) (car d) (car a) (car b))])
           d)]
        [(memq h '(and or))
         (let ([vals (map (lambda (a) (cx-as st a env 'bool e)) args)])
           (fold-left (lambda (acc v) (binop st (if (eq? h 'and) "and.pred" "or.pred") 'bool (car acc) (car v))) (car vals) (cdr vals)))]
        [(eq? h 'not) (unop st "not.pred ~a, ~a;" 'bool (car (cx-as st (car args) env 'bool e)))]
        [(eq? h 'load)
         (let-values ([(addr elem access space) (element-addr st e env (car args) (cadr args))])
           (let ([d (new-val st elem)])
             (ins! st "ld.~a~a.~a ~a, [~a];"
                   (if (eq? space 'shared) "shared" "global")
                   (if (and (eq? space 'global) (eq? access 'ro)) ".nc" "")
                   (tsfx elem) (car d) addr)
             d))]
        [(memq h '(f32 i32 u32 u16 f16))
         (let* ([t0 (ty-of (car args) env)]
                [v (cx st (car args) env (if (memq t0 '(ilit flit)) (if (eq? t0 'ilit) h 'f32) #f))])
           (cast st e v h))]
        [(eq? h 'guard)
         ;; A branch around e (measured faster than predicating its instructions: ptxas schedules
         ;; the straight-line code between the branches better than one long predicated block).
         (let* ([t (resolve (common e (map (lambda (a) (ty-of a env)) (cdr args))) expected)]
                [c (cx-as st (car args) env 'bool e)]
                [d (cx-as st (caddr args) env t e)]
                [out (new-val st t)]
                [skip (label! st)])
           (mov! st t (car out) (car d))
           (ins! st "@!~a bra ~a;" (car c) skip)
           (let ([v (cx-as st (cadr args) env t e)])
             (mov! st t (car out) (car v)))
           (ins! st "~a:" skip)
           out)]
        [(eq? h 'select)
         (let* ([t (resolve (common e (map (lambda (a) (ty-of a env)) (cdr args))) expected)]
                [c (cx-as st (car args) env 'bool e)]
                [a (cx-as st (cadr args) env t e)]
                [b (cx-as st (caddr args) env t e)]
                [d (new-val st t)])
           (ins! st "selp.~a ~a, ~a, ~a, ~a;" (if (eq? t 'f32) "f32" "b32") (car d) (car a) (car b) (car c))
           d)]
        [(memq h float-unary)
         (let ([v (car (cx-as st (car args) env 'f32 e))])
           (case h
             [(exp) (f-exp st v)]
             [(log) (let ([l (new-val st 'f32)] [d (new-val st 'f32)])
                      (ins! st "lg2.approx.ftz.f32 ~a, ~a;" (car l) v)
                      (ins! st "mul.f32 ~a, ~a, ~a;" (car d) (car l) ln2)
                      d)]
             [(sqrt) (unop st "sqrt.rn.f32 ~a, ~a;" 'f32 v)]
             [(inv-sqrt) (unop st "rsqrt.approx.ftz.f32 ~a, ~a;" 'f32 v)]
             [(tanh) (f-tanh st v)]
             [(sin cos) (f-trig st h v)]
             [(floor) (unop st "cvt.rmi.f32.f32 ~a, ~a;" 'f32 v)]
             [(ceil) (unop st "cvt.rpi.f32.f32 ~a, ~a;" 'f32 v)]))]
        [(eq? h 'abs)
         (let* ([t (resolve (ty-of (car args) env) expected)] [v (cx-as st (car args) env t e)])
           (case t
             [(f32) (unop st "abs.f32 ~a, ~a;" t (car v))]
             [(i32) (unop st "abs.s32 ~a, ~a;" t (car v))]
             [else v]))]
        [(eq? h 'fma)
         (let ([vs (map (lambda (a) (cx-as st a env 'f32 e)) args)] [d (new-val st 'f32)])
           (ins! st "fma.rn.f32 ~a, ~a, ~a, ~a;" (car d) (car (car vs)) (car (cadr vs)) (car (caddr vs)))
           d)]
        [(eq? h 'dp4a)
         ;; (dp4a a b acc): acc + the dot product of the four signed bytes of a and of b
         (let* ([a (cast st e (cx st (car args) env 'u32) 'u32)]
                [b (cast st e (cx st (cadr args) env 'u32) 'u32)]
                [c (cx-as st (caddr args) env 'i32 e)]
                [d (new-val st 'i32)])
           (ins! st "dp4a.s32.s32 ~a, ~a, ~a, ~a;" (car d) (car a) (car b) (car c))
           d)]
        [(eq? h 'shuffle-xor)
         ;; (shuffle-xor v mask): the value of v in lane (lane xor mask) of the warp
         (let* ([t (resolve (ty-of (car args) env) expected)] [v (cx-as st (car args) env t e)] [d (new-val st t)])
           (unless (and (integer? (cadr args)) (exact? (cadr args))) (fail e "the shuffle mask must be a literal"))
           (ins! st "shfl.sync.bfly.b32 ~a, ~a, ~a, 31, 0xffffffff;" (car d) (car v) (cadr args))
           d)]
        [(eq? h 'half->f32)
         (let* ([v (cx st (car args) env 'u32)]
                [hv (if (eq? (cdr v) 'u16) v (cast st e v 'u16))]
                [d (new-val st 'f32)])
           (ins! st "cvt.f32.f16 ~a, ~a;" (car d) (car hv))
           d)]
        [(eq? h 'f32->half)
         (let* ([v (cx-as st (car args) env 'f32 e)] [hv (new-val st 'u16)] [d (new-val st 'u32)])
           (ins! st "cvt.rn.f16.f32 ~a, ~a;" (car hv) (car v))
           (ins! st "cvt.u32.u16 ~a, ~a;" (car d) (car hv))
           d)]
        [(memq h '(sbits ubits))
         (let* ([t (if (eq? h 'sbits) 'i32 'u32)]
                [x (cast st e (cx st (car args) env t) t)]
                [off (cx-as st (cadr args) env 'u32 e)]
                [cnt (cx-as st (caddr args) env 'u32 e)]
                [d (new-val st t)])
           (ins! st "bfe.~a ~a, ~a, ~a, ~a;" (tsfx t) (car d) (car x) (car off) (car cnt))
           d)]
        [(memq h '(local-id group-id global-id num-groups))
         (let ([d (literal-dim e (car args))])
           (cons (case h
                   [(local-id) (special st (string-append "%tid." (dim-name d)))]
                   [(group-id) (special st (string-append "%ctaid." (dim-name d)))]
                   [(num-groups) (special st (string-append "%nctaid." (dim-name d)))]
                   [else (global-id st d)])
                 'u32))]
        [(eq? h 'lane) (cons (special st "%laneid") 'u32)]
        [(eq? h 'subgroup-size) (const! st 'u32 32)]
        [(eq? h 'subgroup-id)
         (cons (memo! st 'subgroup-id (lambda () (let ([r (reg! st 'r)]) (pre! st "shr.u32 ~a, ~a, 5;" r (special st "%tid.x")) r))) 'u32)]
        [(eq? h 'num-subgroups)
         (cons (memo! st 'num-subgroups (lambda () (let ([r (reg! st 'r)]) (pre! st "shr.u32 ~a, ~a, 5;" r (special st "%ntid.x")) r))) 'u32)]
        [(memq h '(subgroup-add subgroup-max subgroup-min))
         (let* ([t (resolve (ty-of (car args) env) expected)] [v (cx-as st (car args) env t e)])
           (when (memq t '(bool u16)) (fail e "~a on ~a" h t))
           (warp-reduce st h t v))]
        [(eq? h 'let)
         (let loop ([bs (cadr e)] [env env])
           (if (null? bs)
               (cx st (caddr e) env expected)
               (let* ([t (resolve (ty-of (cadar bs) env) #f)] [v (cx-as st (cadar bs) env t e)])
                 (loop (cdr bs) (cons (cons (caar bs) (vector 'val (car v) t)) env)))))]
        [else (fail e "unknown operator ~a" h)])))

  (define (literal-dim form d)
    (unless (and (integer? d) (<= 0 d 2)) (fail form "dimension must be 0, 1 or 2"))
    d)

  ;; ---- cooperative matrices (mma.sync) -------------------------------------------------------------

  (define (cm-lookup form env name)
    (let ([b (lookup form env name)])
      (unless (eq? (vector-ref b 0) 'cmat) (fail form "~a is not a cooperative matrix" name))
      b))

  ;; The lane's place in the m16n8k16 fragments: g = lane / 4 (row group), t2 = 2 * (lane % 4).
  (define (frag-g st)
    (memo! st 'frag-g (lambda () (let ([r (reg! st 'r)]) (pre! st "shr.u32 ~a, ~a, 2;" r (special st "%laneid")) r))))
  (define (frag-t2 st)
    (memo! st 'frag-t2
      (lambda ()
        (let ([a (reg! st 'r)] [r (reg! st 'r)])
          (pre! st "and.b32 ~a, ~a, 3;" a (special st "%laneid"))
          (pre! st "shl.b32 ~a, ~a, 1;" r a)
          r))))

  (define (add-const! st reg c)
    (if (= c 0)
        reg
        (let ([d (reg! st 'r)]) (ins! st "add.u32 ~a, ~a, ~a;" d reg c) d)))

  ;; Element offset (in elements) of matrix element (row, col) = (rr + rc, cr + cc) from the matrix
  ;; corner, for a row-major (row * stride + col) or column-major (col * stride + row) layout.
  (define (frag-offset st stride layout rr rc cr cc)
    (let ([row (add-const! st rr rc)] [col (add-const! st cr cc)] [d (reg! st 'r)])
      (if (eq? layout 'row)
          (ins! st "mad.lo.u32 ~a, ~a, ~a, ~a;" d row stride col)
          (ins! st "mad.lo.u32 ~a, ~a, ~a, ~a;" d col stride row))
      d))

  ;; Address of element `base + off` in the same space as `addr` (a shared 32-bit or global 64-bit address).
  (define (offset-addr st space addr off esz)
    (if (eq? space 'shared)
        (let ([a (reg! st 'r)]) (ins! st "mad.lo.u32 ~a, ~a, ~a, ~a;" a off esz addr) a)
        (let ([w (reg! st 'rd)] [a (reg! st 'rd)])
          (ins! st "mul.wide.u32 ~a, ~a, ~a;" w off esz)
          (ins! st "add.u64 ~a, ~a, ~a;" a addr w)
          a)))

  (define (layout-of form l)
    (case l [(row col) l] [else (fail form "layout must be row or col")]))

  ;; (cm-load M buf idx stride row|col): the lane's fragment of the 16x16 matrix whose corner is
  ;; element idx of buf (f16). Pairs of halves that are adjacent in memory are one 32-bit load, which
  ;; needs idx, stride and the element offsets even.
  (define (cm-load! st form env args)
    (let* ([b (cm-lookup form env (car args))] [regs (vector-ref b 1)] [use (vector-ref b 2)]
           [layout (layout-of form (car (cddddr args)))]
           [stride (car (cx-as st (cadddr args) env 'u32 form))])
      (let-values ([(addr elem access space) (element-addr st form env (cadr args) (caddr args))])
        (unless (eq? elem 'f16) (fail form "cm-load reads f16 data on the PTX target"))
        (let* ([g (frag-g st)] [t2 (frag-t2 st)]
               [space-s (if (eq? space 'shared) "shared" "global")]
               [pair-adjacent (if (eq? use 'a) (eq? layout 'row) (eq? layout 'col))]
               ;; (row-reg row-const col-reg col-const) of the first element of fragment register i
               [first-elem
                (lambda (i)
                  (if (eq? use 'a)
                      ;; a0: (g, t2) a1: (g + 8, t2) a2: (g, t2 + 8) a3: (g + 8, t2 + 8)
                      (list g (if (odd? i) 8 0) t2 (if (>= i 2) 8 0))
                      ;; b regs 2h + q: k = t2 + 8q (+ 0 / 1), n = g + 8h
                      (list t2 (* 8 (mod i 2)) g (* 8 (div i 2)))))]
               [step (lambda (e) (if (eq? use 'a) (list (car e) (cadr e) (caddr e) (+ (cadddr e) 1)) (list (car e) (+ (cadr e) 1) (caddr e) (cadddr e))))])
          (for-each
            (lambda (reg i)
              (let* ([e0 (first-elem i)] [e1 (step e0)]
                     [o0 (apply frag-offset st stride layout e0)])
                (if pair-adjacent
                    (ins! st "ld.~a.b32 ~a, [~a];" space-s reg (offset-addr st space addr o0 2))
                    (let ([o1 (apply frag-offset st stride layout e1)] [h0 (reg! st 'h)] [h1 (reg! st 'h)])
                      (ins! st "ld.~a.b16 ~a, [~a];" space-s h0 (offset-addr st space addr o0 2))
                      (ins! st "ld.~a.b16 ~a, [~a];" space-s h1 (offset-addr st space addr o1 2))
                      (ins! st "mov.b32 ~a, {~a, ~a};" reg h0 h1)))))
            regs (let loop ([i 0] [acc '()]) (if (= i 4) (reverse acc) (loop (+ i 1) (cons i acc)))))))))

  ;; (cm-store M buf idx stride row|col): the accumulator's eight values to f32 elements.
  (define (cm-store! st form env args)
    (let* ([b (cm-lookup form env (car args))] [regs (vector-ref b 1)]
           [layout (layout-of form (car (cddddr args)))]
           [stride (car (cx-as st (cadddr args) env 'u32 form))])
      (unless (eq? (vector-ref b 2) 'c) (fail form "cm-store stores accumulators"))
      (let-values ([(addr elem access space) (element-addr st form env (cadr args) (caddr args))])
        (unless (eq? elem 'f32) (fail form "cm-store writes f32 data on the PTX target"))
        (let ([g (frag-g st)] [t2 (frag-t2 st)] [space-s (if (eq? space 'shared) "shared" "global")])
          ;; c[4h + 2r + j]: row g + 8r, column t2 + j + 8h
          (let loop ([i 0] [regs regs])
            (unless (null? regs)
              (let* ([h (div i 4)] [r (div (mod i 4) 2)] [j (mod i 2)]
                     [off (frag-offset st stride layout g (* 8 r) t2 (+ j (* 8 h)))])
                (ins! st "st.~a.f32 [~a], ~a;" space-s (offset-addr st space addr off 4) (car regs)))
              (loop (+ i 1) (cdr regs))))))))

  ;; ---- statements --------------------------------------------------------------------------------

  (define (cstmts st stmts env)
    (let loop ([stmts stmts] [env env])
      (if (null? stmts) env (loop (cdr stmts) (cs st (car stmts) env)))))

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
           (let* ([init (cx-as st (caddr args) env t s)] [r (reg! st (cls t))])
             (mov! st t r (car init))
             (cons (cons name (vector 'var r t)) env)))]
        [(set!)
         (let ([b (lookup s env (car args))])
           (unless (eq? (vector-ref b 0) 'var) (fail s "~a is not a var" (car args)))
           (let ([v (cx-as st (cadr args) env (vector-ref b 2) s)])
             (mov! st (vector-ref b 2) (vector-ref b 1) (car v))
             env))]
        [(store)
         (let-values ([(addr elem access space) (element-addr st s env (car args) (cadr args))])
           (let ([v (cx-as st (caddr args) env elem s)])
             (ins! st "st.~a.~a [~a], ~a;" (if (eq? space 'shared) "shared" "global") (tsfx elem) addr (car v))
             env))]
        [(vload)
         ;; (vload (n0 n1 [n2 n3]) buf idx): one ld.v2 / ld.v4; idx * element size must be aligned to
         ;; the vector width (the caller, i.e. the lowering that picked this kernel, guarantees it).
         (let* ([names (car args)] [n (length names)]
                [b (buffer-like s env (cadr args))]
                [elem (vector-ref b 2)])
           (unless (and (memv n '(2 4)) (memq elem '(f32 u32 i32)) (eq? (vector-ref b 0) 'buf)) (fail s "vload needs 2 or 4 words from a global buffer"))
           (let-values ([(addr elem access space) (element-addr st s env (cadr args) (caddr args))])
             (let ([regs (map (lambda (nm) (new-val st elem)) names)])
               (ins! st "ld.global~a.v~a.~a {~a}, [~a];"
                     (if (eq? access 'ro) ".nc" "") n (tsfx elem) (join (map car regs) ", ") addr)
               (fold-left (lambda (env pr) (cons (cons (car pr) (vector 'val (car (cdr pr)) elem)) env))
                          env (map cons names regs)))))]
        [(begin) (cstmts st args env) env]
        [(when unless)
         (let ([c (cx-as st (car args) env 'bool s)] [end (label! st)])
           (ins! st (if (eq? h 'when) "@!~a bra ~a;" "@~a bra ~a;") (car c) end)
           (cstmts st (cdr args) env)
           (ins! st "~a:" end)
           env)]
        [(if)
         (let ([c (cx-as st (car args) env 'bool s)] [els (label! st)] [end (label! st)])
           (ins! st "@!~a bra ~a;" (car c) els)
           (cs st (cadr args) env)
           (ins! st "bra ~a;" end)
           (ins! st "~a:" els)
           (when (= (length args) 3) (cs st (caddr args) env))
           (ins! st "~a:" end)
           env)]
        [(for) (cfor st s env) env]
        [(barrier) (ins! st "bar.sync 0;") env]
        [(cmat)
         ;; (cmat name use type): a 16x16 matrix held as the fragments of mma.sync.m16n8k16 -- a, b: f16,
         ;; four 32-bit registers (two halves each); c: f32, eight registers (two n8 halves)
         (let* ([name (car args)] [use (cadr args)] [ty (caddr args)])
           (hashtable-set! (ps-memo st) 'mma #t)
           (unless (or (and (memq use '(a b)) (eq? ty 'f16)) (and (eq? use 'c) (eq? ty 'f32)))
             (fail s "the PTX target has a and b matrices of f16 and c matrices of f32"))
           (let ([regs (if (eq? use 'c)
                           (map (lambda (i) (reg! st 'f)) '(0 1 2 3 4 5 6 7))
                           (map (lambda (i) (reg! st 'r)) '(0 1 2 3)))])
             (cons (cons name (vector 'cmat regs use ty)) env)))]
        [(cm-zero)
         (for-each (lambda (r) (ins! st "mov.f32 ~a, 0f00000000;" r)) (vector-ref (cm-lookup s env (car args)) 1))
         env]
        [(cm-load)
         (cm-load! st s env args)
         env]
        [(cm-store)
         (cm-store! st s env args)
         env]
        [(cm-muladd)
         ;; (cm-muladd dst a b c): dst = a * b + c, two m16n8k16 products (the n halves)
         (let ([d (vector-ref (cm-lookup s env (car args)) 1)] [a (vector-ref (cm-lookup s env (cadr args)) 1)]
               [b (vector-ref (cm-lookup s env (caddr args)) 1)] [c (vector-ref (cm-lookup s env (cadddr args)) 1)])
           (for-each
             (lambda (h)
               (let ([part (lambda (l) (list (list-ref l (* 4 h)) (list-ref l (+ (* 4 h) 1)) (list-ref l (+ (* 4 h) 2)) (list-ref l (+ (* 4 h) 3))))])
                 (ins! st "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {~a}, {~a}, {~a, ~a}, {~a};"
                       (join (part d) ", ") (join a ", ") (list-ref b (* 2 h)) (list-ref b (+ (* 2 h) 1)) (join (part c) ", "))))
             '(0 1))
           env)]
        [else (fail s "unknown statement ~a" h)])))

  ;; Loops whose bounds are literal integers and that run few times are unrolled here, the
  ;; loop variable becoming a constant in every copy of the body (what the SPIR-V path gets from
  ;; the driver's compiler); the rest are counted loops.
  (define max-unroll 32)

  (define (cfor st s env)
    (let* ([spec (cadr s)] [lo (cadr spec)] [hi (caddr spec)]
           [step (if (pair? (cdddr spec)) (cadddr spec) 1)])
      (if (and (integer? lo) (exact? lo) (integer? hi) (exact? hi) (integer? step) (exact? step) (> step 0)
               (<= (div (+ (- hi lo) (- step 1)) step) max-unroll))
          (let loop ([v lo])
            (when (< v hi)
              (let ([c (const! st 'u32 v)])
                (cstmts st (cddr s) (cons (cons (car spec) (vector 'val (car c) 'u32)) env)))
              (loop (+ v step))))
          (cfor-loop st s env))))

  (define (cfor-loop st s env)
    (let* ([spec (cadr s)] [name (car spec)] [lo (cadr spec)] [hi (caddr spec)]
           [step (if (pair? (cdddr spec)) (cadddr spec) 1)]
           [t0 (common s (map (lambda (x) (ty-of x env)) (list lo hi step)))]
           [t (case t0 [(ilit) 'u32] [(f32 bool flit) (fail s "loop bounds must be integers")] [else t0])]
           [i (reg! st 'r)]
           [head (label! st)] [end (label! st)])
      (mov! st t i (car (cx-as st lo env t s)))
      (ins! st "~a:" head)
      (let ([c (new-val st 'bool)] [hv (cx-as st hi env t s)])
        (ins! st "~a ~a, ~a, ~a;" (cmp-op '< t s) (car c) i (car hv))
        (ins! st "@!~a bra ~a;" (car c) end))
      (cstmts st (cddr s) (cons (cons name (vector 'val i t)) env))
      (let ([sv (cx-as st step env t s)])
        (ins! st "add.u32 ~a, ~a, ~a;" i i (car sv)))
      (ins! st "bra ~a;" head)
      (ins! st "~a:" end)))

  ;; ---- kernels -----------------------------------------------------------------------------------

  (define (clause? f names) (and (pair? f) (memq (car f) names) #t))

  (define (join strings sep)
    (if (null? strings) "" (fold-left (lambda (acc s) (string-append acc sep s)) (car strings) (cdr strings))))

  (define (compile-kernel-ptx k)
    (unless (and (pair? k) (eq? (car k) 'kernel)) (error 'tree-ptx "expected (kernel name clause ... body ...)" k))
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
           [st (make-ps)]
           [buffer-specs (find 'buffers)]
           [push-specs (find 'push)]
           [shared-specs (find 'shared)]
           [buffers (let loop ([i 0] [bs buffer-specs] [acc '()])
                      (if (null? bs)
                          (reverse acc)
                          (loop (+ i 1) (cdr bs)
                                (cons (cons (caar bs) (vector 'buf (string-append "b" (number->string i)) (caddr (car bs)) (cadr (car bs)) i)) acc))))]
           [pushes (let loop ([i 0] [ps push-specs] [acc '()])
                     (if (null? ps)
                         (reverse acc)
                         (loop (+ i 1) (cdr ps) (cons (cons (caar ps) (vector 'push i (cadar ps))) acc))))]
           [shareds (map (lambda (s) (cons (car s) (vector 'shared (string-append "sh_" (symbol->string (car s))) (cadr s) (caddr s)))) shared-specs)]
           [env (append buffers pushes shareds)]
           [name (symbol->string (cadr k))]
           [push-bytes (* 4 (length push-specs))])
      (cstmts st body env)
      (let* ([params (append (map (lambda (i) (fmt "    .param .u64 b~a" i)) (let loop ([i 0] [acc '()]) (if (= i (length buffer-specs)) (reverse acc) (loop (+ i 1) (cons i acc)))))
                             (if (> push-bytes 0) (list (fmt "    .param .align 4 .b8 pc[~a]" push-bytes)) '()))]
             [regs (map (lambda (c)
                          (let ([n (hashtable-ref (ps-counts st) (car c) 0)])
                            (fmt "  .reg .~a %~a<~a>;" (cdr c) (car c) (max n 1))))
                        '((p . pred) (r . b32) (f . f32) (rd . b64) (h . b16)))]
             [decls (map (lambda (s) (fmt "  .shared .align 16 .b8 sh_~a[~a];" (car s) (* (esize (cadr s)) (caddr s)))) shared-specs)]
             [text (join (append
                           (list ".version 7.0" (if (hashtable-ref (ps-memo st) 'mma #f) ".target sm_80" ".target sm_70") ".address_size 64" ""
                                 (fmt ".visible .entry ~a(" name)
                                 (join params ",\n")
                                 ")"
                                 (fmt ".reqntid ~a, ~a, ~a" (car wg) (cadr wg) (caddr wg))
                                 "{")
                           regs decls
                           (reverse (ps-prelude st))
                           (map (lambda (l) (if (char=? (string-ref l (- (string-length l) 1)) #\:) l (string-append "  " l))) (reverse (ps-lines st)))
                           (list "  ret;" "}" ""))
                         "\n")])
        (list text name (map cadr buffer-specs) push-bytes wg))))
)
