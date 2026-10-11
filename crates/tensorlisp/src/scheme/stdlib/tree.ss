;; (tl tree) -- Tree, the target-agnostic kernel language, and its SPIR-V compiler.
;; docs/design/tree.md.
;;
;;   (compile-kernel
;;     '(kernel softmax
;;        (workgroup 256)
;;        (buffers (x ro f32) (y wo f32))      ; binding 0, 1; ro | wo | rw
;;        (push (ncols u32))                   ; push-constant words
;;        (shared (red f32 256))               ; workgroup memory
;;        (var m f32 -inf.0)
;;        (for (c (local-id 0) ncols 256) (set! m (max m (load x c))))
;;        ...))
;;   => a bytevector of SPIR-V (little-endian words) for a Vulkan compute shader.
;;
;; A kernel is data: Scheme builds it with quasiquote and ordinary functions (that is
;; the macro layer: unrolling, tiling and fusion are list manipulation at staging time).
;; Types are f32 i32 u32 bool; integer and float literals adapt to the type around them.
;;
;; Statements:  (let ([n e] ...) stmt ...)  (var n type init)  (set! n e)  (store b i e)
;;              (when c stmt ...)  (unless c stmt ...)  (if c stmt [stmt])  (begin stmt ...)
;;              (for (i lo hi [step]) stmt ...)  (barrier)
;; Expressions: literals, names, (+ - * / rem min max pow) (< <= > >= = !=) (and or not)
;;              (load b i) (select c a b) (let ([n e] ...) e) (f32 e) (i32 e) (u32 e)
;;              (exp log sqrt inv-sqrt tanh sin cos floor ceil abs) (fma a b c)
;;              (bit-and bit-or bit-xor shl shr)
;;              (local-id d) (group-id d) (global-id d) (num-groups d)
;;              (lane) (subgroup-id) (subgroup-size) (num-subgroups)
;;              (subgroup-add e) (subgroup-max e) (subgroup-min e)
(library (tl tree)
  ;; ty-of and friends are shared with the PTX emitter, (tl tree ptx).
  (export compile-kernel ty-of common resolve value-type? fail lookup arith-ops cmp-ops bit-ops float-unary builtin-ops)
  (import (rnrs) (tl tree spirv) (only (chezscheme) format))

  (define (fail form fmt . args)
    (error 'tree (string-append (apply format #f fmt args) "\n  in " (format #f "~s" form))))

  ;; ---- types -------------------------------------------------------------------------------

  (define (type-id m t)
    (intern! m (list 'type t)
      (lambda (id)
        (case t
          [(f32) (global! m 22 id 32)]
          [(i32) (global! m 21 id 32 1)]
          [(u32) (global! m 21 id 32 0)]
          [(bool) (global! m 20 id)]
          [(void) (global! m 19 id)]
          [(u16) (capability! m 22) (capability! m 4433) (extension! m "SPV_KHR_16bit_storage") (global! m 21 id 16 0)]
          [(f16) (capability! m 9) (capability! m 4433) (extension! m "SPV_KHR_16bit_storage") (global! m 22 id 16)]
          [(vec2f) (global! m 23 id (type-id m 'f32) 2)]
          [else (error 'tree "unknown type" t)]))))

  (define (ptr-type m storage elem)
    (intern! m (list 'ptr storage elem) (lambda (id) (global! m 32 id storage elem))))

  (define (const m t v)
    (let ([v (if (eq? t 'f32) (inexact v) v)])
      (intern! m (list 'const t v)
        (lambda (id)
          (let ([ty (type-id m t)])
            (case t
              [(bool) (global! m (if v 41 42) ty id)]
              [(f32) (global! m 43 ty id (f32-bits v))]
              [else (global! m 43 ty id (bitwise-and v #xffffffff))]))))))

  (define (value-type? t) (and (memq t '(f32 i32 u32 bool u16 f16)) #t))

  ;; ---- environment -------------------------------------------------------------------------
  ;; name -> #(val id type) | #(var ptr type) | #(push index type) | #(buf var elem access)
  ;;       | #(shared var elem length)

  (define (lookup form env name)
    (cond [(assq name env) => cdr]
          [else (fail form "unbound name ~a" name)]))

  (define (sym-type form env name)
    (let ([b (lookup form env name)])
      (case (vector-ref b 0)
        [(val var push) (vector-ref b (if (eq? (vector-ref b 0) 'push) 2 2))]
        [else (fail form "~a is a buffer, not a value" name)])))

  ;; The single type of `types` (literals adapt), or 'ilit / 'flit when all are literals.
  (define (common form types)
    (let ([concrete (filter value-type? types)])
      (cond
        [(pair? concrete)
         (for-each (lambda (t)
                     (unless (eq? t (car concrete))
                       (fail form "type mismatch: ~a vs ~a" (car concrete) t)))
                   concrete)
         (car concrete)]
        [(memq 'flit types) 'f32]
        [else 'ilit])))

  (define (resolve t expected)
    (case t
      [(ilit) (if (memq expected '(f32 i32 u32)) expected 'i32)]
      [(flit) 'f32]
      [else t]))

  (define (buffer-like form env name)
    (let ([b (lookup form env name)])
      (case (vector-ref b 0)
        [(buf shared) b]
        [else (fail form "~a is not a buffer or shared array" name)])))

  (define arith-ops '(+ - * / min max rem pow))
  (define cmp-ops '(< <= > >= = !=))
  (define bit-ops '(bit-and bit-or bit-xor shl shr))
  (define float-unary '(exp log sqrt inv-sqrt tanh sin cos floor ceil))
  (define builtin-ops '(local-id group-id global-id num-groups lane subgroup-id subgroup-size num-subgroups))

  ;; Type of an expression without emitting anything.
  (define (ty-of e env)
    (cond
      [(boolean? e) 'bool]
      [(and (number? e) (exact? e)) 'ilit]
      [(number? e) 'flit]
      [(symbol? e) (sym-type e env e)]
      [(pair? e)
       (let ([h (car e)] [args (cdr e)])
         (cond
           [(memq h arith-ops) (common e (map (lambda (a) (ty-of a env)) args))]
           [(memq h bit-ops) (common e (map (lambda (a) (ty-of a env)) args))]
           [(memq h cmp-ops) 'bool]
           [(memq h '(and or not)) 'bool]
           [(eq? h 'load)
            (let ([b (buffer-like e env (cadr e))]) (vector-ref b 2))]
           [(memq h '(f32 i32 u32 u16 f16)) h]
           [(eq? h 'select) (common e (map (lambda (a) (ty-of a env)) (cddr e)))]
           [(eq? h 'guard) (common e (map (lambda (a) (ty-of a env)) (cddr e)))]
           [(memq h float-unary) 'f32]
           [(eq? h 'abs) (resolve (ty-of (cadr e) env) #f)]
           [(eq? h 'fma) 'f32]
           [(eq? h 'half->f32) 'f32]
           [(eq? h 'dp4a) 'i32]
           [(eq? h 'shuffle-xor) (resolve (ty-of (cadr e) env) #f)]
           [(eq? h 'f32->half) 'u32]
           [(eq? h 'sbits) 'i32]
           [(eq? h 'ubits) 'u32]
           [(memq h builtin-ops) 'u32]
           [(memq h '(subgroup-add subgroup-max subgroup-min)) (resolve (ty-of (cadr e) env) #f)]
           [(eq? h 'let)
            (ty-of (caddr e)
                   (let loop ([bs (cadr e)] [env env])
                     (if (null? bs)
                         env
                         (loop (cdr bs)
                               (cons (cons (caar bs) (vector 'val 0 (resolve (ty-of (cadar bs) env) #f)))
                                     env)))))]
           [else (fail e "unknown operator ~a" h)]))]
      [else (fail e "cannot compile this form")]))

  ;; ---- builtins, push constants ------------------------------------------------------------

  (define (vec3-type m)
    (intern! m '(type vec3-u32) (lambda (id) (global! m 23 id (type-id m 'u32) 3))))

  (define (builtin-vec3 m which)
    (cached! m (list 'builtin which)
      (lambda ()
        (let* ([v3 (vec3-type m)] [pty (ptr-type m 1 v3)] [var (fresh! m)] [val (fresh! m)])
          (global! m 59 pty var 1)
          (annotate! m 71 var 11 which)
          (interface! m var)
          (prelude! m 61 v3 val var)
          val))))

  (define (builtin-component m which d)
    (cached! m (list 'builtin-component which d)
      (lambda ()
        (let ([id (fresh! m)])
          (prelude! m 81 (type-id m 'u32) id (builtin-vec3 m which) d)
          id))))

  (define (builtin-scalar m which)
    (cached! m (list 'builtin-scalar which)
      (lambda ()
        (capability! m 61)
        (let* ([u (type-id m 'u32)] [pty (ptr-type m 1 u)] [var (fresh! m)] [val (fresh! m)])
          (global! m 59 pty var 1)
          (annotate! m 71 var 11 which)
          (interface! m var)
          (prelude! m 61 u val var)
          val))))

  (define (push-value m idx ty)
    (cached! m (list 'push idx)
      (lambda ()
        (let* ([tid (type-id m ty)] [ptr (fresh! m)] [val (fresh! m)]
               [var (cached! m 'push-var (lambda () (error 'tree "no push block")))])
          (prelude! m 65 (ptr-type m 9 tid) ptr var (const m 'u32 idx))
          (prelude! m 61 tid val ptr)
          val))))

  ;; ---- expressions -------------------------------------------------------------------------

  (define (emit-op m opcode ty . operands)
    (let ([id (fresh! m)])
      (apply body! m opcode (type-id m ty) id operands)
      (cons id ty)))

  (define (emit-ext m ty inst . args)
    (let ([id (fresh! m)])
      (apply body! m 12 (type-id m ty) id (import-ext! m "GLSL.std.450") inst args)
      (cons id ty)))

  (define (literal-int-dim form d)
    (unless (and (integer? d) (<= 0 d 2)) (fail form "dimension must be 0, 1 or 2"))
    d)

  ;; Compiles `e` expecting type `t` exactly.
  (define (cx-as m e env t form)
    (let ([v (cx m e env t)])
      (unless (eq? (cdr v) t)
        (fail form "expected ~a, got ~a for ~s" t (cdr v) e))
      v))

  ;; Element offset of binding `k` (kernels declared `(offsets prefix)` receive the residual byte
  ;; offset of every binding as leading push words; elements are four bytes).
  (define (buffer-offset m k elem)
    (cached! m (list 'buf-offset k)
      (lambda ()
        (let ([raw (push-value m k 'u32)] [id (fresh! m)])
          (prelude! m 194 (type-id m 'u32) id raw (const m 'u32 (if (memq elem '(u16 f16)) 1 2)))
          id))))

  (define (element-ptr m form env name idx)
    (let* ([b (buffer-like form env name)]
           [i0 (cx m idx env 'u32)]
           [elem (vector-ref b 2)]
           [checked (begin (unless (memq (cdr i0) '(u32 i32)) (fail form "index must be an integer, got ~a" (cdr i0))) #t)]
           [i (if (and (eq? (vector-ref b 0) 'buf) (memo-ref m 'prefix))
                  (let ([u (cast m form i0 'u32)])
                    (emit-op m 128 'u32 (car u) (buffer-offset m (vector-ref b 4) elem)))
                  i0)])
      (let ([id (fresh! m)] [var (vector-ref b 1)])
        (if (eq? (vector-ref b 0) 'buf)
            (body! m 65 (ptr-type m 12 (type-id m elem)) id var (const m 'u32 0) (car i))
            (body! m 65 (ptr-type m 4 (type-id m elem)) id var (car i)))
        (cons id elem))))

  (define (arith-opcode form h t)
    (let ([f (eq? t 'f32)] [u (eq? t 'u32)])
      (case h
        [(+) (if f 129 128)]
        [(-) (if f 131 130)]
        [(*) (if f 133 132)]
        [(/) (cond [f 136] [u 134] [else 135])]
        [(rem) (cond [f 140] [u 137] [else 138])]
        [(bit-and) 199] [(bit-or) 197] [(bit-xor) 198] [(shl) 196]
        [(shr) (if u 194 195)]
        [else (fail form "no opcode for ~a" h)])))

  (define (cmp-opcode form h t)
    (let ([f (eq? t 'f32)] [u (eq? t 'u32)] [b (eq? t 'bool)])
      (case h
        [(<) (cond [f 184] [u 176] [else 177])]
        [(<=) (cond [f 188] [u 178] [else 179])]
        [(>) (cond [f 186] [u 172] [else 173])]
        [(>=) (cond [f 190] [u 174] [else 175])]
        [(=) (cond [f 180] [b 164] [else 170])]
        [(!=) (cond [f 182] [b 165] [else 171])]
        [else (fail form "no comparison ~a" h)])))

  (define (cast m form v t)
    (let ([s (cdr v)])
      (cond
        [(eq? s t) v]
        [(and (eq? s 'f32) (eq? t 'i32)) (emit-op m 110 t (car v))]
        [(and (eq? s 'f32) (eq? t 'u32)) (emit-op m 109 t (car v))]
        [(and (eq? s 'i32) (eq? t 'f32)) (emit-op m 111 t (car v))]
        [(and (eq? s 'u32) (eq? t 'f32)) (emit-op m 112 t (car v))]
        [(and (memq s '(i32 u32)) (memq t '(i32 u32))) (emit-op m 124 t (car v))]
        [(and (eq? s 'u16) (memq t '(u32 i32))) (let ([u (emit-op m 113 'u32 (car v))]) (if (eq? t 'u32) u (emit-op m 124 t (car u))))]
        [(and (eq? s 'u32) (eq? t 'u16)) (emit-op m 113 t (car v))]
        [(and (eq? s 'f32) (eq? t 'f16)) (emit-op m 115 t (car v))]
        [(and (eq? s 'f16) (eq? t 'f32)) (emit-op m 115 t (car v))]
        [(and (eq? s 'i32) (eq? t 'u16)) (emit-op m 113 t (car (emit-op m 124 'u32 (car v))))]
        [else (fail form "cannot convert ~a to ~a" s t)])))

  (define (cx m e env expected)
    (cond
      [(boolean? e) (cons (const m 'bool e) 'bool)]
      [(and (number? e) (exact? e))
       (let ([t (if (memq expected '(f32 i32 u32)) expected 'i32)])
         (cons (const m t e) t))]
      [(number? e) (cons (const m 'f32 e) 'f32)]
      [(symbol? e)
       (let ([b (lookup e env e)])
         (case (vector-ref b 0)
           [(val) (cons (vector-ref b 1) (vector-ref b 2))]
           [(var) (let ([id (fresh! m)])
                    (body! m 61 (type-id m (vector-ref b 2)) id (vector-ref b 1))
                    (cons id (vector-ref b 2)))]
           [(push) (cons (push-value m (vector-ref b 1) (vector-ref b 2)) (vector-ref b 2))]
           [else (fail e "~a is a buffer, not a value" e)]))]
      [(pair? e) (cx-form m e env expected)]
      [else (fail e "cannot compile this form")]))

  (define (cx-form m e env expected)
    (let ([h (car e)] [args (cdr e)])
      (cond
        ;; arithmetic, n-ary left folds; (- x) negates
        [(memq h (append arith-ops bit-ops))
         (when (null? args) (fail e "~a needs operands" h))
         (let* ([t (resolve (common e (map (lambda (a) (ty-of a env)) args)) expected)]
                [vals (map (lambda (a) (cx-as m a env t e)) args)])
           (when (eq? t 'bool) (fail e "~a on bool" h))
           (when (eq? t 'u16) (fail e "~a on u16: convert with (u32 ...) first" h))
           (when (eq? t 'f16) (fail e "~a on f16: f16 is for storage and matrices only" h))
           (when (and (memq h bit-ops) (eq? t 'f32)) (fail e "~a on f32" h))
           (cond
             [(and (eq? h '-) (null? (cdr vals)))
              (emit-op m (if (eq? t 'f32) 127 126) t (car (car vals)))]
             [(memq h '(min max))
              (fold-left (lambda (acc v)
                           (emit-ext m t
                                     (case t
                                       [(f32) (if (eq? h 'max) 40 37)]
                                       [(u32) (if (eq? h 'max) 41 38)]
                                       [else (if (eq? h 'max) 42 39)])
                                     (car acc) (car v)))
                         (car vals) (cdr vals))]
             [(eq? h 'pow)
              (unless (eq? t 'f32) (fail e "pow needs f32"))
              (fold-left (lambda (acc v) (emit-ext m t 26 (car acc) (car v))) (car vals) (cdr vals))]
             [else
              (let ([op (arith-opcode e h t)])
                (fold-left (lambda (acc v) (emit-op m op t (car acc) (car v))) (car vals) (cdr vals)))]))]
        [(memq h cmp-ops)
         (unless (= (length args) 2) (fail e "~a takes two operands" h))
         (let* ([t (resolve (common e (map (lambda (a) (ty-of a env)) args)) #f)]
                [a (cx-as m (car args) env t e)]
                [b (cx-as m (cadr args) env t e)])
           (emit-op m (cmp-opcode e h t) 'bool (car a) (car b)))]
        [(memq h '(and or))
         (let ([vals (map (lambda (a) (cx-as m a env 'bool e)) args)])
           (fold-left (lambda (acc v) (emit-op m (if (eq? h 'and) 167 166) 'bool (car acc) (car v)))
                      (car vals) (cdr vals)))]
        [(eq? h 'not)
         (emit-op m 168 'bool (car (cx-as m (car args) env 'bool e)))]
        [(eq? h 'load)
         (let* ([p (element-ptr m e env (car args) (cadr args))] [id (fresh! m)])
           (body! m 61 (type-id m (cdr p)) id (car p))
           (cons id (cdr p)))]
        [(memq h '(f32 i32 u32 u16 f16))
         (let* ([t0 (ty-of (car args) env)]
                [v (cx m (car args) env (if (memq t0 '(ilit flit)) (if (eq? t0 'ilit) h 'f32) #f))])
           (cast m e v h))]
        [(eq? h 'guard)
         ;; (guard c e d): e where c holds, else d; e is not evaluated (not loaded from) otherwise.
         (let* ([t (resolve (common e (map (lambda (a) (ty-of a env)) (cdr args))) expected)]
                [c (cx-as m (car args) env 'bool e)]
                [d (cx-as m (caddr args) env t e)]
                [var (local-variable m t)]
                [then (fresh! m)] [merge (fresh! m)])
           (body! m 62 var (car d))
           (body! m 247 merge 0)
           (body! m 250 (car c) then merge)
           (body! m 248 then)
           (let ([v (cx-as m (cadr args) env t e)])
             (body! m 62 var (car v))
             (body! m 249 merge))
           (body! m 248 merge)
           (let ([id (fresh! m)])
             (body! m 61 (type-id m t) id var)
             (cons id t)))]
        [(eq? h 'select)
         (let* ([t (resolve (common e (map (lambda (a) (ty-of a env)) (cdr args))) expected)]
                [c (cx-as m (car args) env 'bool e)]
                [a (cx-as m (cadr args) env t e)]
                [b (cx-as m (caddr args) env t e)])
           (emit-op m 169 t (car c) (car a) (car b)))]
        [(memq h float-unary)
         (let ([v (cx-as m (car args) env 'f32 e)])
           (emit-ext m 'f32
                     (case h [(exp) 27] [(log) 28] [(sqrt) 31] [(inv-sqrt) 32] [(tanh) 21]
                       [(sin) 13] [(cos) 14] [(floor) 8] [(ceil) 9])
                     (car v)))]
        [(eq? h 'abs)
         (let* ([t (resolve (ty-of (car args) env) expected)] [v (cx-as m (car args) env t e)])
           (case t
             [(f32) (emit-ext m t 4 (car v))]
             [(i32) (emit-ext m t 5 (car v))]
             [else v]))]
        [(eq? h 'dp4a)
         ;; (dp4a a b acc): acc + the dot product of the four signed bytes of a and of b
         ;; (SPV_KHR_integer_dot_product; PTX has dp4a)
         (capability! m 6019)
         (capability! m 6018)
         (extension! m "SPV_KHR_integer_dot_product")
         (let* ([a (cast m e (cx m (car args) env 'u32) 'u32)]
                [b (cast m e (cx m (cadr args) env 'u32) 'u32)]
                [c (cx-as m (caddr args) env 'i32 e)]
                [dot (emit-op m 4450 'i32 (car a) (car b) 0)])
           (emit-op m 128 'i32 (car dot) (car c)))]
        [(eq? h 'shuffle-xor)
         ;; (shuffle-xor v mask): the value of v in lane (lane xor mask) of the subgroup
         (capability! m 61)
         (capability! m 65)
         (let* ([t (resolve (ty-of (car args) env) expected)] [v (cx-as m (car args) env t e)])
           (unless (and (integer? (cadr args)) (exact? (cadr args))) (fail e "the shuffle mask must be a literal"))
           (emit-op m 346 t (const m 'u32 3) (car v) (const m 'u32 (cadr args))))]
        [(eq? h 'half->f32)
         (let* ([v (cx m (car args) env 'u32)]
                [u (cast m e v 'u32)]
                [pair (emit-ext m 'vec2f 62 (car u))]
                [id (fresh! m)])
           (body! m 81 (type-id m 'f32) id (car pair) 0)
           (cons id 'f32))]
        [(eq? h 'f32->half)
         (let* ([v (cx-as m (car args) env 'f32 e)]
                [vec (let ([id (fresh! m)])
                       (body! m 80 (type-id m 'vec2f) id (car v) (const m 'f32 0.0))
                       id)])
           (emit-ext m 'u32 58 vec))]
        [(memq h '(sbits ubits))
         (let* ([t (if (eq? h 'sbits) 'i32 'u32)]
                [x (cast m e (cx m (car args) env t) t)]
                [off (cx-as m (cadr args) env 'u32 e)]
                [cnt (cx-as m (caddr args) env 'u32 e)])
           (emit-op m (if (eq? h 'sbits) 202 203) t (car x) (car off) (car cnt)))]
        [(eq? h 'fma)
         (let ([vs (map (lambda (a) (cx-as m a env 'f32 e)) args)])
           (emit-ext m 'f32 50 (car (car vs)) (car (cadr vs)) (car (caddr vs))))]
        [(memq h '(local-id group-id global-id num-groups))
         (let ((d (literal-int-dim e (car args))))
           (cons (builtin-component m (case h [(local-id) 27] [(group-id) 26] [(global-id) 28] [else 24]) d)
                 'u32))]
        [(eq? h 'lane) (cons (builtin-scalar m 41) 'u32)]
        [(eq? h 'subgroup-id) (cons (builtin-scalar m 40) 'u32)]
        [(eq? h 'subgroup-size) (cons (builtin-scalar m 36) 'u32)]
        [(eq? h 'num-subgroups) (cons (builtin-scalar m 38) 'u32)]
        [(memq h '(subgroup-add subgroup-max subgroup-min))
         (let* ([t (resolve (ty-of (car args) env) expected)] [v (cx-as m (car args) env t e)])
           (when (eq? t 'bool) (fail e "~a on bool" h))
           (capability! m 61)
           (capability! m 63)
           (emit-op m
                    (case h
                      [(subgroup-add) (if (eq? t 'f32) 350 349)]
                      [(subgroup-max) (case t [(f32) 358] [(u32) 357] [else 356])]
                      [else (case t [(f32) 355] [(u32) 354] [else 353])])
                    t (const m 'u32 3) 0 (car v)))]
        [(eq? h 'let)
         (let loop ([bs (cadr e)] [env env])
           (if (null? bs)
               (cx m (caddr e) env expected)
               (let* ([t (resolve (ty-of (cadar bs) env) #f)] [v (cx-as m (cadar bs) env t e)])
                 (loop (cdr bs) (cons (cons (caar bs) (vector 'val (car v) t)) env)))))]
        [else (fail e "unknown operator ~a" h)])))

  ;; ---- statements --------------------------------------------------------------------------

  (define (label! m id) (body! m 248 id))

  ;; Cooperative matrices (VK_KHR_cooperative_matrix): 16x16, subgroup scope. Uses: a, b (f16 or f32
  ;; inputs) and c (accumulator).
  (define (cm-type m form use ty)
    (unless (memq use '(a b c)) (fail form "matrix use must be a, b or c"))
    (unless (memq ty '(f16 f32)) (fail form "matrix element type must be f16 or f32"))
    (capability! m 6022)
    (capability! m 5345)
    (extension! m "SPV_KHR_vulkan_memory_model")
    (extension! m "SPV_KHR_cooperative_matrix")
    (intern! m (list 'cm use ty)
      (lambda (id)
        (global! m 4456 id (type-id m ty) (const m 'u32 3) (const m 'u32 16) (const m 'u32 16)
                 (const m 'u32 (case use [(a) 0] [(b) 1] [else 2]))))))

  (define (cm-lookup form env name)
    (let ([b (lookup form env name)])
      (unless (eq? (vector-ref b 0) 'cmat) (fail form "~a is not a cooperative matrix" name))
      b))

  (define (local-variable m ty)
    (let ([id (fresh! m)])
      (local-var! m 59 (ptr-type m 7 (type-id m ty)) id 7)
      id))

  (define (cstmts m stmts env)
    (let loop ([stmts stmts] [env env])
      (if (null? stmts)
          env
          (loop (cdr stmts) (cs m (car stmts) env)))))

  ;; Compiles one statement; returns the environment for the statements after it.
  (define (cs m s env)
    (unless (pair? s) (fail s "not a statement"))
    (let ([h (car s)] [args (cdr s)])
      (case h
        [(let)
         (let loop ([bs (car args)] [inner env])
           (if (null? bs)
               (cstmts m (cdr args) inner)
               (let* ([t (resolve (ty-of (cadar bs) inner) #f)] [v (cx-as m (cadar bs) inner t s)])
                 (loop (cdr bs) (cons (cons (caar bs) (vector 'val (car v) t)) inner)))))
         env]
        [(var)
         (let* ([name (car args)] [t (cadr args)] [init (cx-as m (caddr args) env t s)]
                [ptr (local-variable m t)])
           (unless (value-type? t) (fail s "unknown type ~a" t))
           (body! m 62 ptr (car init))
           (cons (cons name (vector 'var ptr t)) env))]
        [(set!)
         (let ([b (lookup s env (car args))])
           (unless (eq? (vector-ref b 0) 'var) (fail s "~a is not a var" (car args)))
           (let ([v (cx-as m (cadr args) env (vector-ref b 2) s)])
             (body! m 62 (vector-ref b 1) (car v))
             env))]
        [(store)
         (let* ([p (element-ptr m s env (car args) (cadr args))]
                [v (cx-as m (caddr args) env (cdr p) s)])
           (body! m 62 (car p) (car v))
           env)]
        [(vload)
         ;; (vload (n0 n1 ...) buf idx): binds ni to buf[idx + i]. One vector load on targets that
         ;; have them (the caller guarantees the alignment), scalar loads here.
         (let loop ([names (car args)] [i 0] [env env])
           (if (null? names)
               env
               (let ([v (cx m `(load ,(cadr args) (+ ,(caddr args) ,i)) env #f)])
                 (loop (cdr names) (+ i 1) (cons (cons (car names) (vector 'val (car v) (cdr v))) env)))))]
        [(begin) (cstmts m args env) env]
        [(when unless)
         (let* ([c (cx-as m (car args) env 'bool s)]
                [then (fresh! m)] [merge (fresh! m)])
           (body! m 247 merge 0)
           (if (eq? h 'when)
               (body! m 250 (car c) then merge)
               (body! m 250 (car c) merge then))
           (label! m then)
           (cstmts m (cdr args) env)
           (body! m 249 merge)
           (label! m merge)
           env)]
        [(if)
         (let* ([c (cx-as m (car args) env 'bool s)]
                [then (fresh! m)] [merge (fresh! m)]
                [els (and (= (length args) 3) (fresh! m))])
           (body! m 247 merge 0)
           (body! m 250 (car c) then (or els merge))
           (label! m then)
           (cs m (cadr args) env)
           (body! m 249 merge)
           (when els
             (label! m els)
             (cs m (caddr args) env)
             (body! m 249 merge))
           (label! m merge)
           env)]
        [(for) (cfor m s env) env]
        [(cmat)
         ;; (cmat name use type): a 16x16 cooperative matrix variable; use a | b | c, type f16 | f32
         (let* ([name (car args)] [use (cadr args)] [ty (caddr args)]
                [tid (cm-type m s use ty)] [ptr (fresh! m)])
           (local-var! m 59 (ptr-type m 7 tid) ptr 7)
           (cons (cons name (vector 'cmat ptr use ty tid)) env))]
        [(cm-zero)
         (let ([b (cm-lookup s env (car args))])
           (body! m 62 (vector-ref b 1) (intern! m (list 'cm-null (vector-ref b 4)) (lambda (id) (global! m 46 (vector-ref b 4) id))))
           env)]
        [(cm-load cm-store)
         ;; (cm-load M buf idx stride row|col), (cm-store M buf idx stride row|col): element idx of
         ;; buf (a buffer or shared array) is the matrix's corner; stride in elements
         (let* ([b (cm-lookup s env (car args))]
                [p (element-ptr m s env (cadr args) (caddr args))]
                [stride (cx-as m (cadddr args) env 'u32 s)]
                [layout (case (car (cddddr args)) [(row) 0] [(col) 1] [else (fail s "layout must be row or col")])]
                [lay (const m 'u32 layout)])
           (cond
             [(eq? h 'cm-load)
              (let ([id (fresh! m)])
                (body! m 4457 (vector-ref b 4) id (car p) lay (car stride))
                (body! m 62 (vector-ref b 1) id))
              env]
             [else
              (let ([id (fresh! m)])
                (body! m 61 (vector-ref b 4) id (vector-ref b 1))
                (body! m 4458 (car p) id lay (car stride)))
              env]))]
        [(cm-convert)
         ;; (cm-convert dst src): element type conversion (f32 -> f16)
         (let ([d (cm-lookup s env (car args))] [src (cm-lookup s env (cadr args))] [v (fresh! m)] [r (fresh! m)])
           (body! m 61 (vector-ref src 4) v (vector-ref src 1))
           (body! m 115 (vector-ref d 4) r v)
           (body! m 62 (vector-ref d 1) r)
           env)]
        [(cm-muladd)
         ;; (cm-muladd dst a b c): dst = a * b + c
         (let* ([d (cm-lookup s env (car args))] [a (cm-lookup s env (cadr args))]
                [bb (cm-lookup s env (caddr args))] [c (cm-lookup s env (cadddr args))]
                [la (fresh! m)] [lb (fresh! m)] [lc (fresh! m)] [r (fresh! m)])
           (body! m 61 (vector-ref a 4) la (vector-ref a 1))
           (body! m 61 (vector-ref bb 4) lb (vector-ref bb 1))
           (body! m 61 (vector-ref c 4) lc (vector-ref c 1))
           (body! m 4459 (vector-ref d 4) r la lb lc)
           (body! m 62 (vector-ref d 1) r)
           env)]
        [(barrier)
         (body! m 224 (const m 'u32 2) (const m 'u32 2) (const m 'u32 #x108))
         env]
        [else (fail s "unknown statement ~a" h)])))

  (define (cfor m s env)
    (let* ([spec (cadr s)] [name (car spec)] [lo (cadr spec)] [hi (caddr spec)]
           [step (if (pair? (cdddr spec)) (cadddr spec) 1)]
           [t0 (common s (map (lambda (x) (ty-of x env)) (list lo hi step)))]
           [t (case t0 [(ilit) 'u32] [(f32 bool flit) (fail s "loop bounds must be integers")] [else t0])]
           [ivar (local-variable m t)]
           [header (fresh! m)] [cond-l (fresh! m)] [body-l (fresh! m)] [cont (fresh! m)] [merge (fresh! m)])
      (body! m 62 ivar (car (cx-as m lo env t s)))
      (body! m 249 header)
      (label! m header)
      (body! m 246 merge cont 0)
      (body! m 249 cond-l)
      (label! m cond-l)
      (let* ([iv (fresh! m)])
        (body! m 61 (type-id m t) iv ivar)
        (let ([c (emit-op m (if (eq? t 'u32) 176 177) 'bool iv (car (cx-as m hi env t s)))])
          (body! m 250 (car c) body-l merge)))
      (label! m body-l)
      (let ([iv (fresh! m)])
        (body! m 61 (type-id m t) iv ivar)
        (cstmts m (cddr s) (cons (cons name (vector 'val iv t)) env)))
      (body! m 249 cont)
      (label! m cont)
      (let ([iv (fresh! m)])
        (body! m 61 (type-id m t) iv ivar)
        (let ([next (emit-op m 128 t iv (car (cx-as m step env t s)))])
          (body! m 62 ivar (car next))))
      (body! m 249 header)
      (label! m merge)))

  ;; ---- kernels -----------------------------------------------------------------------------

  (define (clause? f names) (and (pair? f) (memq (car f) names) #t))

  (define (declare-buffer m i spec)
    (let* ([name (car spec)] [access (cadr spec)] [elem (caddr spec)]
           [eid (type-id m elem)]
           [rt (intern! m (list 'rtarray eid)
                 (lambda (id) (global! m 29 id eid) (annotate! m 71 id 6 (if (memq elem '(u16 f16)) 2 4))))]
           [st (intern! m (list 'bufstruct eid access)
                 (lambda (id)
                   (global! m 30 id rt)
                   (annotate! m 71 id 2)
                   (annotate! m 72 id 0 35 0)
                   (case access
                     [(ro) (annotate! m 72 id 0 24)]
                     [(wo) (annotate! m 72 id 0 25)]
                     [(rw) #f]
                     [else (error 'tree "buffer access must be ro, wo or rw" access)])))]
           [var (fresh! m)])
      (global! m 59 (ptr-type m 12 st) var 12)
      (annotate! m 71 var 34 0)
      (annotate! m 71 var 33 i)
      (cons name (vector 'buf var elem access i))))

  (define (declare-push m specs)
    (let* ([tys (map (lambda (s) (type-id m (cadr s))) specs)]
           [st (fresh! m)] [var (fresh! m)])
      (apply global! m 30 st tys)
      (annotate! m 71 st 2)
      (let loop ([i 0] [specs specs])
        (unless (null? specs)
          (annotate! m 72 st i 35 (* 4 i))
          (loop (+ i 1) (cdr specs))))
      (global! m 59 (ptr-type m 9 st) var 9)
      (cached! m 'push-var (lambda () var))
      (let loop ([i 0] [specs specs] [acc '()])
        (if (null? specs)
            (reverse acc)
            (loop (+ i 1) (cdr specs)
                  (cons (cons (caar specs) (vector 'push i (cadar specs))) acc))))))

  (define (declare-shared m spec)
    (let* ([name (car spec)] [elem (cadr spec)] [len (caddr spec)]
           [eid (type-id m elem)]
           [arr (intern! m (list 'array eid len) (lambda (id) (global! m 28 id eid (const m 'u32 len))))]
           [var (fresh! m)])
      (global! m 59 (ptr-type m 4 arr) var 4)
      (cons name (vector 'shared var elem len))))

  (define (compile-kernel k)
    (unless (and (pair? k) (eq? (car k) 'kernel)) (error 'tree "expected (kernel name clause ... body ...)" k))
    (let* ([forms (cddr k)]
           [clauses (filter (lambda (f) (clause? f '(workgroup buffers push shared offsets))) forms)]
           [body (filter (lambda (f) (not (clause? f '(workgroup buffers push shared offsets)))) forms)]
           [find (lambda (name) (let ([c (filter (lambda (f) (eq? (car f) name)) clauses)])
                                  (if (null? c) '() (apply append (map cdr c)))))]
           [wg0 (find 'workgroup)]
           [wg (list (if (pair? wg0) (car wg0) 1)
                     (if (and (pair? wg0) (pair? (cdr wg0))) (cadr wg0) 1)
                     (if (and (pair? wg0) (pair? (cdr wg0)) (pair? (cddr wg0))) (caddr wg0) 1))]
           [m (new-module)]
           [void-t (type-id m 'void)]
           [fn-t (intern! m '(type fn) (lambda (id) (global! m 33 id void-t)))]
           [buffers (let loop ([i 0] [bs (find 'buffers)] [acc '()])
                      (if (null? bs) (reverse acc) (loop (+ i 1) (cdr bs) (cons (declare-buffer m i (car bs)) acc))))]
           [prefix? (and (memq 'prefix (find 'offsets)) #t)]
           [prefix-note (when prefix? (cached! m 'prefix (lambda () #t)))]
           [push-specs (append (if prefix?
                                   (map (lambda (i) (list (string->symbol (string-append "$off" (number->string i))) 'u32))
                                        (let loop ([i 0] [acc '()]) (if (= i (length (find 'buffers))) (reverse acc) (loop (+ i 1) (cons i acc)))))
                                   '())
                               (find 'push))]
           [pushes (if (null? push-specs) '() (declare-push m push-specs))]
           [shareds (map (lambda (s) (declare-shared m s)) (find 'shared))]
           [env (append buffers pushes shareds)]
           [fn (fresh! m)]
           [entry (fresh! m)])
      (cstmts m body env)
      (module-bytes m "main" fn void-t fn-t entry wg)))
)
