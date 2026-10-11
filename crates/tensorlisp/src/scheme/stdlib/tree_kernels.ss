;; (tl tree kernels) -- Tree kernels for ggml-style graph nodes: strided 4-D elementwise maps,
;; copies, repeat/concat/gather, layernorm, masked softmax, row sums and a batched tiled
;; matmul. Every kernel is `(offsets prefix)`, so any view offset works, and every kernel is
;; generated from a Scheme procedure whose arguments are the specialization (operation,
;; tile shape, ...). The Rust lowering (crates/kernels/src/vulkan/lower/tree.rs) asks for one
;; by expression, e.g. "(k:binary 'add)", and packs the push constants in the order each
;; kernel declares them below. Strides are in elements, extents in elements, innermost first.
(library (tl tree kernels)
  (export k:ew k:ew-q8 k:rowchain k:binary k:unary k:concat k:arange k:get-rows k:norm k:rms-norm k:norm-rope k:soft-max k:sum-rows
          k:matmul k:matmul-cm k:matmul-q8mma k:matvec k:rope k:set-rows k:argmax k:argmax-part k:argmax-final k:flash-attn k:flash-attn-part k:flash-attn-combine k:paged-attn-part k:paged-cache-write k:quantize-q8 k:matvec-q8 k:gateup-q8 k:repack-q8 k:matvec-q8r)
  (import (rnrs))

  (define wg 256)

  (define (name . parts)
    (string->symbol
      (apply string-append
             (map (lambda (p) (cond [(number? p) (number->string p)] [(symbol? p) (symbol->string p)] [else p]))
                  parts))))

  (define (range a b) (if (>= a b) '() (cons a (range (+ a 1) b))))
  (define (u32s . names) (map (lambda (n) (list n 'u32)) names))
  (define (strides prefix) (map (lambda (k) (name prefix k)) (range 0 4)))

  ;; Linear element index of this thread over a possibly 2-D grid (the host splits long grids).
  (define flat-index `(+ (global-id 0) (* (global-id 1) (* (num-groups 0) ,wg))))
  (define row-index '(+ (group-id 0) (* (group-id 1) (num-groups 0))))

  ;; i0..i3 from `idx` over extents ne0 ne1 ne2.
  (define dec-dst
    '([i0 (rem idx ne0)] [r1 (/ idx ne0)] [i1 (rem r1 ne1)] [r2 (/ r1 ne1)] [i2 (rem r2 ne2)] [i3 (/ r2 ne2)]))
  ;; j0..j3 from `idx` over the source extents sne0 sne1 sne2.
  (define dec-src
    '([j0 (rem idx sne0)] [q1 (/ idx sne0)] [j1 (rem q1 sne1)] [q2 (/ q1 sne1)] [j2 (rem q2 sne2)] [j3 (/ q2 sne2)]))
  ;; i1 i2 i3 of a row index over ne1 ne2.
  (define dec-row '([i1 (rem row ne1)] [rr (/ row ne1)] [i2 (rem rr ne2)] [i3 (/ rr ne2)]))

  (define (dot idxs strides) `(+ ,@(map (lambda (i s) `(* ,i ,s)) idxs strides)))
  (define dst-idx '(i0 i1 i2 i3))
  (define src-idx '(j0 j1 j2 j3))
  (define (mods idxs exts) (map (lambda (i e) `(rem ,i ,e)) idxs exts))

  ;; Buffer element types: f16 data is a u16 array (converted with half->f32 / f32->half), q8_0
  ;; data a u16 array of 17-word blocks (an f16 scale, 32 int8 quants).
  (define (elem-ty ty) (case ty [(f32) 'f32] [(f16 q8_0) 'u16] [(i32) 'i32] [else (error 'tree-kernels "no such element type" ty)]))
  (define (load-f32 ty buf idx)
    (case ty
      [(f32) `(load ,buf ,idx)]
      [(f16) `(half->f32 (load ,buf ,idx))]
      [(i32) `(f32 (load ,buf ,idx))]
      [else (error 'tree-kernels "cannot load as f32" ty)]))
  (define (store-f32 ty buf idx val)
    (case ty
      [(f32) `(store ,buf ,idx ,val)]
      [(f16) `(store ,buf ,idx (u16 (f32->half ,val)))]
      [(i32) `(store ,buf ,idx (i32 ,val))]
      [else (error 'tree-kernels "cannot store from f32" ty)]))

  ;; f32 expression for element k of the row of A that starts at `rb` (an element index; for q8_0 a
  ;; block index), A being f32, f16 or q8_0 data in buffer `a`.
  (define (a-elem atype rb k)
    (case atype
      [(f32 f16) (load-f32 atype 'a `(+ ,rb ,k))]
      [(q8_0)
       `(let ([qb (* (+ ,rb (/ ,k 32)) 17)] [qj (rem ,k 32)])
          (* (half->f32 (load a qb))
             (f32 (sbits (u32 (load a (+ qb 1 (/ qj 2)))) (* (rem qj 2) 8) 8))))]
      [else (error 'tree-kernels "no such weight type" atype)]))

  ;; ---- elementwise ------------------------------------------------------------------------

  ;; dst = a OP b with b broadcast over a.
  ;; push: n ne0 ne1 ne2 a0 a1 a2 a3 bn0 bn1 bn2 bn3 b0 b1 b2 b3 d0 d1 d2 d3
  (define (k:binary op)
    (let ([f (case op [(add) '+] [(sub) '-] [(mul) '*] [(div) '/] [else (error 'k:binary "unknown op" op)])])
      `(kernel ,(name 'binary_ op)
         (offsets prefix) (workgroup ,wg)
         (buffers (a ro f32) (b ro f32) (d wo f32))
         (push ,@(apply u32s (append '(n ne0 ne1 ne2) (strides 'a) (strides 'bn) (strides 'b) (strides 'd))))
         (let ([idx ,flat-index])
           (when (< idx n)
             (let ,dec-dst
               (store d ,(dot dst-idx (strides 'd))
                      (,f (load a ,(dot dst-idx (strides 'a)))
                          (load b ,(dot (mods dst-idx (strides 'bn)) (strides 'b)))))))))))

  (define (unary-expr fn x)
    (case fn
      [(copy) x]
      [(scale) `(+ (* ,x p0) p1)]
      [(sqr) `(* ,x ,x)]
      [(sqrt) `(sqrt ,x)]
      [(sin) `(sin ,x)]
      [(cos) `(cos ,x)]
      [(log) `(log ,x)]
      [(abs) `(abs ,x)]
      [(neg) `(- ,x)]
      [(exp) `(exp ,x)]
      [(step) `(select (> ,x 0.0) 1.0 0.0)]
      [(sgn) `(select (> ,x 0.0) 1.0 (select (< ,x 0.0) -1.0 0.0))]
      [(floor) `(floor ,x)]
      [(ceil) `(ceil ,x)]
      [(tanh) `(tanh ,x)]
      [(relu) `(max ,x 0.0)]
      [(sigmoid) `(/ 1.0 (+ 1.0 (exp (- ,x))))]
      [(silu) `(/ ,x (+ 1.0 (exp (- ,x))))]
      [(gelu) `(* 0.5 ,x (+ 1.0 (tanh (* 0.79788456080286535588 (+ ,x (* 0.044715 ,x ,x ,x))))))]
      [(gelu_quick) `(* ,x (/ 1.0 (+ 1.0 (exp (* -1.702 ,x)))))]
      [else (error 'k:unary "unknown function" fn)]))

  ;; dst = fn(src), element i of dst <- element i of src in linear order (mode `reshape`, which is
  ;; how a copy between differently shaped tensors works), at the same indices (`same`), or with
  ;; the source index taken modulo its extents (`repeat`).
  ;; push: n ne0 ne1 ne2 sne0 sne1 sne2 sne3 a0 a1 a2 a3 d0 d1 d2 d3 p0:f32 p1:f32
  ;; dst = fn(src) with element types sty / dty (f32, f16, i32).
  (define k:unary
    (case-lambda
      [(fn mode) (k:unary fn mode 'f32 'f32)]
      [(fn mode sty dty)
       `(kernel ,(name 'unary_ fn '_ mode '_ sty '_ dty)
          (offsets prefix) (workgroup ,wg)
          (buffers (a ro ,(elem-ty sty)) (d wo ,(elem-ty dty)))
          (push ,@(apply u32s (append '(n ne0 ne1 ne2 sne0 sne1 sne2 sne3) (strides 'a) (strides 'd))) (p0 f32) (p1 f32))
          (let ([idx ,flat-index])
            (when (< idx n)
              (let (,@dec-dst ,@(if (eq? mode 'reshape) dec-src '()))
                (let ([x ,(load-f32 sty 'a (case mode
                                              [(same) (dot dst-idx (strides 'a))]
                                              [(reshape) (dot src-idx (strides 'a))]
                                              [(repeat) (dot (mods dst-idx '(sne0 sne1 sne2 sne3)) (strides 'a))]
                                              [else (error 'k:unary "unknown mode" mode)]))])
                  ,(store-f32 dty 'd (dot dst-idx (strides 'd)) (unary-expr fn 'x)))))))]))

;; A fused elementwise region (vulkan/fuse_tree.rs): dst[e] = result, where `lets` are the
  ;; region's nodes in order, (tN expr), over the inputs x0, x1, ... and the element index idx.
  ;; Inputs are (type kind): `full` reads element e, `row` reads e mod rnI, `scalar` element 0.
  ;; (unary f x) is the unary function f of x (see unary-expr); c0, c1, ... are f32 constants.
  ;; push: n [rnI for each row input] [c0 ...]
  (define (ew-expand e)
    (cond [(not (pair? e)) e]
          [(eq? (car e) 'unary) (unary-expr (cadr e) (ew-expand (caddr e)))]
          [else (map ew-expand e)]))

  (define (ew-parts inputs lets nconst)
    (let ([idxs (range 0 (length inputs))])
      (define (in-name i) (name 'i i))
      (define (rn-name i) (name 'rn i))
      (define (load-in i spec)
        (load-f32 (car spec) (in-name i)
                  (case (cadr spec) [(full) 'idx] [(row) `(rem idx ,(rn-name i))] [(scalar) 0] [else (error 'k:ew "unknown input kind" spec)])))
      (values
        (map (lambda (i s) `(,(in-name i) ro ,(elem-ty (car s)))) idxs inputs)
        (append (apply append (map (lambda (i s) (if (eq? (cadr s) 'row) (list (list (rn-name i) 'u32)) '())) idxs inputs))
                (map (lambda (j) (list (name 'c j) 'f32)) (range 0 nconst)))
        (append (map (lambda (i s) `(,(name 'x i) ,(load-in i s))) idxs inputs)
                (map (lambda (b) (list (car b) (ew-expand (cadr b)))) lets)))))

  (define (k:ew inputs out-ty lets result nconst)
    (let-values ([(bufs pushes binds) (ew-parts inputs lets nconst)])
      `(kernel ew
         (offsets prefix) (workgroup ,wg)
         (buffers ,@bufs (d wo ,(elem-ty out-ty)))
         (push (n u32) ,@pushes)
         (let ([idx ,flat-index])
           (when (< idx n)
             (let ,binds
               ,(store-f32 out-ty 'd 'idx result)))))))

  ;; The same, but the f32 result is quantized for the dp4a mat-vecs that read it (n % 32 == 0):
  ;; a thread takes four consecutive elements; buffers xq, xd instead of d.
  (define (k:ew-q8 inputs out-ty lets result nconst)
    (let-values ([(bufs pushes binds) (ew-parts inputs lets nconst)])
      `(kernel ew_q8
         (offsets prefix) (workgroup ,wg)
         (buffers ,@bufs (xq wo u32) (xd wo f32))
         (push (n u32) ,@pushes)
         (let ([g0 ,flat-index])
           (let ([g (min g0 (- (/ n 4) 1))])
             (let (,@(map (lambda (j) `(,(name 'v j) (let ([idx (+ (* g 4) ,j)] ,@binds) ,result))) '(0 1 2 3)))
               ,@(q8-stmts '(v0 v1 v2 v3) 'g '(/ g 8) 'g '(< g0 (/ n 4)))))))))

  ;; A chain of row operations in one launch (vulkan/fuse_tree.rs `row_chains`): one workgroup per
  ;; row of n elements, the row held in registers (a thread owns groups of four consecutive
  ;; elements), a list of stages applied to it in turn, and any stage's result written out.
  ;;   (norm w eps)      x <- x * inv-sqrt(mean(x^2) + eps) [* w]    w: input index or #f, eps: constant index
  ;;   (ew lets result)  x <- result, where lets are (tN expr) over x (the current row), the other
  ;;                     inputs x1 x2 ... and constants c0 c1 ... (as in k:ew)
  ;; Input 0 is the row to start from; the others are `full` (same shape), `row` (index mod rnI) or `scalar`.
  ;; outs: for each output buffer, (stage-index kind): kind f32 (buffer oK) or q8 (xq, xd; at most one).
  ;; push: nrows [rnI for each row input] [c0 ...]
  (define (k:rowchain n inputs stages outs nconst)
    (let* ([ng (div n 4)] [slots (range 0 (div (+ ng (- wg 1)) wg))] [js (range 0 4)]
           [idxs (range 0 (length inputs))] [stage-ids (range 0 (length stages))])
      (define (in-name i) (name 'i i))
      (define (rn-name i) (name 'rn i))
      (define (xv s j) (name 'xv s '_ j))
      (define (g s) (name 'g s))
      (define (valid s) (name 'valid s))
      (define (e s) (name 'e s))
      (define (load-in i spec s j)
        (load-f32 (car spec) (in-name i)
                  (case (cadr spec)
                    [(full) `(+ rb ,(e s) ,j)]
                    [(row) `(rem (+ ,(e s) ,j) ,(rn-name i))]
                    [(scalar) 0]
                    [else (error 'k:rowchain "unknown input kind" spec)])))
      ;; statements applying stage st (index k) to every slot / element
      (define (stage-stmts st k)
        (case (car st)
          [(norm)
           (let ([w (cadr st)] [eps (name 'c (caddr st))] [sq (name 'sq k)] [tot (name 'tot k)] [inv (name 'inv k)])
             `((var ,sq f32 0.0)
               ,@(apply append (map (lambda (s) (map (lambda (j) `(set! ,sq (+ ,sq (select ,(valid s) (* ,(xv s j) ,(xv s j)) 0.0)))) js)) slots))
               ,@(reduce 'add sq tot)
               (var ,inv f32 (inv-sqrt (+ (/ ,tot (f32 ,n)) ,eps)))
               ,@(apply append
                   (map (lambda (s)
                          (map (lambda (j)
                                 `(set! ,(xv s j)
                                        ,(if w
                                             `(* (* ,(xv s j) ,inv) ,(load-in w (list-ref inputs w) s j))
                                             `(* ,(xv s j) ,inv))))
                               js))
                        slots))))]
          [(ew)
           (let ([lets (cadr st)] [result (caddr st)])
             (apply append
               (map (lambda (s)
                      (map (lambda (j)
                             `(set! ,(xv s j)
                                    (let (,@(map (lambda (i) (list (name 'x i) (load-in i (list-ref inputs i) s j))) (cdr idxs))
                                          (x ,(xv s j))
                                          ,@(map (lambda (b) (list (car b) (ew-expand (cadr b)))) lets))
                                      ,result)))
                           js))
                    slots)))]
          [else (error 'k:rowchain "unknown stage" st)]))
      (define (out-stmts o k)
        (let ([kind (cadr o)])
          (case kind
            [(f32)
             (apply append
               (map (lambda (s)
                      (map (lambda (j) `(when ,(valid s) (store ,(name 'o k) (+ rb ,(e s) ,j) ,(xv s j)))) js))
                    slots))]
            [(q8)
             (apply append
               (map (lambda (s)
                      (q8-stmts (map (lambda (j) (xv s j)) js)
                                `(+ (* row ,(div n 4)) ,(g s))
                                `(+ (* row ,(div n 32)) (/ ,(g s) 8))
                                (g s)
                                (valid s)))
                    slots))]
            [else (error 'k:rowchain "unknown output kind" o)])))
      `(kernel rowchain
         (offsets prefix) (workgroup ,wg)
         (buffers ,@(map (lambda (i spec) `(,(in-name i) ro ,(elem-ty (car spec)))) idxs inputs)
                  ,@(apply append
                      (map (lambda (o k)
                             (case (cadr o)
                               [(f32) (list `(,(name 'o k) wo f32))]
                               [(q8) (list '(xq wo u32) '(xd wo f32))]))
                           outs (range 0 (length outs)))))
         (push (nrows u32)
               ,@(apply append (map (lambda (i s) (if (eq? (cadr s) 'row) (list (list (rn-name i) 'u32)) '())) idxs inputs))
               ,@(map (lambda (j) (list (name 'c j) 'f32)) (range 0 nconst)))
         (shared (red f32 32))
         (let ([tid (local-id 0)] [row ,row-index])
           (when (< row nrows)
             (let ([rb (* row ,n)])
               ,@(apply append
                   (map (lambda (s)
                          `((var ,(g s) u32 (+ tid ,(* s wg)))
                            (var ,(valid s) bool (< ,(g s) ,ng))
                            (var ,(e s) u32 (* (min ,(g s) ,(- ng 1)) 4))))
                        slots))
               ,@(apply append
                   (map (lambda (s)
                          (map (lambda (j) `(var ,(xv s j) f32 ,(load-in 0 (car inputs) s j))) js))
                        slots))
               ,@(apply append
                   (map (lambda (st k)
                          (append (stage-stmts st k)
                                  (apply append
                                    (map (lambda (o ok) (if (= (car o) k) (out-stmts o ok) '()))
                                         outs (range 0 (length outs))))))
                        stages stage-ids))))))))

  ;; dst = concat(a, b) along `dim`; ad is a's extent along it.
  ;; push: n ne0 ne1 ne2 ad a0 a1 a2 a3 b0 b1 b2 b3 d0 d1 d2 d3
  (define (k:concat dim)
    (let ([id (list-ref dst-idx dim)])
      `(kernel ,(name 'concat dim)
         (offsets prefix) (workgroup ,wg)
         (buffers (a ro f32) (b ro f32) (d wo f32))
         (push ,@(apply u32s (append '(n ne0 ne1 ne2 ad) (strides 'a) (strides 'b) (strides 'd))))
         (let ([idx ,flat-index])
           (when (< idx n)
             (let ,dec-dst
               (if (< ,id ad)
                   (store d ,(dot dst-idx (strides 'd)) (load a ,(dot dst-idx (strides 'a))))
                   (store d ,(dot dst-idx (strides 'd))
                          (load b ,(dot (map (lambda (i) (if (eq? i id) `(- ,id ad) i)) dst-idx) (strides 'b)))))))))))

  ;; d[i] = start + i * step.   push: n start:f32 step:f32
  (define (k:arange)
    `(kernel arange
       (offsets prefix) (workgroup ,wg)
       (buffers (d wo f32))
       (push (n u32) (start f32) (step f32))
       (let ([idx ,flat-index])
         (when (< idx n) (store d idx (+ start (* (f32 idx) step)))))))

  ;; dst[i0, i1, i2, i3] = a[i0, rows[i1, i2, i3], i2, i3], a being f32, f16 or q8_0 (strides in
  ;; that type's elements / blocks).
  ;; push: n ne0 ne1 ne2 a0 a1 a2 a3 x1 x2 x3 d0 d1 d2 d3   (x: strides of the index tensor)
  (define (k:get-rows atype)
    `(kernel ,(name 'get_rows_ atype)
       (offsets prefix) (workgroup ,wg)
       (buffers (a ro ,(elem-ty atype)) (ix ro i32) (d wo f32))
       (push ,@(apply u32s (append '(n ne0 ne1 ne2) (strides 'a) '(x1 x2 x3) (strides 'd))))
       (let ([idx ,flat-index])
         (when (< idx n)
           (let (,@dec-dst [row (u32 (load ix (+ (* i1 x1) (* i2 x2) (* i3 x3))))])
             (store d ,(dot dst-idx (strides 'd))
                    ,(a-elem atype '(+ (* row a1) (* i2 a2) (* i3 a3)) 'i0)))))))

  ;; ---- row kernels (one workgroup per row) ----------------------------------------------------

  ;; Statements: reduce the per-thread partial `v` (a var) over the block with `op` (add | max)
  ;; into a new var `out`, equal in every thread. Uses shared `red` (at least 32 slots).
  (define (reduce op v out)
    (let ([sg (if (eq? op 'add) 'subgroup-add 'subgroup-max)]
          [comb (if (eq? op 'add) '+ 'max)]
          [init (if (eq? op 'add) 0.0 -inf.0)])
      `((let ([s (,sg ,v)]) (when (= (lane) 0) (store red (subgroup-id) s)))
        (barrier)
        (var ,out f32 ,init)
        (for (ri 0 (num-subgroups)) (set! ,out (,comb ,out (load red ri))))
        (barrier))))

  ;; Layer norm over ne0 (no affine): (x - mean) / sqrt(var + eps); the destination is dense.
  ;; push: nrows ncols ne1 ne2 a1 a2 a3 eps:f32
  (define (k:norm)
    `(kernel layernorm
       (offsets prefix) (workgroup ,wg)
       (buffers (a ro f32) (d wo f32))
       (push ,@(u32s 'nrows 'ncols 'ne1 'ne2 'a1 'a2 'a3) (eps f32))
       (shared (red f32 32))
       (let ([tid (local-id 0)] [row ,row-index])
         (when (< row nrows)
           (let (,@dec-row [abase (+ (* i1 a1) (* i2 a2) (* i3 a3))] [dbase (* row ncols)])
             (var sum f32 0.0)
             (for (c tid ncols ,wg) (set! sum (+ sum (load a (+ abase c)))))
             ,@(reduce 'add 'sum 'total)
             (let ([mean (/ total (f32 ncols))])
               (var sq f32 0.0)
               (for (c tid ncols ,wg)
                 (let ([dv (- (load a (+ abase c)) mean)]) (set! sq (+ sq (* dv dv)))))
               ,@(reduce 'add 'sq 'total2)
               (let ([inv (inv-sqrt (+ (/ total2 (f32 ncols)) eps))])
                 (for (c tid ncols ,wg)
                   (store d (+ dbase c) (* (- (load a (+ abase c)) mean) inv))))))))))

  ;; softmax(x * scale + mask) per row; mask rows are indexed (i1, i2 mod mne2, i3 mod mne3).
  ;; push: nrows ncols ne1 ne2 a1 a2 a3 scale:f32 [m1 m2 m3 mne2 mne3]
  (define (k:soft-max mask?)
    (define (v c)
      (if mask?
          `(+ (* (load a (+ abase ,c)) scale) (load mk (+ mbase ,c)))
          `(* (load a (+ abase ,c)) scale)))
    `(kernel ,(if mask? 'soft_max_mask 'soft_max)
       (offsets prefix) (workgroup ,wg)
       (buffers (a ro f32) ,@(if mask? '((mk ro f32)) '()) (d wo f32))
       (push ,@(u32s 'nrows 'ncols 'ne1 'ne2 'a1 'a2 'a3) (scale f32) ,@(if mask? (u32s 'm1 'm2 'm3 'mne2 'mne3) '()))
       (shared (red f32 32))
       (let ([tid (local-id 0)] [row ,row-index])
         (when (< row nrows)
           (let (,@dec-row
                 [abase (+ (* i1 a1) (* i2 a2) (* i3 a3))]
                 ,@(if mask? '([mbase (+ (* i1 m1) (* (rem i2 mne2) m2) (* (rem i3 mne3) m3))]) '())
                 [dbase (* row ncols)])
             (var m f32 -inf.0)
             (for (c tid ncols ,wg) (set! m (max m ,(v 'c))))
             ,@(reduce 'max 'm 'mx)
             (var s f32 0.0)
             (for (c tid ncols ,wg) (set! s (+ s (exp (- ,(v 'c) mx)))))
             ,@(reduce 'add 's 'total)
             (let ([inv (/ 1.0 total)])
               (for (c tid ncols ,wg)
                 (store d (+ dbase c) (* (exp (- ,(v 'c) mx)) inv)))))))))

  ;; dst[row] = scale * sum of the row.   push: nrows ncols ne1 ne2 a1 a2 a3 scale:f32
  (define (k:sum-rows)
    `(kernel sum_rows
       (offsets prefix) (workgroup ,wg)
       (buffers (a ro f32) (d wo f32))
       (push ,@(u32s 'nrows 'ncols 'ne1 'ne2 'a1 'a2 'a3) (scale f32))
       (shared (red f32 32))
       (let ([tid (local-id 0)] [row ,row-index])
         (when (< row nrows)
           (let (,@dec-row [abase (+ (* i1 a1) (* i2 a2) (* i3 a3))])
             (var sum f32 0.0)
             (for (c tid ncols ,wg) (set! sum (+ sum (load a (+ abase c)))))
             ,@(reduce 'add 'sum 'total)
             (when (= tid 0) (store d row (* total scale))))))))

  ;; The statements filling `total` tile elements with `nt` threads, unrolled: thread tid takes
  ;; elements tid, tid + nt, ... so the loads of one tile are independent and can all be in
  ;; flight before the first store (a loop would wait for each load in turn).
  (define (tile-loads total nt body)
    (map (lambda (i)
           (let ([l `(+ tid ,(* i nt))])
             (if (<= (* (+ i 1) nt) total)
                 (body l)
                 `(when (< ,l ,total) ,(body l)))))
         (range 0 (div (+ total nt -1) nt))))

  ;; ---- matmul -----------------------------------------------------------------------------------

  ;; dst[n, m] = sum_k a[m, k] * b[n, k], batched over (i2, i3) = z of the grid with a broadcast by
  ;; r2, r3 (ggml's mul_mat); a BM x BN tile per workgroup, each thread owning a TM x TN register
  ;; block spread over the tile, BK-deep shared tiles with a padded stride, zero-filled edges.
  ;; push: m n k a1 b1 d1 a2 a3 b2 b3 d2 d3 ne12 r2 r3
  ;; A is f32, f16 or q8_0 (dequantized while filling the shared tile; strides in blocks for q8_0).
  (define (k:matmul atype BM BN BK TM TN)
    (let* ([nx (div BM TM)] [ny (div BN TN)] [nt (* nx ny)] [ld (+ BK 1)]
           [is (range 0 TM)] [js (range 0 TN)])
      (define (acc i j) (name 'acc_ i '_ j))
      `(kernel ,(name 'matmul_ atype '_ BM 'x BN 'x BK '_ TM 'x TN)
         (offsets prefix) (workgroup ,nt)
         (buffers (a ro ,(elem-ty atype)) (b ro f32) (c wo f32))
         (push ,@(u32s 'm 'n 'k 'a1 'b1 'd1 'a2 'a3 'b2 'b3 'd2 'd3 'ne12 'r2 'r3))
         (shared (as f32 ,(* BM ld)) (bs f32 ,(* BN ld)))
         (let ([tid (local-id 0)]
               [m0 (* (group-id 0) ,BM)]
               [n0 (* (group-id 1) ,BN)]
               [tx (rem (local-id 0) ,nx)]
               [ty (/ (local-id 0) ,nx)]
               [z (group-id 2)])
           (let ([i2 (rem z ne12)] [i3 (/ z ne12)])
             (let ([aoff (+ (* (/ i2 r2) a2) (* (/ i3 r3) a3))]
                   [boff (+ (* i2 b2) (* i3 b3))]
                   [doff (+ (* i2 d2) (* i3 d3))])
               ,@(apply append (map (lambda (i) (map (lambda (j) `(var ,(acc i j) f32 0.0)) js)) is))
               (for (k0 0 k ,BK)
                 ,@(tile-loads (* BM BK) nt
                     (lambda (l)
                       `(let ([mm (/ ,l ,BK)] [kk (rem ,l ,BK)])
                          (store as (+ (* mm ,ld) kk)
                                 (guard (and (< (+ m0 mm) m) (< (+ k0 kk) k))
                                        ,(a-elem atype '(+ aoff (* (+ m0 mm) a1)) '(+ k0 kk))
                                        0.0)))))
                 ,@(tile-loads (* BN BK) nt
                     (lambda (l)
                       `(let ([nn (/ ,l ,BK)] [kk (rem ,l ,BK)])
                          (store bs (+ (* nn ,ld) kk)
                                 (guard (and (< (+ n0 nn) n) (< (+ k0 kk) k))
                                        (load b (+ boff (* (+ n0 nn) b1) k0 kk))
                                        0.0)))))
                 (barrier)
                 (for (kk 0 ,BK)
                   ,@(map (lambda (i) `(var ,(name 'ra_ i) f32 (load as (+ (* (+ tx ,(* i nx)) ,ld) kk)))) is)
                   ,@(map (lambda (j) `(var ,(name 'rb_ j) f32 (load bs (+ (* (+ ty ,(* j ny)) ,ld) kk)))) js)
                   ,@(apply append
                       (map (lambda (i)
                              (map (lambda (j) `(set! ,(acc i j) (fma ,(name 'ra_ i) ,(name 'rb_ j) ,(acc i j)))) js))
                            is)))
                 (barrier))
               ,@(apply append
                   (map (lambda (i)
                          (map (lambda (j)
                                 `(let ([mm (+ m0 tx ,(* i nx))] [nn (+ n0 ty ,(* j ny))])
                                    (when (and (< mm m) (< nn n))
                                      (store c (+ doff (* nn d1) mm) ,(acc i j)))))
                               js))
                        is))))))))

  ;; The same product on tensor cores (cooperative matrices, f16 inputs, f32 accumulation): the
  ;; shared tiles are f16, a workgroup is WM x WN subgroups of 32 lanes, each subgroup owns
  ;; (BM/WM/16) x (BN/WN/16) 16x16 accumulators.  Same buffers and push words as k:matmul.  The
  ;; accumulators go through shared memory to the destination, so edges and strides need no care.
  (define (k:matmul-cm atype BM BN BK WM WN)
    (let* ([nsg (* WM WN)] [nt (* 32 nsg)] [ld (+ BK 8)]
           [fm (div BM (* WM 16))] [fn (div BN (* WN 16))]
           [is (range 0 fm)] [js (range 0 fn)] [ks (range 0 (div BK 16))])
      (define (acc i j) (name 'acc_ i '_ j))
      (define (fa i) (name 'fa_ i))
      (define (fb j) (name 'fb_ j))
      `(kernel ,(name 'matmulcm_ atype '_ BM 'x BN 'x BK '_ WM 'x WN)
         (offsets prefix) (workgroup ,nt)
         (buffers (a ro ,(elem-ty atype)) (b ro f32) (c wo f32))
         (push ,@(u32s 'm 'n 'k 'a1 'b1 'd1 'a2 'a3 'b2 'b3 'd2 'd3 'ne12 'r2 'r3))
         (shared (as f16 ,(* BM ld)) (bs f16 ,(* BN ld)) (cs f32 ,(* 256 nsg)))
         (let ([tid (local-id 0)]
               [m0 (* (group-id 0) ,BM)]
               [n0 (* (group-id 1) ,BN)]
               [sg (/ (local-id 0) 32)]
               [l (rem (local-id 0) 32)]
               [z (group-id 2)])
           (let ([i2 (rem z ne12)] [i3 (/ z ne12)] [wm (rem sg ,WM)] [wn (/ sg ,WM)])
             (let ([aoff (+ (* (/ i2 r2) a2) (* (/ i3 r3) a3))]
                   [boff (+ (* i2 b2) (* i3 b3))]
                   [doff (+ (* i2 d2) (* i3 d3))])
               ,@(apply append (map (lambda (i) (map (lambda (j) `(cmat ,(acc i j) c f32)) js)) is))
               ,@(map (lambda (i) `(cmat ,(fa i) a f16)) is)
               ,@(map (lambda (j) `(cmat ,(fb j) b f16)) js)
               ,@(apply append (map (lambda (i) (map (lambda (j) `(cm-zero ,(acc i j))) js)) is))
               (for (k0 0 k ,BK)
                 ,@(tile-loads (* BM BK) nt
                     (lambda (l2)
                       `(let ([mm (/ ,l2 ,BK)] [kk (rem ,l2 ,BK)])
                          (store as (+ (* mm ,ld) kk)
                                 (f16 (guard (and (< (+ m0 mm) m) (< (+ k0 kk) k))
                                             ,(a-elem atype '(+ aoff (* (+ m0 mm) a1)) '(+ k0 kk))
                                             0.0))))))
                 ,@(tile-loads (* BN BK) nt
                     (lambda (l2)
                       `(let ([nn (/ ,l2 ,BK)] [kk (rem ,l2 ,BK)])
                          (store bs (+ (* nn ,ld) kk)
                                 (f16 (guard (and (< (+ n0 nn) n) (< (+ k0 kk) k))
                                             (load b (+ boff (* (+ n0 nn) b1) k0 kk))
                                             0.0))))))
                 (barrier)
                 ,@(apply append
                     (map (lambda (s)
                            (append
                              (map (lambda (i)
                                     `(cm-load ,(fa i) as (+ (* (+ (* wm ,(* fm 16)) ,(* i 16)) ,ld) ,(* s 16)) ,ld row))
                                   is)
                              (map (lambda (j)
                                     `(cm-load ,(fb j) bs (+ (* (+ (* wn ,(* fn 16)) ,(* j 16)) ,ld) ,(* s 16)) ,ld col))
                                   js)
                              (apply append
                                (map (lambda (i)
                                       (map (lambda (j) `(cm-muladd ,(acc i j) ,(fa i) ,(fb j) ,(acc i j))) js))
                                     is))))
                          ks))
                 (barrier))
               ,@(apply append
                   (map (lambda (i)
                          (apply append
                            (map (lambda (j)
                                   `((cm-store ,(acc i j) cs (* sg 256) 16 col)
                                     (barrier)
                                     ,@(map (lambda (e)
                                              `(let ([idx (+ l ,(* e 32))])
                                                 (let ([mm (+ m0 (* wm ,(* fm 16)) ,(* i 16) (rem idx 16))]
                                                       [nn (+ n0 (* wn ,(* fn 16)) ,(* j 16) (/ idx 16))])
                                                   (when (and (< mm m) (< nn n))
                                                     (store c (+ doff (* nn d1) mm) (load cs (+ (* sg 256) idx)))))))
                                            (range 0 8))
                                     (barrier)))
                                 js)))
                        is))))))))

  ;; q8_0 weights x int8 activations (k:quantize-q8) on the int8 tensor cores (PTX only: mma.sync m16n8k32).
  ;; One k step is one q8_0 block: the integer product of a 32-wide weight block and activation block is
  ;; scaled by their two scales and added to an f32 accumulator, so the result is the dp4a mat-vec's.
  ;; A tile of BM rows x BN columns per workgroup, WM x WN warps each owning (BM/WM) x (BN/WN) of it.
  ;; push: m n k a1 xq1 xd1 d1   (a1: q8_0 blocks per weight row, xq1: u32 words between activation columns,
  ;; xd1: scales between columns, d1: output stride between columns; k % 32 == 0)
  (define (k:matmul-q8mma BM BN WM WN)
    (let* ([nsg (* WM WN)] [nt (* 32 nsg)]
           [fm (div BM (* WM 16))] [fn (div BN (* WN 8))]
           [is (range 0 fm)] [js (range 0 fn)] [es '(0 1 2 3)])
      (define (acc i j e) (name 'acc_ i '_ j '_ e))
      (define (ar i q) (name 'ar_ i '_ q))
      (define (br j q) (name 'br_ j '_ q))
      (define (row i) `(+ (* wm ,(* fm 16)) ,(* i 16) g))
      (define (col j) `(+ (* wn ,(* fn 8)) ,(* j 8)))
      ;; statements loading block `kbx` (an expression) of this thread's share of both tiles into registers
      (define (fetch kbx)
        (append
          (map (lambda (e)
                 `(let ([idx (+ tid ,(* e nt))])
                    (let ([mm (/ idx 8)] [w (rem idx 8)])
                      (let ([base (* (+ (* (+ m0 mm) a1) ,kbx) 17)] [ok (< (+ m0 mm) m)])
                        (set! ,(name 'ra_ e) (guard ok (bit-or (u32 (load a (+ base 1 (* w 2)))) (shl (u32 (load a (+ base 2 (* w 2)))) 16)) 0))
                        (when (= w 0) (set! ,(name 'rs_ e) (guard ok (half->f32 (load a base)) 0.0)))))))
               (range 0 (div (* BM 8) nt)))
          (map (lambda (e)
                 `(let ([idx (+ tid ,(* e nt))])
                    (let ([nn (/ idx 8)] [w (rem idx 8)])
                      (let ([ok (< (+ n0 nn) n)])
                        (set! ,(name 'rb_ e) (guard ok (load xq (+ (* (+ n0 nn) xq1) (* ,kbx 8) w)) 0))
                        (when (= w 0) (set! ,(name 'rx_ e) (guard ok (load xd (+ (* (+ n0 nn) xd1) ,kbx)) 0.0)))))))
               (range 0 (div (* BN 8) nt)))))
      (unless (and (= (mod (* BM 8) nt) 0) (= (mod (* BN 8) nt) 0) (= (* fm WM 16) BM) (= (* fn WN 8) BN))
        (error 'k:matmul-q8mma "tile does not divide the workgroup" BM BN WM WN))
      `(kernel ,(name 'matmul_q8mma_ BM 'x BN '_ WM 'x WN)
         (offsets prefix) (workgroup ,nt)
         (buffers (a ro u16) (xq ro u32) (xd ro f32) (c wo f32))
         (push ,@(u32s 'm 'n 'k 'a1 'xq1 'xd1 'd1))
         (shared (as u32 ,(* BM 12)) (bs u32 ,(* BN 12)) (asc f32 ,BM) (bsc f32 ,BN))
         (let ([tid (local-id 0)] [m0 (* (group-id 0) ,BM)] [n0 (* (group-id 1) ,BN)])
           (let ([sg (/ tid 32)] [l (rem tid 32)] [nkb (/ k 32)])
             (let ([wm (rem sg ,WM)] [wn (/ sg ,WM)] [g (/ l 4)] [t (rem l 4)])
               ,@(apply append (map (lambda (i) (apply append (map (lambda (j) (map (lambda (e) `(var ,(acc i j e) f32 0.0)) es)) js))) is))
               ,@(map (lambda (e) `(var ,(name 'ia e) i32 0)) es)
               ;; The next block's global loads are issued before the current block is multiplied and land in
               ;; registers (ra_e / rs_e for the weight tile, rb_e / rx_e for the activations); they are written
               ;; to shared memory at the top of the following step, so their latency hides behind the mma.
               ,@(apply append
                   (map (lambda (e)
                          `((var ,(name 'ra_ e) u32 0) (var ,(name 'rs_ e) f32 0.0)))
                        (range 0 (div (* BM 8) nt))))
               ,@(apply append
                   (map (lambda (e)
                          `((var ,(name 'rb_ e) u32 0) (var ,(name 'rx_ e) f32 0.0)))
                        (range 0 (div (* BN 8) nt))))
               ,@(fetch '0)
               (for (kb 0 nkb)
                 ,@(map (lambda (e)
                          `(let ([idx (+ tid ,(* e nt))])
                             (let ([mm (/ idx 8)] [w (rem idx 8)])
                               (store as (+ (* mm 12) w) ,(name 'ra_ e))
                               (when (= w 0) (store asc mm ,(name 'rs_ e))))))
                        (range 0 (div (* BM 8) nt)))
                 ,@(map (lambda (e)
                          `(let ([idx (+ tid ,(* e nt))])
                             (let ([nn (/ idx 8)] [w (rem idx 8)])
                               (store bs (+ (* nn 12) w) ,(name 'rb_ e))
                               (when (= w 0) (store bsc nn ,(name 'rx_ e))))))
                        (range 0 (div (* BN 8) nt)))
                 (barrier)
                 ,@(fetch '(min (+ kb 1) (- nkb 1)))
                 ,@(apply append
                     (map (lambda (i)
                            `((var ,(ar i 0) u32 (load as (+ (* ,(row i) 12) t)))
                              (var ,(ar i 1) u32 (load as (+ (* (+ ,(row i) 8) 12) t)))
                              (var ,(ar i 2) u32 (load as (+ (* ,(row i) 12) 4 t)))
                              (var ,(ar i 3) u32 (load as (+ (* (+ ,(row i) 8) 12) 4 t)))
                              (var ,(name 'wl_ i) f32 (load asc ,(row i)))
                              (var ,(name 'wh_ i) f32 (load asc (+ ,(row i) 8)))))
                          is))
                 ,@(apply append
                     (map (lambda (j)
                            `((var ,(br j 0) u32 (load bs (+ (* (+ ,(col j) g) 12) t)))
                              (var ,(br j 1) u32 (load bs (+ (* (+ ,(col j) g) 12) 4 t)))
                              (var ,(name 'xs_ j '_ 0) f32 (load bsc (+ ,(col j) (* t 2))))
                              (var ,(name 'xs_ j '_ 1) f32 (load bsc (+ ,(col j) (* t 2) 1)))))
                          js))
                 ,@(apply append
                     (map (lambda (i)
                            (apply append
                              (map (lambda (j)
                                     `(,@(map (lambda (e) `(set! ,(name 'ia e) 0)) es)
                                       (mma-s8 ,(map (lambda (e) (name 'ia e)) es)
                                               ,(map (lambda (q) (ar i q)) '(0 1 2 3))
                                               ,(list (br j 0) (br j 1)))
                                       (set! ,(acc i j 0) (fma (f32 ia0) (* ,(name 'wl_ i) ,(name 'xs_ j '_ 0)) ,(acc i j 0)))
                                       (set! ,(acc i j 1) (fma (f32 ia1) (* ,(name 'wl_ i) ,(name 'xs_ j '_ 1)) ,(acc i j 1)))
                                       (set! ,(acc i j 2) (fma (f32 ia2) (* ,(name 'wh_ i) ,(name 'xs_ j '_ 0)) ,(acc i j 2)))
                                       (set! ,(acc i j 3) (fma (f32 ia3) (* ,(name 'wh_ i) ,(name 'xs_ j '_ 1)) ,(acc i j 3)))))
                                   js)))
                          is))
                 (barrier))
               ,@(apply append
                   (map (lambda (i)
                          (apply append
                            (map (lambda (j)
                                   (map (lambda (e)
                                          `(let ([mm (+ m0 ,(row i) ,(if (>= e 2) 8 0))]
                                                 [nn (+ n0 ,(col j) (* t 2) ,(mod e 2))])
                                             (when (and (< mm m) (< nn n))
                                               (store c (+ (* nn d1) mm) ,(acc i j e)))))
                                        es))
                                 js)))
                        is))))))))

  ;; ---- matrix-vector (a few columns) -----------------------------------------------------------

  ;; dst[c, row] = sum_k a[row, k] * b[c, k] for c < ncols (1..8): a subgroup takes R rows, lanes
  ;; taking 16-element chunks (half a q8_0 block) of the rows; the b chunk is loaded once for all R
  ;; rows (the activations are re-read by every row otherwise, which costs more than the weights).
  ;; Reduced with a subgroup sum. K % 16 == 0.
  ;; push: m k a1 b1 d1   (rows per workgroup = R * number of subgroups of the 128 threads)
  (define k:matvec
    (case-lambda
      [(atype ncols) (k:matvec atype ncols 1)]
      [(atype ncols R)
       (let ([cols (range 0 ncols)] [rs (range 0 R)])
         (define (acc r c) (name 'acc_ r '_ c))
         (define (rb r) (name 'rb r))
         (define (w i) (name 'w i))
         (define (x c i) (name 'x_ c '_ i))
         (define (weights r)
           (case atype
             [(f32 f16) (map (lambda (i) `(var ,(w i) f32 ,(load-f32 atype 'a `(+ ,(rb r) k0 ,i)))) (range 0 16))]
             [(q8_0)
              (cons `(var qs f32 (half->f32 (load a (* (+ ,(rb r) (/ ch 2)) 17))))
                    (apply append
                      (map (lambda (i)
                             (list `(var ,(name 'q i) u32 (u32 (load a (+ (* (+ ,(rb r) (/ ch 2)) 17) 1 (* (rem ch 2) 8) ,i))))
                                   `(var ,(w (* 2 i)) f32 (* qs (f32 (sbits ,(name 'q i) 0 8))))
                                   `(var ,(w (+ 1 (* 2 i))) f32 (* qs (f32 (sbits ,(name 'q i) 8 8))))))
                           (range 0 8))))]
             [else (error 'k:matvec "no such weight type" atype)]))
         `(kernel ,(name 'matvec_ atype '_ ncols 'x R)
            (offsets prefix) (workgroup 128)
            (buffers (a ro ,(elem-ty atype)) (b ro f32) (d wo f32))
            (push ,@(u32s 'm 'k 'a1 'b1 'd1))
            (let ([row0 (* (+ (* (group-id 0) (num-subgroups)) (subgroup-id)) ,R)] [l (lane)])
              (let (,@(map (lambda (r) `(,(rb r) (* (min (+ row0 ,r) (- m 1)) a1))) rs))
                ,@(apply append (map (lambda (r) (map (lambda (c) `(var ,(acc r c) f32 0.0)) cols)) rs))
                (for (ch l (/ k 16) (subgroup-size))
                  (let ([k0 (* ch 16)])
                    ,@(apply append (map (lambda (c) (map (lambda (i) `(var ,(x c i) f32 (load b (+ (* ,c b1) k0 ,i)))) (range 0 16))) cols))
                    ,@(apply append
                        (map (lambda (r)
                               (append (weights r)
                                       (apply append
                                         (map (lambda (c)
                                                (map (lambda (i) `(set! ,(acc r c) (fma ,(w i) ,(x c i) ,(acc r c)))) (range 0 16)))
                                              cols))))
                             rs))))
                ,@(apply append
                    (map (lambda (r)
                           (map (lambda (c)
                                  `(let ([t (subgroup-add ,(acc r c))])
                                     (when (and (= l 0) (< (+ row0 ,r) m)) (store d (+ (* ,c d1) row0 ,r) t))))
                                cols))
                         rs))))))]))

  ;; ---- rms norm, rope, set_rows, argmax -------------------------------------------------------------

  ;; Statements quantizing four consecutive elements (values `vs`, symbols) to int8 blocks as
  ;; k:quantize-q8 does: eight consecutive lanes hold the 32 elements of a block and share its amax
  ;; through a butterfly (so all eight must run this together). `g` is the lane's group index
  ;; (4 elements each), `widx` the u32 word to write in xq, `bidx` the scale slot in xd, `valid`
  ;; whether this lane stores (it takes part in the shuffles regardless).
  (define (q8-stmts vs widx bidx g valid)
    (define (rnd t) `(i32 (select (< ,t 0.0) (- (floor (+ (- ,t) 0.5))) (floor (+ ,t 0.5)))))
    (let ([v0 (car vs)] [v1 (cadr vs)] [v2 (caddr vs)] [v3 (cadddr vs)])
      `((let ([a0 (max (max (abs ,v0) (abs ,v1)) (max (abs ,v2) (abs ,v3)))]
              [a1 (max a0 (shuffle-xor a0 1))]
              [a2 (max a1 (shuffle-xor a1 2))]
              [amax (max a2 (shuffle-xor a2 4))])
          (let ([inv (select (> amax 0.0) (/ 127.0 amax) 0.0)])
            (let ([q0 ,(rnd `(* ,v0 inv))] [q1 ,(rnd `(* ,v1 inv))] [q2 ,(rnd `(* ,v2 inv))] [q3 ,(rnd `(* ,v3 inv))])
              (when ,valid
                (store xq ,widx (bit-or (bit-or (bit-and (u32 q0) 255) (shl (bit-and (u32 q1) 255) 8))
                                        (bit-or (shl (bit-and (u32 q2) 255) 16) (shl (bit-and (u32 q3) 255) 24))))
                (when (= (rem ,g 8) 0)
                  (store xd ,bidx (/ amax 127.0))))))))))

  ;; y = x * inv-sqrt(mean(x^2) + eps) [* w[c mod wn]].  push: nrows ncols ne1 ne2 a1 a2 a3 eps:f32 [wn]
  ;; With q8? the result is not stored as f32 but quantized for the dp4a mat-vecs that consume it:
  ;; buffers xq, xd instead of d (rows dense, ncols % 32 == 0).
  (define k:rms-norm
    (case-lambda
      [(weighted?) (k:rms-norm weighted? #f)]
      [(weighted? q8?)
       (define (val e) ; element e of the normalized row
         (if weighted?
             `(* (* (load a (+ abase ,e)) inv) (load w (rem ,e wn)))
             `(* (load a (+ abase ,e)) inv)))
       `(kernel ,(name (if weighted? 'rms_norm_mul 'rms_norm) (if q8? '_q8 ""))
          (offsets prefix) (workgroup ,wg)
          (buffers (a ro f32) ,@(if weighted? '((w ro f32)) '()) ,@(if q8? '((xq wo u32) (xd wo f32)) '((d wo f32))))
          (push ,@(u32s 'nrows 'ncols 'ne1 'ne2 'a1 'a2 'a3) (eps f32) ,@(if weighted? (u32s 'wn) '()))
          (shared (red f32 32))
          (let ([tid (local-id 0)] [row ,row-index])
            (when (< row nrows)
              (let (,@dec-row [abase (+ (* i1 a1) (* i2 a2) (* i3 a3))] [dbase (* row ncols)])
                (var sq f32 0.0)
                (for (c tid ncols ,wg) (let ([v (load a (+ abase c))]) (set! sq (+ sq (* v v)))))
                ,@(reduce 'add 'sq 'total)
                (let ([inv (inv-sqrt (+ (/ total (f32 ncols)) eps))])
                  ,@(if q8?
                        `((for (g tid (/ ncols 4) ,wg)
                            (let ([e (* g 4)])
                              (let ([v0 ,(val 'e)] [v1 ,(val '(+ e 1))] [v2 ,(val '(+ e 2))] [v3 ,(val '(+ e 3))])
                                ,@(q8-stmts '(v0 v1 v2 v3) '(+ (* row (/ ncols 4)) g) '(+ (* row (/ ncols 32)) (/ g 8)) 'g #t)))))
                        `((for (c tid ncols ,wg)
                            (store d (+ dbase c) ,(val 'c))))))))))]))

  ;; rms norm [* w] of a row, then (mode norm | neox) rope of it, then the store: dense rows of
  ;; `dty` (f32 | f16), or with set-rows? the single row goes to d[i0 + ix[0] * d1] (the K / V cache).
  ;; One workgroup per row; ncols is a compile-time constant (the normalized row waits in shared
  ;; memory for its rotation partner).  Source rows are dense, row = i1 + ne1 * i2 (positions per i2).
  ;; push: nrows ne1 n_dims has_ff eps:f32 freq_scale ext attn corr0 corr1 theta_scale [wn] [d1]
  (define (k:norm-rope ncols weighted? mode dty set-rows?)
    (define neox? (eq? mode 'neox))
    (define (val e)
      (if weighted?
          `(* (* (load a (+ abase ,e)) inv) (load w (rem ,e wn)))
          `(* (load a (+ abase ,e)) inv)))
    (define (out e v) (store-f32 dty 'd `(+ obase ,e) v))
    `(kernel ,(name 'norm_rope_ (if weighted? 'w "") (or mode 'plain) '_ dty (if set-rows? '_set "") '_ ncols)
       (offsets prefix) (workgroup ,wg)
       (buffers (a ro f32) ,@(if weighted? '((w ro f32)) '()) ,@(if mode '((pos ro i32) (ff ro f32)) '())
                ,@(if set-rows? '((ix ro i32)) '()) (d wo ,(elem-ty dty)))
       (push ,@(u32s 'nrows 'ne1 'n_dims 'has_ff) (eps f32) (freq_scale f32) (ext f32) (attn f32)
             (corr0 f32) (corr1 f32) (theta_scale f32) ,@(if weighted? (u32s 'wn) '()) ,@(if set-rows? (u32s 'd1) '()))
       (shared (red f32 32) (nrm f32 ,ncols))
       (let ([tid (local-id 0)] [row ,row-index])
         (when (< row nrows)
           (let ([abase (* row ,ncols)]
                 [obase ,(if set-rows? '(* (u32 (load ix 0)) d1) `(* row ,ncols))])
             (var sq f32 0.0)
             (for (c tid ,ncols ,wg) (let ([v (load a (+ abase c))]) (set! sq (+ sq (* v v)))))
             ,@(reduce 'add 'sq 'total)
             (let ([inv (inv-sqrt (+ (/ total ,(inexact ncols)) eps))])
               ,@(if mode
                     `((for (c tid ,ncols ,wg) (store nrm c ,(val 'c)))
                       (barrier)
                       (for (p tid (/ n_dims 2) ,wg)
                         (let ([i0 (* p 2)] [i2 (/ row ne1)])
                           (let ([theta-extrap (/ (* (f32 (load pos i2)) (pow theta_scale (f32 p)))
                                                  (select (!= has_ff 0) (load ff p) 1.0))])
                             (let ([interp (* freq_scale theta-extrap)]
                                   [ramp (- 1.0 (min 1.0 (max 0.0 (/ (- (f32 p) corr0) (max 0.001 (- corr1 corr0))))))])
                               (let ([mix (* ramp ext)])
                                 (let ([theta (select (!= ext 0.0) (+ (* interp (- 1.0 mix)) (* theta-extrap mix)) interp)]
                                       [mscale (select (!= ext 0.0) (* attn (+ 1.0 (* 0.1 (log (/ 1.0 freq_scale))))) attn)])
                                   (let ([cs (* (cos theta) mscale)] [sn (* (sin theta) mscale)]
                                         [ia ,(if neox? 'p 'i0)] [ib ,(if neox? '(+ p (/ n_dims 2)) '(+ i0 1))])
                                     (let ([x0 (load nrm ia)] [x1 (load nrm ib)])
                                       ,(out 'ia '(- (* x0 cs) (* x1 sn)))
                                       ,(out 'ib '(+ (* x0 sn) (* x1 cs)))))))))))
                       (for (c (+ n_dims tid) ,ncols ,wg) ,(out 'c '(load nrm c))))
                     `((for (c tid ,ncols ,wg) ,(out 'c (val 'c)))))))))))

  ;; ggml's rope (normal pairs adjacent, neox pairs split by n_dims/2), one thread per rotated pair:
  ;; row layout [ne0, ne1 (heads), ne2 (tokens)], positions in an i32 buffer, optional frequency
  ;; factors; the tail past n_dims is copied.  mode: norm | neox
  ;; push: n ne0 ne1 ne2 n_dims a1 a2 a3 d1 d2 d3 has_ff freq_scale:f32 ext:f32 attn:f32 corr0:f32 corr1:f32 theta_scale:f32
  (define (k:rope mode)
    (define neox? (eq? mode 'neox))
    `(kernel ,(name 'rope_ mode)
       (offsets prefix) (workgroup ,wg)
       (buffers (a ro f32) (pos ro i32) (ff ro f32) (d wo f32))
       (push ,@(u32s 'n 'ne0 'ne1 'ne2 'n_dims 'a1 'a2 'a3 'd1 'd2 'd3 'has_ff)
             (freq_scale f32) (ext f32) (attn f32) (corr0 f32) (corr1 f32) (theta_scale f32))
       (let ([idx ,flat-index])
         (when (< idx n)
           (let ([p (rem idx (/ ne0 2))] [row (/ idx (/ ne0 2))])
             (let ([i0 (* p 2)] [i1 (rem row ne1)] [r2 (/ row ne1)])
               (let ([i2 (rem r2 ne2)] [i3 (/ r2 ne2)])
                 (let ([abase (+ (* i1 a1) (* i2 a2) (* i3 a3))] [dbase (+ (* i1 d1) (* i2 d2) (* i3 d3))])
                   (if (>= i0 n_dims)
                       (begin
                         (store d (+ dbase i0) (load a (+ abase i0)))
                         (store d (+ dbase i0 1) (load a (+ abase i0 1))))
                       (let ([theta-extrap (/ (* (f32 (load pos i2)) (pow theta_scale (f32 p)))
                                              (select (!= has_ff 0) (load ff p) 1.0))])
                         (let ([interp (* freq_scale theta-extrap)]
                                [ramp (- 1.0 (min 1.0 (max 0.0 (/ (- (f32 p) corr0) (max 0.001 (- corr1 corr0))))))]
                                [mix (* ramp ext)]
                                [theta (select (!= ext 0.0) (+ (* interp (- 1.0 mix)) (* theta-extrap mix)) interp)]
                                [mscale (select (!= ext 0.0) (* attn (+ 1.0 (* 0.1 (log (/ 1.0 freq_scale))))) attn)]
                                [cs (* (cos theta) mscale)]
                                [sn (* (sin theta) mscale)]
                                [ia ,(if neox? '(+ abase p) '(+ abase i0))]
                                [ib ,(if neox? '(+ abase p (/ n_dims 2)) '(+ abase i0 1))]
                                [oa ,(if neox? '(+ dbase p) '(+ dbase i0))]
                                [ob ,(if neox? '(+ dbase p (/ n_dims 2)) '(+ dbase i0 1))]
                                [x0 (load a ia)]
                                [x1 (load a ib)])
                           (store d oa (- (* x0 cs) (* x1 sn)))
                           (store d ob (+ (* x0 sn) (* x1 cs))))))))))))))

  ;; dst[i0, idx[i1, i2 mod xn2, i3 mod xn3], i2, i3] = src[i0, i1, i2, i3], dst f32 or f16.
  ;; push: n ne0 ne1 ne2 a0 a1 a2 a3 x1 x2 x3 xn2 xn3 d0 d1 d2 d3
  (define (k:set-rows dty)
    `(kernel ,(name 'set_rows_ dty)
       (offsets prefix) (workgroup ,wg)
       (buffers (a ro f32) (ix ro i32) (d wo ,(elem-ty dty)))
       (push ,@(apply u32s (append '(n ne0 ne1 ne2) (strides 'a) '(x1 x2 x3 xn2 xn3) (strides 'd))))
       (let ([idx ,flat-index])
         (when (< idx n)
           (let (,@dec-dst [row (u32 (load ix (+ (* i1 x1) (* (rem i2 xn2) x2) (* (rem i3 xn3) x3))))])
             ,(store-f32 dty 'd '(+ (* i0 d0) (* row d1) (* i2 d2) (* i3 d3))
                         (load-f32 'f32 'a (dot dst-idx (strides 'a)))))))))

  ;; Index of the first maximum of each row (i32).  push: nrows ncols a1
  (define (k:argmax)
    `(kernel argmax
       (offsets prefix) (workgroup ,wg)
       (buffers (a ro f32) (d wo i32))
       (push ,@(u32s 'nrows 'ncols 'a1))
       (shared (red f32 32) (redi u32 32))
       (let ([tid (local-id 0)] [row ,row-index])
         (when (< row nrows)
           (let ([base (* row a1)])
             (var best f32 -inf.0)
             (for (c tid ncols ,wg) (set! best (max best (load a (+ base c)))))
             ,@(reduce 'max 'best 'gmax)
             (var first u32 ncols)
             (for (c tid ncols ,wg)
               (when (and (= (load a (+ base c)) gmax) (< c first)) (set! first c)))
             (let ([s (subgroup-min first)]) (when (= (lane) 0) (store redi (subgroup-id) s)))
             (barrier)
             (var res u32 ncols)
             (for (ri 0 (num-subgroups)) (set! res (min res (load redi ri))))
             (when (= tid 0) (store d row (i32 res))))))))

  ;; Long rows (the logits of a vocabulary): a workgroup per chunk of `chunk` elements finds the chunk's
  ;; maximum and the first index of it, (value, index as f32) pairs go to `part`; k:argmax-final merges them.
  ;; grid: chunks x rows.   push: nrows ncols a1 nchunks
  (define (k:argmax-part chunk)
    `(kernel argmax_part
       (offsets prefix) (workgroup ,wg)
       (buffers (a ro f32) (part wo f32))
       (push ,@(u32s 'nrows 'ncols 'a1 'nchunks))
       (shared (red f32 32) (redi u32 32))
       (let ([tid (local-id 0)] [ch (group-id 0)] [row (group-id 1)])
         (let ([base (* row a1)] [c0 (* ch ,chunk)])
           (let ([c1 (min (+ c0 ,chunk) ncols)])
             (var best f32 -inf.0)
             (for (c (+ c0 tid) c1 ,wg) (set! best (max best (load a (+ base c)))))
             ,@(reduce 'max 'best 'gmax)
             (var first u32 ncols)
             (for (c (+ c0 tid) c1 ,wg)
               (when (and (= (load a (+ base c)) gmax) (< c first)) (set! first c)))
             (let ([s (subgroup-min first)]) (when (= (lane) 0) (store redi (subgroup-id) s)))
             (barrier)
             (var res u32 ncols)
             (for (ri 0 (num-subgroups)) (set! res (min res (load redi ri))))
             (when (= tid 0)
               (store part (* 2 (+ (* row nchunks) ch)) gmax)
               (store part (+ (* 2 (+ (* row nchunks) ch)) 1) (f32 res))))))))

  ;; grid: rows.   push: nrows nchunks
  (define (k:argmax-final)
    `(kernel argmax_final
       (offsets prefix) (workgroup ,wg)
       (buffers (part ro f32) (d wo i32))
       (push ,@(u32s 'nrows 'nchunks))
       (shared (red f32 32) (redi u32 32))
       (let ([tid (local-id 0)] [row (group-id 0)])
         (var best f32 -inf.0)
         (for (c tid nchunks ,wg) (set! best (max best (load part (* 2 (+ (* row nchunks) c))))))
         ,@(reduce 'max 'best 'gmax)
         (var first u32 4294967295)
         (for (c tid nchunks ,wg)
           (when (= (load part (* 2 (+ (* row nchunks) c))) gmax)
             (set! first (min first (u32 (load part (+ (* 2 (+ (* row nchunks) c)) 1)))))))
         (let ([s (subgroup-min first)]) (when (= (lane) 0) (store redi (subgroup-id) s)))
         (barrier)
         (var res u32 4294967295)
         (for (ri 0 (num-subgroups)) (set! res (min res (load redi ri))))
         (when (= tid 0) (store d row (i32 res))))))

  ;; ---- flash attention ---------------------------------------------------------------------------

  ;; softmax(q . k * scale [softcapped] + mask) . v for one (query, head, batch) per workgroup,
  ;; K and V f16 or f32, the scores held in shared memory (kv <= maxkv), the mask f16. A subgroup
  ;; takes a key at a time (lanes over the head dimension), threads then take output elements.
  ;; push: nq nh hsk hsv kv gqa q1 q2 q3 k1 k2 k3 v1 v2 v3 d1 d2 d3 m1 m2 m3 mne2 mne3 scale:f32 cap:f32
  (define (k:flash-attn ktype vtype mask? softcap? maxkv)
    `(kernel flash_attn
       (offsets prefix) (workgroup ,wg)
       (buffers (q ro f32) (k ro ,(elem-ty ktype)) (v ro ,(elem-ty vtype)) ,@(if mask? '((mk ro u16)) '()) (d wo f32))
       (push ,@(u32s 'nq 'nh 'hsk 'hsv 'kv 'gqa 'q1 'q2 'q3 'k1 'k2 'k3 'v1 'v2 'v3 'd1 'd2 'd3 'm1 'm2 'm3 'mne2 'mne3)
             (scale f32) (cap f32))
       (shared (s f32 ,maxkv) (red f32 32))
       (let ([tid (local-id 0)] [qi (group-id 0)] [h (group-id 1)] [bi (group-id 2)])
         (let ([kvh (/ h gqa)])
           (let ([qbase (+ (* qi q1) (* h q2) (* bi q3))]
                 [kbase (+ (* kvh k2) (* bi k3))]
                 [vbase (+ (* kvh v2) (* bi v3))]
                 [mbase (+ (* qi m1) (* (rem h mne2) m2) (* (rem bi mne3) m3))])
             ,@(if mask?
                   (append `((var vm f32 0.0)
                             (for (j tid kv ,wg)
                               (when (> (half->f32 (load mk (+ mbase j))) -1.0e30)
                                 (set! vm (max vm (f32 (+ j 1)))))))
                           (reduce 'max 'vm 'vmax)
                           '((var kvl u32 (u32 vmax))))
                   '((var kvl u32 kv)))
             (for (j (subgroup-id) kvl (num-subgroups))
               (let ([mval ,(if mask? '(half->f32 (load mk (+ mbase j))) 0.0)])
                 (if (> mval -1.0e30)
                     (begin
                       (var dot f32 0.0)
                       (for (c (lane) hsk (subgroup-size))
                         (set! dot (fma (load q (+ qbase c)) ,(load-f32 ktype 'k '(+ kbase (* j k1) c)) dot)))
                       (let ([t (subgroup-add dot)])
                         (when (= (lane) 0)
                           (var sc f32 (* t scale))
                           ,@(if softcap? '((set! sc (* cap (tanh (/ sc cap))))) '())
                           ,@(if mask? '((set! sc (+ sc mval))) '())
                           (store s j sc))))
                     (when (= (lane) 0) (store s j -inf.0)))))
             (barrier)
             (var mx f32 -inf.0)
             (for (j tid kvl ,wg) (set! mx (max mx (load s j))))
             ,@(reduce 'max 'mx 'gmax)
             (var sum f32 0.0)
             (for (j tid kvl ,wg)
               (let ([p (exp (- (load s j) gmax))]) (store s j p) (set! sum (+ sum p))))
             ,@(reduce 'add 'sum 'gsum)
             (for (dd tid hsv ,wg)
               (var acc f32 0.0)
               (for (j 0 kvl)
                 (let ([p (load s j)])
                   (when (> p 0.0)
                     (set! acc (fma p ,(load-f32 vtype 'v '(+ vbase (* j v1) dd)) acc)))))
               (store d (+ (* h d1) (* qi d2) (* bi d3) dd) (/ acc gsum))))))))

  ;; Split-KV flash attention for few queries (decode): one workgroup per (key chunk, head, query)
  ;; instead of one per (head, query), so that a long cache keeps the whole GPU busy. Pass one
  ;; writes, per chunk, its maximum score m, the sum l of exp(score - m) and the unnormalized
  ;; output o = sum exp(score - m) * v; pass two merges the chunks of a head.
  ;; A chunk with no unmasked key writes l = 0 and does no more work.
  ;; The head sizes are compile-time (hsk % 32 == 0): each lane keeps its slice of the query in
  ;; registers, a subgroup takes four keys at a time with all their loads in flight before the
  ;; first multiply (a loop over a dynamic trip count would wait for each load in turn), and the
  ;; value sum takes eight keys at a time.
  ;; part: [m, l, o[0..hsv)] per (batch, query, head, chunk).
  ;; grid: x = chunk, y = head, z = batch * nq + query
  ;; push: nq nh hsk hsv kv gqa nchunks q1 q2 q3 k1 k2 k3 v1 v2 v3 m1 m2 m3 mne2 mne3 scale:f32 cap:f32
  (define (k:flash-attn-part ktype vtype mask? softcap? chunk hsk-c hsv-c)
    (let* ([rs (range 0 (div hsk-c 32))] [ku 4] [vu 8]
           [us (range 0 ku)] [vs (range 0 vu)] [dds (range 0 (div (+ hsv-c (- wg 1)) wg))])
      (define (qr j) (name 'qr j))
      (define (jj u) (name 'jj u))
      (define (dot u) (name 'dot u))
      (define (acc u) (name 'acc u))
      `(kernel flash_attn_part
         (offsets prefix) (workgroup ,wg)
         (buffers (q ro f32) (k ro ,(elem-ty ktype)) (v ro ,(elem-ty vtype)) ,@(if mask? '((mk ro u16)) '()) (part wo f32))
         (push ,@(u32s 'nq 'nh 'hsk 'hsv 'kv 'gqa 'nchunks 'q1 'q2 'q3 'k1 'k2 'k3 'v1 'v2 'v3 'm1 'm2 'm3 'mne2 'mne3)
               (scale f32) (cap f32))
         (shared (s f32 ,chunk) (red f32 32))
         (let ([tid (local-id 0)] [ch (group-id 0)] [h (group-id 1)])
           (let ([qi (rem (group-id 2) nq)] [bi (/ (group-id 2) nq)])
             (let ([kvh (/ h gqa)] [c0 (* ch ,chunk)])
               (let ([c1 (min (+ c0 ,chunk) kv)]
                     [qbase (+ (* qi q1) (* h q2) (* bi q3))]
                     [kbase (+ (* kvh k2) (* bi k3))]
                     [vbase (+ (* kvh v2) (* bi v3))]
                     [mbase (+ (* qi m1) (* (rem h mne2) m2) (* (rem bi mne3) m3))]
                     [pbase (* (+ (* (+ (* bi nq) qi) nh) h) nchunks)])
                 (let ([pb (* (+ pbase ch) (+ hsv 2))])
                   ;; the last unmasked key of the chunk (+1)
                   ,@(if mask?
                         (append `((var vm f32 0.0)
                                   (for (j (+ c0 tid) c1 ,wg)
                                     (when (> (half->f32 (load mk (+ mbase j))) -1.0e30)
                                       (set! vm (max vm (f32 (+ (- j c0) 1)))))))
                                 (reduce 'max 'vm 'vmax)
                                 '((var n u32 (u32 vmax))))
                         '((var n u32 (- c1 c0))))
                   (when (= n 0)
                     (when (= tid 0) (store part pb -inf.0) (store part (+ pb 1) 0.0)))
                   (when (> n 0)
                     (begin
                       ,@(map (lambda (j) `(var ,(qr j) f32 (load q (+ qbase (lane) ,(* 32 j))))) rs)
                       ;; scores, a subgroup taking keys sg, sg + S, sg + 2S, sg + 3S, then the next four S apart
                       (for (ib 0 (/ (+ n (- (* ,ku (num-subgroups)) 1)) (* ,ku (num-subgroups))))
                         (begin
                           ,@(map (lambda (u)
                                    `(var ,(jj u) u32 (+ (* (+ (* ib ,ku) ,u) (num-subgroups)) (subgroup-id))))
                                  us)
                           ;; loads first (a key past the chunk reads the last one: its result is not stored)
                           ,@(apply append
                               (map (lambda (u)
                                      (map (lambda (j)
                                             `(var ,(name 'kv u '_ j) f32
                                                   ,(load-f32 ktype 'k `(+ kbase (* (+ c0 (min ,(jj u) (- n 1))) k1) (lane) ,(* 32 j)))))
                                           rs))
                                    us))
                           ,@(map (lambda (u)
                                    `(var ,(dot u) f32 0.0))
                                  us)
                           ,@(apply append
                               (map (lambda (u)
                                      (map (lambda (j) `(set! ,(dot u) (fma ,(qr j) ,(name 'kv u '_ j) ,(dot u)))) rs))
                                    us))
                           ,@(map (lambda (u)
                                    `(let ([t (subgroup-add ,(dot u))])
                                       (when (and (= (lane) 0) (< ,(jj u) n))
                                         (var sc f32 (* t scale))
                                         ,@(if softcap? '((set! sc (* cap (tanh (/ sc cap))))) '())
                                         ,@(if mask?
                                               `((let ([mval (half->f32 (load mk (+ mbase c0 ,(jj u))))])
                                                   (store s ,(jj u) (select (> mval -1.0e30) (+ sc mval) -inf.0))))
                                               `((store s ,(jj u) sc))))))
                                  us)))
                       (barrier)
                       (var mx f32 -inf.0)
                       (for (jj tid n ,wg) (set! mx (max mx (load s jj))))
                       ,@(reduce 'max 'mx 'gmax)
                       (var sum f32 0.0)
                       (for (jj tid n ,wg)
                         (let ([p (exp (- (load s jj) gmax))]) (store s jj p) (set! sum (+ sum p))))
                       ,@(reduce 'add 'sum 'gsum)
                       (when (= tid 0) (store part pb gmax) (store part (+ pb 1) gsum))
                       ;; values: thread dd takes eight keys at a time, loads before multiplies
                       ,@(map (lambda (i)
                                `(let ([dd (+ tid ,(* i wg))])
                                   (when (< dd hsv)
                                     (var acc f32 0.0)
                                     (for (jb 0 (/ n ,vu))
                                       (begin
                                         ,@(map (lambda (u)
                                                  `(var ,(name 'vv u) f32 ,(load-f32 vtype 'v `(+ vbase (* (+ c0 (* jb ,vu) ,u) v1) dd))))
                                                vs)
                                         ,@(map (lambda (u)
                                                  `(set! acc (fma (load s (+ (* jb ,vu) ,u)) ,(name 'vv u) acc)))
                                                vs)))
                                     (for (jj (* (/ n ,vu) ,vu) n)
                                       (set! acc (fma (load s jj) ,(load-f32 vtype 'v '(+ vbase (* (+ c0 jj) v1) dd)) acc)))
                                     (store part (+ pb 2 dd) acc))))
                              dds)))))))))))

  ;; ---- paged attention (vLLM pool layouts, see docs/design/kernel-launcher.md) ---------------------
  ;; Key pool f16 [blocks, kv_heads, hs/8, bs, 8], value pool f16 [blocks, kv_heads, hs, bs]; kvb and
  ;; kvh are the strides (elements) of a block and of a kv head inside it. One workgroup per (chunk of
  ;; `chunk` keys, head, sequence); chunks past the sequence's context length write l = 0 and the merge
  ;; (k:flash-attn-combine, one query per sequence) skips them. hs % 32 == 0.
  ;; grid: nchunks x heads x seqs.   push: heads nchunks gqa maxblocks kvb kvh scale:f32
  ;; q f16 [hs, heads, seqs], tb i32 [maxblocks, seqs], ln i32 [seqs]; part as in k:flash-attn-part.
  (define (k:paged-attn-part hs bs chunk)
    (let* ([rs (range 0 (div hs 32))] [dds (range 0 (div (+ hs (- wg 1)) wg))])
      `(kernel paged_attn_part
         (offsets prefix) (workgroup ,wg)
         (buffers (q ro u16) (k ro u16) (v ro u16) (tb ro i32) (ln ro i32) (part wo f32))
         (push ,@(u32s 'heads 'nchunks 'gqa 'maxblocks 'kvb 'kvh) (scale f32))
         (shared (s f32 ,chunk) (red f32 32))
         (let ([tid (local-id 0)] [ch (group-id 0)] [h (group-id 1)] [si (group-id 2)])
           (let ([kvi (/ h gqa)] [c0 (* ch ,chunk)] [ctx (u32 (load ln si))]
                 [pb (* (+ (* (+ (* si heads) h) nchunks) ch) ,(+ hs 2))]
                 [tbase (* si maxblocks)])
             (var n u32 0)
             (when (< c0 ctx) (set! n (min (- ctx c0) ,chunk)))
             (when (= n 0)
               (when (= tid 0) (store part pb -inf.0) (store part (+ pb 1) 0.0)))
             (when (> n 0)
               (begin
                 ,@(map (lambda (j) `(var ,(name 'qr j) f32 (half->f32 (load q (+ (* (+ (* si heads) h) ,hs) (lane) ,(* 32 j)))))) rs)
                 ;; scores: a subgroup takes the keys sg, sg + S, ...
                 (let ([lo (+ (* (/ (lane) 8) ,(* bs 8)) (rem (lane) 8))] [hb (* kvi kvh)])
                   (for (jj (subgroup-id) n (num-subgroups))
                     (let ([j (+ c0 jj)])
                       (let ([kb (+ (* (u32 (load tb (+ tbase (/ j ,bs)))) kvb) hb (* (rem j ,bs) 8) lo)])
                         (var dot f32 0.0)
                         ,@(map (lambda (j) `(set! dot (fma ,(name 'qr j) (half->f32 (load k (+ kb ,(* 4 j bs 8)))) dot))) rs)
                         (let ([t (subgroup-add dot)])
                           (when (= (lane) 0) (store s jj (* t scale))))))))
                 (barrier)
                 (var mx f32 -inf.0)
                 (for (jj tid n ,wg) (set! mx (max mx (load s jj))))
                 ,@(reduce 'max 'mx 'gmax)
                 (var sum f32 0.0)
                 (for (jj tid n ,wg)
                   (let ([p (exp (- (load s jj) gmax))]) (store s jj p) (set! sum (+ sum p))))
                 ,@(reduce 'add 'sum 'gsum)
                 (when (= tid 0) (store part pb gmax) (store part (+ pb 1) gsum))
                 ,@(map (lambda (i)
                          `(let ([dd (+ tid ,(* i wg))])
                             (when (< dd ,hs)
                               (var acc f32 0.0)
                               (let ([vb (+ (* kvi kvh) (* dd ,bs))])
                                 (for (jj 0 n)
                                   (let ([j (+ c0 jj)])
                                     (set! acc (fma (load s jj)
                                                    (half->f32 (load v (+ vb (* (u32 (load tb (+ tbase (/ j ,bs)))) kvb) (rem j ,bs))))
                                                    acc)))))
                               (store part (+ pb 2 dd) acc))))
                        dds))))))))

  ;; Writes key and value f16 [hs * kv_heads, tokens] (dense) into the pools at slot = block * bs + offset,
  ;; slots i32 [2 * tokens] (one little-endian i64 per token; a negative slot skips the token).
  ;; One thread per element.   push: n hs kvheads kvb kvh
  (define (k:paged-cache-write bs)
    `(kernel paged_cache_write
       (offsets prefix) (workgroup ,wg)
       (buffers (key ro u16) (value ro u16) (kc rw u16) (vc rw u16) (slots ro i32))
       (push ,@(u32s 'n 'hs 'kvheads 'kvb 'kvh))
       (let ([idx ,flat-index])
         (when (< idx n)
           (let ([tok (/ idx (* hs kvheads))] [r (rem idx (* hs kvheads))])
             (let ([sl (load slots (* tok 2))] [hd (/ r hs)] [d (rem r hs)])
               (when (>= sl 0)
                 (let ([slot (u32 sl)])
                   (let ([base (+ (* (/ slot ,bs) kvb) (* hd kvh))] [off (rem slot ,bs)])
                     (store kc (+ base (* (/ d 8) ,(* bs 8)) (* off 8) (rem d 8)) (load key idx))
                     (store vc (+ base (* d ,bs) off) (load value idx)))))))))))

  ;; Merges the chunks of one (head, query, batch): d = sum_c exp(m_c - M) o_c / sum_c exp(m_c - M) l_c.
  ;; grid: x = head, y = batch * nq + query.   push: nq nh hsv nchunks d1 d2 d3
  (define k:flash-attn-combine
    (case-lambda
      [() (k:flash-attn-combine #f 'f32)]
      [(q8?) (k:flash-attn-combine q8? 'f32)]
      [(q8? dty)
       `(kernel ,(if q8? 'flash_attn_combine_q8 (if (eq? dty 'f16) 'flash_attn_combine_f16 'flash_attn_combine))
          (offsets prefix) (workgroup ,wg)
          (buffers (part ro f32) ,@(if q8? '((xq wo u32) (xd wo f32)) `((d wo ,(elem-ty dty)))))
          (push ,@(u32s 'nq 'nh 'hsv 'nchunks) ,@(if q8? '() (u32s 'd1 'd2 'd3)))
          (shared (red f32 32))
          (let ([tid (local-id 0)] [h (group-id 0)])
            (let ([qi (rem (group-id 1) nq)] [bi (/ (group-id 1) nq)])
              (let ([pbase (* (+ (* (+ (* bi nq) qi) nh) h) nchunks)])
                (var mx f32 -inf.0)
                (for (c 0 nchunks)
                  (when (> (load part (+ (* (+ pbase c) (+ hsv 2)) 1)) 0.0)
                    (set! mx (max mx (load part (* (+ pbase c) (+ hsv 2)))))))
                (var total f32 0.0)
                (for (c 0 nchunks)
                  (let ([pb (* (+ pbase c) (+ hsv 2))])
                    (when (> (load part (+ pb 1)) 0.0)
                      (set! total (+ total (* (load part (+ pb 1)) (exp (- (load part pb) mx))))))))
                ,@(if q8?
                      ;; a thread merges four consecutive outputs, then quantizes them (dense output)
                      `((for (g tid (/ hsv 4) ,wg)
                          ,@(map (lambda (j) `(var ,(name 'acc j) f32 0.0)) '(0 1 2 3))
                          (for (c 0 nchunks)
                            (let ([pb (* (+ pbase c) (+ hsv 2))])
                              (when (> (load part (+ pb 1)) 0.0)
                                (let ([w (exp (- (load part pb) mx))])
                                  ,@(map (lambda (j) `(set! ,(name 'acc j) (fma (load part (+ pb 2 (* g 4) ,j)) w ,(name 'acc j)))) '(0 1 2 3))))))
                          (let (,@(map (lambda (j) `(,(name 'v j) (select (> total 0.0) (/ ,(name 'acc j) total) 0.0))) '(0 1 2 3)))
                            ,@(q8-stmts '(v0 v1 v2 v3)
                                        '(+ (* (+ (* (+ (* bi nq) qi) nh) h) (/ hsv 4)) g)
                                        '(+ (* (+ (* (+ (* bi nq) qi) nh) h) (/ hsv 32)) (/ g 8))
                                        'g #t))))
                      `((for (dd tid hsv ,wg)
                          (var acc f32 0.0)
                          (for (c 0 nchunks)
                            (let ([pb (* (+ pbase c) (+ hsv 2))])
                              (when (> (load part (+ pb 1)) 0.0)
                                (set! acc (fma (load part (+ pb 2 dd)) (exp (- (load part pb) mx)) acc)))))
                          ,(store-f32 dty 'd '(+ (* h d1) (* qi d2) (* bi d3) dd) '(select (> total 0.0) (/ acc total) 0.0)))))))))]))

  ;; ---- int8 dot products (CUDA only: dp4a / warp shuffles are PTX intrinsics) ----------------------
  ;; Learned from ggml-cuda's mul_mat_vec_q (its PTX is ~13k dp4a): quantize the activations to
  ;; int8 blocks once, then a q8_0 weight chunk times an activation chunk is four dp4a instead of
  ;; sixteen dequantize + multiply-add, and the activation traffic is a quarter.

  ;; x (f32 rows of k) -> xq (four int8 per u32, k/4 words per row) and xd (f32 scale per 32
  ;; elements, k/32 per row), scale = amax / 127. One thread per four elements; eight threads form
  ;; a block, so its amax is a butterfly over eight lanes. k % 32 == 0. Threads past n repeat the
  ;; last group (they must still take part in the shuffles) and do not store.
  ;; push: n k x1      n = rows * k / 4
  (define (k:quantize-q8)
    `(kernel quantize_q8
       (offsets prefix) (workgroup ,wg)
       (buffers (x ro f32) (xq wo u32) (xd wo f32))
       (push ,@(u32s 'n 'k 'x1))
       (let ([idx ,flat-index])
         (let ([g (min idx (- n 1))])
           (let ([row (/ g (/ k 4))] [gi (rem g (/ k 4))])
             (let ([base (+ (* row x1) (* gi 4))])
               (let ([v0 (load x base)] [v1 (load x (+ base 1))] [v2 (load x (+ base 2))] [v3 (load x (+ base 3))])
                 ,@(q8-stmts '(v0 v1 v2 v3) 'g '(+ (* row (/ k 32)) (/ gi 8)) 'gi '(< idx n)))))))))

  ;; dst[c, row] = sum_k a[row, k] * x[c, k] for q8_0 weights a and activations quantized by
  ;; k:quantize-q8, ncols columns, a subgroup taking R rows (the activation words are loaded once
  ;; per chunk for all R rows). A chunk is half a q8_0 block: 16 weights, four dp4a.
  ;; push: m k a1 xq1 xd1 d1     (xq1: u32 words between columns, xd1: scales between columns)
  ;; U chunks per lane per iteration: all their loads are issued before the first dp4a, so a row of
  ;; k = 1536 (three chunks per lane) is one batch of loads instead of three that each wait.
  (define k:matvec-q8
    (case-lambda
      [(ncols R) (k:matvec-q8 ncols R 1)]
      [(ncols R U)
       (let ([cols (range 0 ncols)] [rs (range 0 R)] [us (range 0 U)])
         (define (acc r c) (name 'acc_ r '_ c))
         (define (rb r) (name 'rb r))
         (define (ws r u) (name 'ws r '_ u))
         (define (w r j u) (name 'w_ r '_ j '_ u))
         (define (xw c j u) (name 'xw_ c '_ j '_ u))
         (define (xs c u) (name 'xs c '_ u))
         (define (blk u) (name 'blk u))
         (define (hf u) (name 'hf u))
         (define (valid u) (name 'valid u))
         `(kernel ,(name 'matvec_q8dot_ ncols 'x R (if (= U 1) "" (name 'u U)))
            (offsets prefix) (workgroup 128)
            (buffers (a ro u16) (xq ro u32) (xd ro f32) (d wo f32))
            (push ,@(u32s 'm 'k 'a1 'xq1 'xd1 'd1))
            (let ([row0 (* (+ (* (group-id 0) (num-subgroups)) (subgroup-id)) ,R)] [l (lane)] [nch (/ k 16)])
              (let (,@(map (lambda (r) `(,(rb r) (* (min (+ row0 ,r) (- m 1)) a1))) rs))
                ,@(apply append (map (lambda (r) (map (lambda (c) `(var ,(acc r c) f32 0.0)) cols)) rs))
                (for (i 0 (/ (+ nch (- (* ,U (subgroup-size)) 1)) (* ,U (subgroup-size))))
                  (begin
                    ;; chunk u of this iteration; one past the end reads the last chunk and counts for nothing
                    ,@(apply append
                        (map (lambda (u)
                               `((var ,(name 'ch u) u32 (+ l (* (subgroup-size) (+ (* i ,U) ,u))))
                                 (var ,(valid u) bool (< ,(name 'ch u) nch))
                                 (var ,(blk u) u32 (/ (min ,(name 'ch u) (- nch 1)) 2))
                                 (var ,(hf u) u32 (rem (min ,(name 'ch u) (- nch 1)) 2))))
                             us))
                    ,@(apply append
                        (map (lambda (u)
                               (apply append
                                 (map (lambda (c)
                                        (cons `(var ,(xs c u) f32 (select ,(valid u) (load xd (+ (* ,c xd1) ,(blk u))) 0.0))
                                              (map (lambda (j) `(var ,(xw c j u) u32 (load xq (+ (* ,c xq1) (* ,(blk u) 8) (* ,(hf u) 4) ,j)))) (range 0 4))))
                                      cols)))
                             us))
                    ,@(apply append
                        (map (lambda (u)
                               (apply append
                                 (map (lambda (r)
                                        (let ([base `(* (+ ,(rb r) ,(blk u)) 17)])
                                          (cons `(var ,(ws r u) f32 (half->f32 (load a ,base)))
                                                (map (lambda (j)
                                                       `(var ,(w r j u) u32
                                                             (bit-or (u32 (load a (+ ,base 1 (* ,(hf u) 8) ,(* 2 j))))
                                                                     (shl (u32 (load a (+ ,base 2 (* ,(hf u) 8) ,(* 2 j)))) 16))))
                                                     (range 0 4)))))
                                      rs)))
                             us))
                    ,@(apply append
                        (map (lambda (u)
                               (apply append
                                 (map (lambda (r)
                                        (map (lambda (c)
                                               `(set! ,(acc r c)
                                                      (fma (* ,(ws r u) ,(xs c u))
                                                           (f32 (dp4a ,(w r 0 u) ,(xw c 0 u)
                                                                      (dp4a ,(w r 1 u) ,(xw c 1 u)
                                                                            (dp4a ,(w r 2 u) ,(xw c 2 u)
                                                                                  (dp4a ,(w r 3 u) ,(xw c 3 u) 0)))))
                                                           ,(acc r c))))
                                             cols))
                                      rs)))
                             us))))
                ,@(apply append
                    (map (lambda (r)
                           (map (lambda (c)
                                  `(let ([t (subgroup-add ,(acc r c))])
                                     (when (and (= l 0) (< (+ row0 ,r) m)) (store d (+ (* ,c d1) row0 ,r) t))))
                                cols))
                         rs))))))]))

  ;; gelu(a . x) * (b . x) for two q8_0 matrices a, b over one int8 activation vector, written as int8
  ;; blocks (xq, xd): the gate / up / gelu-mul of an FFN in one launch.  A workgroup of 32/R subgroups
  ;; (subgroup size 32) owns 32 rows = one output block; m must be a multiple of 32.
  ;; push: m k a1  (a and b share the row stride)
  (define (k:gateup-q8 R U)
    (let ([rs (range 0 R)] [us (range 0 U)] [mats '(a b)])
      (define (acc b r) (name 'acc_ b r))
      (define (rb b r) (name 'rb_ b r))
      (define (ws b r u) (name 'ws_ b r '_ u))
      (define (w b r j u) (name 'w_ b r '_ j '_ u))
      (define (xw j u) (name 'xw_ j '_ u))
      (define (xs u) (name 'xs_ u))
      (define (blk u) (name 'blk u))
      (define (hf u) (name 'hf u))
      (define (valid u) (name 'valid u))
      (define (each-br f) (apply append (map (lambda (b) (map (lambda (r) (f b r)) rs)) mats)))
      (define (each-br* f) (apply append (each-br f)))
      `(kernel ,(name 'gateup_q8dot_ R (if (= U 1) "" (name 'u U)))
         (offsets prefix) (workgroup ,(* 32 (/ 32 R)))
         (buffers (a ro u16) (b ro u16) (aq ro u32) (ad ro f32) (xq wo u32) (xd wo f32))
         (push ,@(u32s 'm 'k 'a1))
         (shared (hv f32 32))
         (let ([row0 (+ (* (group-id 0) 32) (* (subgroup-id) ,R))] [l (lane)] [nch (/ k 16)])
           (let (,@(each-br (lambda (b r) `(,(rb b r) (* (min (+ row0 ,r) (- m 1)) a1)))))
             ,@(each-br (lambda (b r) `(var ,(acc b r) f32 0.0)))
             (for (i 0 (/ (+ nch (- (* ,U (subgroup-size)) 1)) (* ,U (subgroup-size))))
               (begin
                 ,@(apply append
                     (map (lambda (u)
                            `((var ,(name 'ch u) u32 (+ l (* (subgroup-size) (+ (* i ,U) ,u))))
                              (var ,(valid u) bool (< ,(name 'ch u) nch))
                              (var ,(blk u) u32 (/ (min ,(name 'ch u) (- nch 1)) 2))
                              (var ,(hf u) u32 (rem (min ,(name 'ch u) (- nch 1)) 2))
                              (var ,(xs u) f32 (select ,(valid u) (load ad ,(blk u)) 0.0))
                              ,@(map (lambda (j) `(var ,(xw j u) u32 (load aq (+ (* ,(blk u) 8) (* ,(hf u) 4) ,j)))) (range 0 4))))
                          us))
                 ,@(apply append
                     (map (lambda (u)
                            (each-br*
                              (lambda (b r)
                                (let ([base `(* (+ ,(rb b r) ,(blk u)) 17)])
                                  `((var ,(ws b r u) f32 (half->f32 (load ,b ,base)))
                                    ,@(map (lambda (j)
                                             `(var ,(w b r j u) u32
                                                   (bit-or (u32 (load ,b (+ ,base 1 (* ,(hf u) 8) ,(* 2 j))))
                                                           (shl (u32 (load ,b (+ ,base 2 (* ,(hf u) 8) ,(* 2 j)))) 16))))
                                           (range 0 4)))))))
                          us))
                 ,@(apply append
                     (map (lambda (u)
                            (each-br
                              (lambda (b r)
                                `(set! ,(acc b r)
                                       (fma (* ,(ws b r u) ,(xs u))
                                            (f32 (dp4a ,(w b r 0 u) ,(xw 0 u)
                                                       (dp4a ,(w b r 1 u) ,(xw 1 u)
                                                             (dp4a ,(w b r 2 u) ,(xw 2 u)
                                                                   (dp4a ,(w b r 3 u) ,(xw 3 u) 0)))))
                                            ,(acc b r))))))
                          us))))
             ,@(map (lambda (r)
                      `(let ([tg (subgroup-add ,(acc 'a r))] [tu (subgroup-add ,(acc 'b r))])
                         (when (= l 0) (store hv (+ (* (subgroup-id) ,R) ,r) (* ,(unary-expr 'gelu 'tg) tu)))))
                    rs)
             (barrier)
             (when (= (subgroup-id) 0)
               (let ([grp (group-id 0)])
                 ,@(q8-stmts (map (lambda (i) `(select (< l 8) (load hv (min (+ (* l 4) ,i) 31)) 0.0)) (range 0 4))
                             '(+ (* grp 8) l) 'grp 'l '(< l 8)))))))))

  ;; ---- repacked q8_0 weights and vector loads ----------------------------------------------------------
  ;; A q8_0 block is 34 bytes, so its words are only 2-byte aligned (ggml-cuda builds every 32-bit
  ;; word from two 16-bit loads, see its PTX). Repacked once when the model is loaded, the quants of a
  ;; block become eight aligned u32 words (one 16-byte vector load per half block) and the f16
  ;; scales move to their own array: same bytes, one load instead of nine.

  ;; a: q8_0 blocks as u16 (17 per block) -> qs: 8 u32 per block, ds: the scale of each block.
  ;; push: n   (n = blocks * 8)
  (define (k:repack-q8)
    `(kernel repack_q8
       (offsets prefix) (workgroup ,wg)
       (buffers (a ro u16) (qs wo u32) (ds wo u16))
       (push (n u32))
       (let ([idx ,flat-index])
         (when (< idx n)
           (let ([blk (/ idx 8)] [j (rem idx 8)])
             (let ([base (* blk 17)])
               (store qs idx (bit-or (u32 (load a (+ base 1 (* j 2)))) (shl (u32 (load a (+ base 2 (* j 2)))) 16)))
               (when (= j 0) (store ds blk (load a base)))))))))

  ;; k:matvec-q8 over repacked weights: the weight words and the activation words of a chunk are
  ;; vector loads (qs must be 16-byte aligned, as must xq: the lowering checks).
  ;; push: m k a1 xq1 xd1 d1     (a1: blocks per row of the weights)
  (define (k:matvec-q8r ncols R)
    (let ([cols (range 0 ncols)] [rs (range 0 R)])
      (define (acc r c) (name 'acc_ r '_ c))
      (define (rb r) (name 'rb r))
      (define (ws r) (name 'ws r))
      (define (w r j) (name 'w_ r '_ j))
      (define (xw c j) (name 'xw_ c '_ j))
      (define (xs c) (name 'xs c))
      `(kernel ,(name 'matvec_q8r_ ncols 'x R)
         (offsets prefix) (workgroup 128)
         (buffers (qs ro u32) (ds ro u16) (xq ro u32) (xd ro f32) (d wo f32))
         (push ,@(u32s 'm 'k 'a1 'xq1 'xd1 'd1))
         (let ([row0 (* (+ (* (group-id 0) (num-subgroups)) (subgroup-id)) ,R)] [l (lane)])
           (let (,@(map (lambda (r) `(,(rb r) (* (min (+ row0 ,r) (- m 1)) a1))) rs))
             ,@(apply append (map (lambda (r) (map (lambda (c) `(var ,(acc r c) f32 0.0)) cols)) rs))
             (for (ch l (/ k 16) (subgroup-size))
               (let ([blk (/ ch 2)] [hf (rem ch 2)])
                 ,@(apply append
                     (map (lambda (c)
                            (list `(var ,(xs c) f32 (load xd (+ (* ,c xd1) blk)))
                                  `(vload ,(map (lambda (j) (xw c j)) (range 0 4)) xq (+ (* ,c xq1) (* blk 8) (* hf 4)))))
                          cols))
                 ,@(apply append
                     (map (lambda (r)
                            (append
                              (list `(var ,(ws r) f32 (half->f32 (load ds (+ ,(rb r) blk))))
                                    `(vload ,(map (lambda (j) (w r j)) (range 0 4)) qs (+ (* (+ ,(rb r) blk) 8) (* hf 4))))
                              (map (lambda (c)
                                     `(set! ,(acc r c)
                                            (fma (* ,(ws r) ,(xs c))
                                                 (f32 (dp4a ,(w r 0) ,(xw c 0)
                                                            (dp4a ,(w r 1) ,(xw c 1)
                                                                  (dp4a ,(w r 2) ,(xw c 2)
                                                                        (dp4a ,(w r 3) ,(xw c 3) 0)))))
                                                 ,(acc r c))))
                                   cols)))
                          rs))))
             ,@(apply append
                 (map (lambda (r)
                        (map (lambda (c)
                               `(let ([t (subgroup-add ,(acc r c))])
                                  (when (and (= l 0) (< (+ row0 ,r) m)) (store d (+ (* ,c d1) row0 ,r) t))))
                             cols))
                      rs)))))))
)
