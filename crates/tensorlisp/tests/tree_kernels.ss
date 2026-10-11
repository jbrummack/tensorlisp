;; Tree kernels used by tests/tree_vulkan.rs. Each is a Scheme function returning a
;; (kernel ...) datum: tile sizes, unrolling and reductions are staged here, so the
;; compiler only ever sees straight-line code and counted loops.
(import (rnrs) (tl tree))

(define (range a b) (if (>= a b) '() (cons a (range (+ a 1) b))))
(define (name . parts)
  (string->symbol (apply string-append (map (lambda (p) (if (number? p) (number->string p) (symbol->string p))) parts))))

;; d[i] = a[i] + b[i]
(define add-kernel
  '(kernel add
     (workgroup 256)
     (buffers (a ro f32) (b ro f32) (d wo f32))
     (push (n u32))
     (let ([i (global-id 0)])
       (when (< i n) (store d i (+ (load a i) (load b i)))))))

;; d[i] = gelu(a[i] + b[i]), the tanh approximation: one fused kernel for what ggml runs as two.
(define add-gelu-kernel
  '(kernel add_gelu
     (workgroup 256)
     (buffers (a ro f32) (b ro f32) (d wo f32))
     (push (n u32))
     (let ([i (global-id 0)])
       (when (< i n)
         (let ([x (+ (load a i) (load b i))])
           (store d i (* 0.5 x (+ 1.0 (tanh (* 0.7978845608 (+ x (* 0.044715 x x x))))))))))))

;; Statements reducing red[0..wg) into red[0] with `op` by halving (barrier after each step).
(define (tree-reduce op wg)
  (let loop ([s (quotient wg 2)] [acc '()])
    (if (< s 1)
        (reverse acc)
        (loop (quotient s 2)
              (cons '(barrier)
                    (cons `(when (< tid ,s)
                             (store red tid (,op (load red tid) (load red (+ tid ,s)))))
                          acc))))))

;; One row per workgroup, row length `ncols`: y = softmax(x). Reductions through shared memory.
(define (softmax-shared-kernel wg)
  `(kernel softmax_shared
     (workgroup ,wg)
     (buffers (x ro f32) (y wo f32))
     (push (ncols u32))
     (shared (red f32 ,wg))
     (let ([tid (local-id 0)] [base (* (group-id 0) ncols)])
       (var m f32 -inf.0)
       (for (c tid ncols ,wg) (set! m (max m (load x (+ base c)))))
       (store red tid m)
       (barrier)
       ,@(tree-reduce 'max wg)
       (let ([mx (load red 0)])
         (barrier)
         (var s f32 0.0)
         (for (c tid ncols ,wg) (set! s (+ s (exp (- (load x (+ base c)) mx)))))
         (store red tid s)
         (barrier)
         ,@(tree-reduce '+ wg)
         (let ([inv (/ 1.0 (load red 0))])
           (for (c tid ncols ,wg)
             (store y (+ base c) (* (exp (- (load x (+ base c)) mx)) inv))))))))

;; The same with subgroup reductions: one shared slot per subgroup instead of log2(wg) barriers.
(define (softmax-subgroup-kernel wg)
  `(kernel softmax_subgroup
     (workgroup ,wg)
     (buffers (x ro f32) (y wo f32))
     (push (ncols u32))
     (shared (red f32 32))
     (let ([tid (local-id 0)] [base (* (group-id 0) ncols)])
       (var m f32 -inf.0)
       (for (c tid ncols ,wg) (set! m (max m (load x (+ base c)))))
       (let ([sm (subgroup-max m)])
         (when (= (lane) 0) (store red (subgroup-id) sm)))
       (barrier)
       (var mx f32 -inf.0)
       (for (i 0 (num-subgroups)) (set! mx (max mx (load red i))))
       (barrier)
       (var s f32 0.0)
       (for (c tid ncols ,wg) (set! s (+ s (exp (- (load x (+ base c)) mx)))))
       (let ([ss (subgroup-add s)])
         (when (= (lane) 0) (store red (subgroup-id) ss)))
       (barrier)
       (var total f32 0.0)
       (for (i 0 (num-subgroups)) (set! total (+ total (load red i))))
       (let ([inv (/ 1.0 total)])
         (for (c tid ncols ,wg)
           (store y (+ base c) (* (exp (- (load x (+ base c)) mx)) inv)))))))

;; dst[n * M + m] = sum_k a[m * K + k] * b[n * K + k]   (ggml's mul_mat layout: a is M rows of K)
;; Naive: one thread per output, everything from global memory.
(define matmul-naive-kernel
  '(kernel matmul_naive
     (workgroup 16 16)
     (buffers (a ro f32) (b ro f32) (c wo f32))
     (push (m u32) (n u32) (k u32))
     (let ([mm (global-id 0)] [nn (global-id 1)])
       (when (and (< mm m) (< nn n))
         (var acc f32 0.0)
         (for (kk 0 k) (set! acc (fma (load a (+ (* mm k) kk)) (load b (+ (* nn k) kk)) acc)))
         (store c (+ (* nn m) mm) acc)))))

(define (tile-loads total nt body)
  (map (lambda (i)
         (let ([l `(+ tid ,(* i nt))])
           (if (<= (* (+ i 1) nt) total)
               (body l)
               `(when (< ,l ,total) ,(body l)))))
       (range 0 (quotient (+ total nt -1) nt))))

;; Tiled: a BM x BN tile of c per workgroup, each thread owning a TM x TN register block spread
;; over the tile (so neighbouring threads write neighbouring m), BK-deep shared tiles with a
;; padded stride. Out-of-range tile elements load as zero.
(define (matmul-tiled-kernel BM BN BK TM TN)
  (let* ([nx (quotient BM TM)] [ny (quotient BN TN)] [nt (* nx ny)] [ld (+ BK 1)]
         [is (range 0 TM)] [js (range 0 TN)])
    (define (acc i j) (name 'acc_ i '_ j))
    `(kernel matmul_tiled
       (workgroup ,nt)
       (buffers (a ro f32) (b ro f32) (c wo f32))
       (push (m u32) (n u32) (k u32))
       (shared (as f32 ,(* BM ld)) (bs f32 ,(* BN ld)))
       (let ([tid (local-id 0)]
             [m0 (* (group-id 0) ,BM)]
             [n0 (* (group-id 1) ,BN)]
             [tx (rem (local-id 0) ,nx)]
             [ty (/ (local-id 0) ,nx)])
         ,@(apply append (map (lambda (i) (map (lambda (j) `(var ,(acc i j) f32 0.0)) js)) is))
         (for (k0 0 k ,BK)
           ,@(tile-loads (* BM BK) nt
               (lambda (l)
                 `(let ([mm (/ ,l ,BK)] [kk (rem ,l ,BK)])
                    (store as (+ (* mm ,ld) kk)
                           (guard (and (< (+ m0 mm) m) (< (+ k0 kk) k)) (load a (+ (* (+ m0 mm) k) k0 kk)) 0.0)))))
           ,@(tile-loads (* BN BK) nt
               (lambda (l)
                 `(let ([nn (/ ,l ,BK)] [kk (rem ,l ,BK)])
                    (store bs (+ (* nn ,ld) kk)
                           (guard (and (< (+ n0 nn) n) (< (+ k0 kk) k)) (load b (+ (* (+ n0 nn) k) k0 kk)) 0.0)))))
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
                                (store c (+ (* nn m) mm) ,(acc i j)))))
                         js))
                  is))))))
