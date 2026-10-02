;; Native Metal executor: matrix multiplication lowering.
;;
;; The runtime lowers every graph node to GPU dispatches itself (no ggml
;; backend). For MUL_MAT the decision which kernel to run and how to launch it
;; is made here, in Scheme: this is a port of ggml-metal's ggml_metal_op_mul_mat
;; and the pipeline getters it uses, over plain data.
;;
;; A tensor is described as (type-name (ne0 ne1 ne2 ne3) (nb0 nb1 nb2 nb3)),
;; ggml order, nb in bytes. `props` is (simdgroup-matrix? max-threadgroup-bytes).
;;
;; The reply is a list of dispatches, in order:
;;   (kernel-name consts args (groups-x groups-y groups-z) (threads-x threads-y threads-z) smem)
;;   consts: ((index kind value) ...)            kind: bool | i16 | i32
;;   args:   ((slot tensor src0|src1|dst) ...)   a tensor's buffer at its offset
;;           ((slot bytes struct-name (kind value) ...) ...)
;;                                               a ggml_metal_kargs_<struct-name>, fields
;;                                               in declaration order, kind: bool i16 i32 u32 i64 u64 f32
;; The Rust side checks the packed size of each struct against ggml's definition.

(define (nat-ty d) (car d))
(define (nat-ne d i) (list-ref (cadr d) i))
(define (nat-nb d i) (list-ref (caddr d) i))
(define (nat-transposed? d) (> (nat-nb d 0) (nat-nb d 1)))

;; Function-constant bases, from ggml-metal-impl.h.
(define nat-fc-mul-mv 600)
(define nat-fc-mul-mm 700)

(define (nat-i32 v) (list 'i32 v))
(define (nat-u64 v) (list 'u64 v))
(define (nat-i16 v) (list 'i16 v))

;; Number of simdgroups, src0 rows per simdgroup and threadgroup memory of the
;; matrix-vector kernel for a src0 type. Float types depend on the row length.
;; (nsg nr0 smem name-suffix)
(define (nat-mv-shape ty ne00)
  (cond
    [(member ty '("f32" "f16" "bf16"))
     (if (< ne00 32)
         (list 1 32 0 "_short")
         (let ([nsg (min 4 (quotient (+ ne00 127) 128))])
           (list nsg 2 (* 32 4 2) (if (zero? (remainder ne00 4)) "_4" ""))))]
    [else
     (let ([row (assoc ty '(("q4_0" 2 4 0) ("q4_1" 2 4 0) ("q5_0" 2 4 0) ("q5_1" 2 4 0)
                            ("q8_0" 4 2 256) ("mxfp4" 2 2 128)
                            ("q2_K" 2 4 0) ("q3_K" 2 2 0) ("q4_K" 2 2 0) ("q5_K" 2 1 0) ("q6_K" 2 2 0)
                            ("iq4_nl" 2 2 128) ("iq4_xs" 2 2 128)))])
       (if row
           (list (cadr row) (caddr row) (cadddr row) "")
           (error 'mul-mat "no matrix-vector kernel for this src0 type" ty)))]))

;; Small-batch matrix-matrix (2..8 columns of src1) through the "ext" kernels.
(define (nat-mul-mv-ext s0 s1 dst)
  (let* ([ty0 (nat-ty s0)] [ty1 (nat-ty s1)]
         [ne00 (nat-ne s0 0)] [ne01 (nat-ne s0 1)] [ne02 (nat-ne s0 2)] [ne03 (nat-ne s0 3)]
         [ne10 (nat-ne s1 0)] [ne11 (nat-ne s1 1)] [ne12 (nat-ne s1 2)] [ne13 (nat-ne s1 3)]
         [r2 (quotient ne12 ne02)] [r3 (quotient ne13 ne03)]
         [nsg 2]
         [nxpsg (cond [(and (zero? (remainder ne00 256)) (< ne11 3)) 16]
                      [(zero? (remainder ne00 128)) 8]
                      [else 4])]
         [nypsg (quotient 32 nxpsg)]
         [r0ptg (* nypsg nsg)]
         [r1ptg (case ne11
                  [(2) 2] [(3 6) 3] [(4 7 8) 4] [(5) 5]
                  [else (error 'mul-mat "unsupported ne11 for the small-batch kernel" ne11)])])
    (list
     (list (format "kernel_mul_mv_ext_~a_~a_r1_~a" ty0 ty1 r1ptg)
           (list (list nat-fc-mul-mv 'i16 nsg)
                 (list (+ nat-fc-mul-mv 1) 'i16 nxpsg)
                 (list (+ nat-fc-mul-mv 2) 'i16 ne12)
                 (list (+ nat-fc-mul-mv 3) 'i16 r2)
                 (list (+ nat-fc-mul-mv 4) 'i16 r3))
           (list (cons* 0 'bytes "mul_mv_ext"
                        (list (nat-i32 ne00) (nat-i32 ne01) (nat-i32 ne02)
                              (nat-u64 (nat-nb s0 0)) (nat-u64 (nat-nb s0 1)) (nat-u64 (nat-nb s0 2)) (nat-u64 (nat-nb s0 3))
                              (nat-i32 ne10) (nat-i32 ne11) (nat-i32 ne12)
                              (nat-u64 (nat-nb s1 0)) (nat-u64 (nat-nb s1 1)) (nat-u64 (nat-nb s1 2)) (nat-u64 (nat-nb s1 3))
                              (nat-i32 (nat-ne dst 0)) (nat-i32 (nat-ne dst 1))
                              (nat-i16 r2) (nat-i16 r3)))
                 '(1 tensor src0) '(2 tensor src1) '(3 tensor dst))
           (list (quotient (+ ne01 r0ptg -1) r0ptg) (quotient (+ ne11 r1ptg -1) r1ptg) (* ne12 ne13))
           (list 32 nsg 1)
           0))))

;; Matrix-matrix with simdgroup matrices (the 64 x 32 tile kernel).
(define (nat-mul-mm s0 s1 dst)
  (let* ([ty0 (nat-ty s0)] [ty1 (nat-ty s1)]
         [ne00 (nat-ne s0 0)] [ne01 (nat-ne s0 1)] [ne02 (nat-ne s0 2)] [ne03 (nat-ne s0 3)]
         [ne11 (nat-ne s1 1)] [ne12 (nat-ne s1 2)] [ne13 (nat-ne s1 3)]
         [r2 (quotient ne12 ne02)] [r3 (quotient ne13 ne03)]
         [ne0 (nat-ne dst 0)] [ne1 (nat-ne dst 1)]
         [bc-inp (not (zero? (remainder ne00 32)))]
         [bc-out (or (not (zero? (remainder ne0 64))) (not (zero? (remainder ne1 32))))]
         [nr0 64] [nr1 32] [nsg 4]
         [smem (if bc-out 8192 (+ 4096 2048))])
    (list
     (list (format "kernel_mul_mm_~a_~a" ty0 ty1)
           (list (list nat-fc-mul-mm 'bool bc-inp)
                 (list (+ nat-fc-mul-mm 1) 'bool bc-out)
                 (list (+ nat-fc-mul-mm 2) 'i16 ne12)
                 (list (+ nat-fc-mul-mm 3) 'i16 ne13)
                 (list (+ nat-fc-mul-mm 4) 'i16 r2)
                 (list (+ nat-fc-mul-mm 5) 'i16 r3))
           (list (cons* 0 'bytes "mul_mm"
                        (list (nat-i32 ne00) (nat-i32 ne02)
                              (nat-u64 (nat-nb s0 1)) (nat-u64 (nat-nb s0 2)) (nat-u64 (nat-nb s0 3))
                              (nat-i32 ne12)
                              (nat-u64 (nat-nb s1 0)) (nat-u64 (nat-nb s1 1)) (nat-u64 (nat-nb s1 2)) (nat-u64 (nat-nb s1 3))
                              (nat-i32 ne0) (nat-i32 ne1)
                              (nat-i16 r2) (nat-i16 r3)))
                 '(1 tensor src0) '(2 tensor src1) '(3 tensor dst))
           (list (quotient (+ ne11 nr1 -1) nr1) (quotient (+ ne01 nr0 -1) nr0) (* ne12 ne13))
           (list 32 nsg 1)
           smem))))

;; Matrix-vector kernels, also the fallback for transposed operands.
(define (nat-mul-mv s0 s1 dst)
  (let* ([ty0 (nat-ty s0)] [ty1 (nat-ty s1)]
         [ne00 (nat-ne s0 0)] [ne01 (nat-ne s0 1)] [ne02 (nat-ne s0 2)] [ne03 (nat-ne s0 3)]
         [ne10 (nat-ne s1 0)] [ne11 (nat-ne s1 1)] [ne12 (nat-ne s1 2)] [ne13 (nat-ne s1 3)]
         [r2 (quotient ne12 ne02)] [r3 (quotient ne13 ne03)]
         [shape (nat-mv-shape ty0 ne00)]
         [nsg (car shape)] [nr0 (cadr shape)] [smem (caddr shape)] [suffix (cadddr shape)]
         [nr1 1]
         [rows-per-group (if (member ty0 '("f32" "f16" "bf16" "q8_0")) nr0 (* nr0 nsg))])
    (list
     (list (format "kernel_mul_mv_~a_~a~a" ty0 ty1 suffix)
           (list (list nat-fc-mul-mv 'i16 nsg)
                 (list (+ nat-fc-mul-mv 2) 'i16 ne12)
                 (list (+ nat-fc-mul-mv 3) 'i16 r2)
                 (list (+ nat-fc-mul-mv 4) 'i16 r3))
           (list (cons* 0 'bytes "mul_mv"
                        (list (nat-i32 ne00) (nat-i32 ne01) (nat-i32 ne02)
                              (nat-u64 (nat-nb s0 0)) (nat-u64 (nat-nb s0 1)) (nat-u64 (nat-nb s0 2)) (nat-u64 (nat-nb s0 3))
                              (nat-i32 ne10) (nat-i32 ne11) (nat-i32 ne12)
                              (nat-u64 (nat-nb s1 0)) (nat-u64 (nat-nb s1 1)) (nat-u64 (nat-nb s1 2)) (nat-u64 (nat-nb s1 3))
                              (nat-i32 (nat-ne dst 0)) (nat-i32 (nat-ne dst 1))
                              (nat-i32 nr0)
                              (nat-i16 r2) (nat-i16 r3)))
                 '(1 tensor src0) '(2 tensor src1) '(3 tensor dst))
           (list (quotient (+ ne01 rows-per-group -1) rows-per-group) (quotient (+ ne11 nr1 -1) nr1) (* ne12 ne13))
           (list 32 nsg 1)
           smem))))

(define nat-ext-types
  '("f32" "f16" "bf16" "q1_0" "q2_0" "q4_0" "q4_1" "q5_0" "q5_1" "q8_0" "mxfp4" "iq4_nl"))
(define nat-ext-k-types '("q4_K" "q5_K" "q6_K" "q2_K" "q3_K"))

;; Entry point called from Rust. Returns the dispatches for one MUL_MAT.
(define ($tl-native-lower-mul-mat s0 s1 dst props)
  (let ([ty0 (nat-ty s0)] [ty1 (nat-ty s1)]
        [ne00 (nat-ne s0 0)] [ne02 (nat-ne s0 2)] [ne03 (nat-ne s0 3)]
        [ne10 (nat-ne s1 0)] [ne11 (nat-ne s1 1)] [ne12 (nat-ne s1 2)] [ne13 (nat-ne s1 3)]
        [simdgroup-mm? (car props)])
    (unless (= ne00 ne10)
      (error 'mul-mat "the inner dimensions differ" ne00 ne10))
    (unless (and (zero? (remainder ne12 ne02)) (zero? (remainder ne13 ne03)))
      (error 'mul-mat "src1 batch dimensions must be multiples of src0's" (list ne02 ne03) (list ne12 ne13)))
    (cond
      [(and (string=? ty1 "f32")
            (zero? (remainder ne00 128))
            (or (and (member ty0 nat-ext-types) (<= 2 ne11 8))
                (and (member ty0 nat-ext-k-types) (<= 4 ne11 8))))
       (nat-mul-mv-ext s0 s1 dst)]
      [(and (not (nat-transposed? s0)) (not (nat-transposed? s1))
            simdgroup-mm? (>= ne00 64) (> ne11 8))
       (nat-mul-mm s0 s1 dst)]
      [else (nat-mul-mv s0 s1 dst)])))
