;; (tl tree opt) -- Tree-to-Tree optimization passes and the pass manager.
;; docs/design/tree.md ("Passes").
;;
;; A kernel is data, so a pass is a function from a kernel to a kernel. The compilers translate
;; whatever form they are given in one go; everything that makes the generated code better than
;; the straightforward translation is done here, before them:
;;
;;   fold  constant folding and algebraic identities, dead `when`/`if` arms
;;   licm  loop-invariant arithmetic is hoisted out of `for`
;;   cse   a pure expression computed more than once is bound once (and reused in nested loops)
;;   dce   unused `let` / `var` bindings (and the `set!`s of a var nobody reads) are dropped
;;
;; Passes are registered by name, so a user (or an agent) adds their own:
;;
;;   (register-pass! 'my-pass (lambda (kernel) ... kernel))
;;   (optimize kernel '(fold my-pass dce))     ; default: (fold licm cse dce)
;;
;; and can build them from the traversals exported here: `map-stmt` (rebuild a statement with a
;; function applied to its own expressions and to its nested statement lists), `expr-map`
;; (bottom-up rewrite of an expression), `mentions`, `assigned-vars`, `binders`, `pure?`.
;;
;; What counts as pure: arithmetic, comparisons, casts, math functions, select/guard, the thread
;; builtins, and loads from `ro` buffers (a launch never writes a buffer it declares `ro`).
;; Subgroup operations are not (they depend on which lanes reach them), nor are loads from
;; writable buffers or shared memory (stores and barriers change them).
(library (tl tree opt)
  (export optimize register-pass! pass-names run-pass default-passes
          map-stmt expr-map mentions assigned-vars binders pure? op-count
          pass-fold pass-licm pass-cse pass-dce)
  (import (rnrs) (rnrs hashtables) (only (chezscheme) format))

  ;; ---- traversal ------------------------------------------------------------------------------

  (define clause-heads '(workgroup buffers push shared offsets))
  (define (clause? f) (and (pair? f) (memq (car f) clause-heads) #t))

  ;; Names of the `ro` buffers of a kernel (set while a pass runs).
  (define current-ro '())
  (define counter 0)
  (define (fresh prefix)
    (set! counter (+ counter 1))
    (string->symbol (string-append prefix (number->string counter))))

  (define (kernel-ro k)
    (let loop ([fs (cddr k)] [acc '()])
      (cond [(null? fs) acc]
            [(and (pair? (car fs)) (eq? (caar fs) 'buffers))
             (loop (cdr fs) (append (map car (filter (lambda (b) (eq? (cadr b) 'ro)) (cdar fs))) acc))]
            [else (loop (cdr fs) acc)])))

  ;; Applies (f body) to the statements of a kernel, with the `ro` set of the kernel visible.
  (define (map-body f k)
    (let ([clauses (filter clause? (cddr k))] [body (filter (lambda (x) (not (clause? x))) (cddr k))])
      (set! current-ro (kernel-ro k))
      `(kernel ,(cadr k) ,@clauses ,@(f body))))

  (define (single stmts)
    (cond [(null? stmts) '(begin)]
          [(null? (cdr stmts)) (car stmts)]
          [else `(begin ,@stmts)]))

  ;; map in order (R6RS leaves the order unspecified; the passes below keep state)
  (define (omap f l)
    (if (null? l) '() (let ([x (f (car l))]) (cons x (omap f (cdr l))))))

  ;; Rebuilds statement `s`: `fe` is applied to each expression of the statement itself (in order),
  ;; then `fb` to each nested list of statements (and returns a list of statements).
  (define (map-stmt s fe fb)
    (case (car s)
      [(let) (let* ([bs (omap (lambda (b) (list (car b) (fe (cadr b)))) (cadr s))] [body (fb (cddr s))])
               `(let ,bs ,@body))]
      [(var) `(var ,(cadr s) ,(caddr s) ,(fe (cadddr s)))]
      [(set!) `(set! ,(cadr s) ,(fe (caddr s)))]
      [(store) (let* ([i (fe (caddr s))] [v (fe (cadddr s))]) `(store ,(cadr s) ,i ,v))]
      [(vload) `(vload ,(cadr s) ,(caddr s) ,(fe (cadddr s)))]
      [(begin) `(begin ,@(fb (cdr s)))]
      [(when unless) (let* ([c (fe (cadr s))] [body (fb (cddr s))]) `(,(car s) ,c ,@body))]
      [(if) (let* ([c (fe (cadr s))] [arms (omap (lambda (x) (single (fb (list x)))) (cddr s))]) `(if ,c ,@arms))]
      [(for) (let* ([spec (cadr s)] [bounds (omap fe (cdr spec))] [body (fb (cddr s))])
               `(for ,(cons (car spec) bounds) ,@body))]
      [else s]))

  ;; Bottom-up rewrite of an expression: children first, then (f node).
  (define (expr-map f e)
    (cond
      [(not (pair? e)) e]
      [else
       (f (case (car e)
            [(load) (list 'load (cadr e) (expr-map f (caddr e)))]
            [(let) `(let ,(map (lambda (b) (list (car b) (expr-map f (cadr b)))) (cadr e)) ,(expr-map f (caddr e)))]
            [else (cons (car e) (map (lambda (x) (expr-map f x)) (cdr e)))]))]))

  ;; Symbols in value positions of an expression (let-bound names included).
  (define (mentions e)
    (cond [(symbol? e) (list e)]
          [(not (pair? e)) '()]
          [(eq? (car e) 'load) (mentions (caddr e))]
          [(eq? (car e) 'let)
           (append (apply append (map (lambda (b) (cons (car b) (mentions (cadr b)))) (cadr e))) (mentions (caddr e)))]
          [else (apply append (map mentions (cdr e)))]))

  (define (disjoint? a b) (not (exists (lambda (x) (memq x b)) a)))

  ;; Every var assigned (set!) anywhere in the statements.
  (define (assigned-vars stmts)
    (let ([acc '()])
      (define (walk s)
        (when (eq? (car s) 'set!) (set! acc (cons (cadr s) acc)))
        (map-stmt s (lambda (e) e) (lambda (b) (for-each walk b) b)))
      (for-each walk stmts)
      acc))

  ;; Every name bound anywhere in the statements (let, var, vload, loop variables).
  (define (binders stmts)
    (let ([acc '()])
      (define (walk s)
        (case (car s)
          [(let) (set! acc (append (map car (cadr s)) acc))]
          [(var) (set! acc (cons (cadr s) acc))]
          [(vload) (set! acc (append (cadr s) acc))]
          [(for) (set! acc (cons (car (cadr s)) acc))]
          [else #f])
        (map-stmt s (lambda (e) e) (lambda (b) (for-each walk b) b)))
      (for-each walk stmts)
      acc))

  ;; ---- purity and cost ----------------------------------------------------------------------------

  (define pure-heads
    '(+ - * / min max rem pow < <= > >= = != and or not select guard f32 i32 u32 u16
      exp log sqrt inv-sqrt tanh sin cos floor ceil abs fma half->f32 f32->half dp4a sbits ubits
      bit-and bit-or bit-xor shl shr
      local-id group-id global-id num-groups lane subgroup-id subgroup-size num-subgroups))
  (define cheap-heads '(local-id group-id global-id num-groups lane subgroup-id subgroup-size num-subgroups))
  (define expensive-heads '(/ rem * exp log sin cos tanh pow sqrt inv-sqrt half->f32 f32->half dp4a sbits ubits))

  (define (pure? e)
    (cond [(not (pair? e)) #t]
          [(eq? (car e) 'load) (and (memq (cadr e) current-ro) (pure? (caddr e)))]
          [(memq (car e) pure-heads) (for-all pure? (cdr e))]
          [else #f]))

  ;; Operations in an expression (leaves and thread builtins are free).
  (define (op-count e)
    (cond [(not (pair? e)) 0]
          [(memq (car e) cheap-heads) 0]
          [(eq? (car e) 'load) (+ 1 (op-count (caddr e)))]
          [else (+ 1 (apply + (map op-count (cdr e))))]))

  (define (has-load? e)
    (and (pair? e) (or (eq? (car e) 'load) (exists has-load? (cdr e)))))

  ;; The expression has a type of its own (a name, a float, a cast, ...), as opposed to a bare
  ;; integer literal that takes the type of its context; only those can be bound to a name.
  (define (typed? e)
    (cond [(symbol? e) #t]
          [(boolean? e) #t]
          [(number? e) (inexact? e)]
          [(not (pair? e)) #f]
          [else
           (case (car e)
             [(+ - * / min max rem pow bit-and bit-or bit-xor shl shr abs) (exists typed? (cdr e))]
             [(select guard) (exists typed? (cddr e))]
             [(shuffle-xor subgroup-add subgroup-max subgroup-min) (typed? (cadr e))]
             [else #t])]))

  ;; Worth giving a name: pure, typed, and more than a single cheap operation.
  (define (candidate? e)
    (and (pair? e)
         (typed? e)
         (not (memq (car e) cheap-heads))
         (not (eq? (car e) 'let))
         (pure? e)
         (or (>= (op-count e) 2) (memq (car e) expensive-heads) (has-load? e))))

  ;; ---- fold ---------------------------------------------------------------------------------------

  (define (int? x) (and (number? x) (exact? x) (integer? x)))
  (define (lit? x) (and (number? x) (real? x)))
  (define (fin? x) (and (real? x) (= x x) (not (= x +inf.0)) (not (= x -inf.0))))
  (define (all-lit? l) (for-all lit? l))
  (define (any-flo? l) (exists (lambda (x) (and (number? x) (inexact? x))) l))
  (define (to-float x) (if (exact? x) (inexact x) x))
  (define (nonneg-ints? l) (and (for-all int? l) (for-all (lambda (x) (>= x 0)) l)))

  (define (fold-literals h args)
    (let ([vals (if (any-flo? args) (map to-float args) args)])
      (let ([r (case h
                 [(+) (apply + vals)]
                 [(*) (apply * vals)]
                 [(-) (apply - vals)]
                 [(/) (cond [(any-flo? args) (and (not (zero? (cadr vals))) (apply / vals))]
                            [(and (for-all int? vals) (pair? (cdr vals)) (null? (cddr vals)) (not (zero? (cadr vals))) (zero? (mod (car vals) (cadr vals))))
                             (div (car vals) (cadr vals))]
                            [else #f])]
                 [(rem) (and (nonneg-ints? vals) (pair? (cdr vals)) (positive? (cadr vals)) (mod (car vals) (cadr vals)))]
                 [(min) (apply min vals)]
                 [(max) (apply max vals)]
                 [else #f])])
        (and r (fin? r) r))))

  (define (simplify e)
    (let ([h (car e)] [args (cdr e)])
      (define (same) e)
      (case h
        [(+ - * / rem min max)
         (cond
           [(and (all-lit? args) (pair? args) (fold-literals h args)) => (lambda (v) v)]
           [(and (eq? h '+) (exists (lambda (a) (eqv? a 0)) args) (pair? (cdr args)))
            (let ([rest (filter (lambda (a) (not (eqv? a 0))) args)])
              (cond [(null? rest) 0] [(null? (cdr rest)) (car rest)] [else (cons '+ rest)]))]
           [(and (eq? h '*) (exists (lambda (a) (eqv? a 1)) args) (pair? (cdr args)))
            (let ([rest (filter (lambda (a) (not (eqv? a 1))) args)])
              (cond [(null? rest) 1] [(null? (cdr rest)) (car rest)] [else (cons '* rest)]))]
           [(and (memq h '(- /)) (pair? (cdr args)) (null? (cddr args)) (eqv? (cadr args) (if (eq? h '-) 0 1))) (car args)]
           [else e])]
        [(and)
         (cond [(memq #f args) #f]
               [else (let ([rest (filter (lambda (a) (not (eq? a #t))) args)])
                       (cond [(null? rest) #t] [(null? (cdr rest)) (car rest)] [else (cons 'and rest)]))])]
        [(or)
         (cond [(memq #t args) #t]
               [else (let ([rest (filter (lambda (a) (not (eq? a #f))) args)])
                       (cond [(null? rest) #f] [(null? (cdr rest)) (car rest)] [else (cons 'or rest)]))])]
        [(not)
         (let ([a (car args)])
           (cond [(boolean? a) (not a)]
                 [(and (pair? a) (eq? (car a) 'not)) (cadr a)]
                 [else e]))]
        [(< <= > >= = !=)
         (if (and (= (length args) 2) (all-lit? args))
             (let ([a (car args)] [b (cadr args)])
               (case h [(<) (< a b)] [(<=) (<= a b)] [(>) (> a b)] [(>=) (>= a b)] [(=) (= a b)] [else (not (= a b))]))
             e)]
        [(select)
         (cond [(eq? (car args) #t) (cadr args)]
               [(eq? (car args) #f) (caddr args)]
               [(equal? (cadr args) (caddr args)) (cadr args)]
               [else e])]
        [(shl shr bit-or bit-xor)
         (if (and (= (length args) 2) (eqv? (cadr args) 0)) (car args) e)]
        [else e])))

  (define (fold-stmts stmts)
    (let loop ([ss stmts] [acc '()])
      (if (null? ss)
          (reverse acc)
          (let ([s (fold-stmt (car ss))])
            (loop (cdr ss)
                  (cond [(and (pair? s) (eq? (car s) 'begin) (null? (cdr s))) acc]
                        [else (cons s acc)]))))))

  (define (fold-stmt s)
    (let ([s (map-stmt s (lambda (e) (expr-map simplify e)) fold-stmts)])
      (case (car s)
        [(when) (cond [(eq? (cadr s) #t) `(begin ,@(cddr s))]
                      [(eq? (cadr s) #f) '(begin)]
                      [else s])]
        [(unless) (cond [(eq? (cadr s) #f) `(begin ,@(cddr s))]
                        [(eq? (cadr s) #t) '(begin)]
                        [else s])]
        [(if) (cond [(eq? (cadr s) #t) (caddr s)]
                    [(eq? (cadr s) #f) (if (pair? (cdddr s)) (cadddr s) '(begin))]
                    [else s])]
        [else s])))

  (define (pass-fold k) (map-body fold-stmts k))

  ;; ---- licm ---------------------------------------------------------------------------------------

  ;; Hoists maximal loop-invariant candidates (without loads: a load may sit behind a bounds
  ;; check the loop body makes) out of each `for`.
  (define (licm-stmts stmts) (map licm-stmt stmts))

  (define (licm-stmt s)
    (if (eq? (car s) 'for)
        (licm-for s)
        (map-stmt s (lambda (e) e) licm-stmts)))

  (define (licm-for s)
    (let* ([body (cddr s)]
           [inner (cons (car (cadr s)) (append (binders body) (assigned-vars body)))]
           [hoisted '()]
           [table (make-hashtable equal-hash equal?)])
      (define (invariant? e) (and (candidate? e) (not (has-load? e)) (disjoint? (mentions e) inner)))
      (define (rewrite e)
        (cond
          [(not (pair? e)) e]
          [(invariant? e)
           (let ([n (hashtable-ref table e #f)])
             (or n
                 (let ([n (fresh "$h")])
                   (hashtable-set! table e n)
                   (set! hoisted (cons (list n e) hoisted))
                   n)))]
          [(eq? (car e) 'load) (list 'load (cadr e) (rewrite (caddr e)))]
          [(eq? (car e) 'let) e]
          [(eq? (car e) 'guard) (list 'guard (rewrite (cadr e)) (caddr e) (rewrite (cadddr e)))]
          [else (cons (car e) (map rewrite (cdr e)))]))
      (define (stmts-of b) (map (lambda (x) (rewrite-stmt x)) b))
      (define (rewrite-stmt x) (map-stmt x rewrite stmts-of))
      (let* ([new-body (stmts-of body)]
             [new-body (licm-stmts new-body)]
             ;; the bounds are not rewritten: they are evaluated outside / per iteration as written
             [loop `(for ,(cadr s) ,@new-body)])
        (if (null? hoisted)
            loop
            `(let ,(reverse hoisted) ,loop)))))

  (define (pass-licm k) (map-body licm-stmts k))

  ;; ---- cse ----------------------------------------------------------------------------------------

  ;; Occurrence counts of the candidates of a statement list (everywhere, nested bodies included).
  (define (count-candidates stmts)
    (let ([t (make-hashtable equal-hash equal?)])
      (define (count e)
        (when (pair? e)
          (when (candidate? e) (hashtable-update! t e (lambda (n) (+ n 1)) 0))
          (case (car e)
            [(load) (count (caddr e))]
            [(let) (for-each (lambda (b) (count (cadr b))) (cadr e)) (count (caddr e))]
            [else (for-each count (cdr e))])))
      (define (walk s)
        (map-stmt s (lambda (e) (count e) e) (lambda (b) (for-each walk b) b)))
      (for-each walk stmts)
      t))

  (define (kill avail names)
    (filter (lambda (entry) (disjoint? (mentions (car entry)) names)) avail))

  (define (assoc-equal key alist)
    (let loop ([l alist])
      (cond [(null? l) #f] [(equal? (caar l) key) (car l)] [else (loop (cdr l))])))

  ;; (let ([a e1] [b e2]) body ...) as nested single-binding lets, so that a shared subexpression
  ;; of e2 may be bound between the two.
  (define (split-let s)
    (if (and (eq? (car s) 'let) (pair? (cadr s)) (pair? (cdr (cadr s))))
        `(let (,(car (cadr s))) ,(split-let `(let ,(cdr (cadr s)) ,@(cddr s))))
        s))

  (define (cse-stmts stmts)
    (let ([counts (count-candidates stmts)])
      (define (cse-block stmts avail)
        (if (null? stmts)
            '()
            (let* ([s (split-let (car stmts))]
                   [pending '()]
                   [avail* avail])
              ;; rewrites the expression `e` of this statement; `bind?` = may name new shared values
              (define (rewrite e bind?)
                (cond
                  [(not (pair? e)) e]
                  [(candidate? e)
                   (cond
                     [(assoc-equal e avail*) => cdr]
                     [(and bind? (>= (hashtable-ref counts e 0) 2))
                      (let* ([e2 (children e bind?)] [n (fresh "$c")])
                        (set! pending (cons (list n e2) pending))
                        (set! avail* (cons (cons e n) avail*))
                        n)]
                     [else (children e bind?)])]
                  [else (children e bind?)]))
              (define (children e bind?)
                (case (car e)
                  [(load) (list 'load (cadr e) (rewrite (caddr e) bind?))]
                  [(let) e]
                  [(guard) (list 'guard (rewrite (cadr e) bind?) (rewrite (caddr e) #f) (rewrite (cadddr e) bind?))]
                  [else (cons (car e) (omap (lambda (x) (rewrite x bind?)) (cdr e)))]))
              (define (nested body shadow)
                (cse-block body (kill avail* shadow)))
              (let* ([bound-here
                      (case (car s)
                        [(var) (list (cadr s))]
                        [(vload) (cadr s)]
                        [else '()])]
                     [s*
                      (case (car s)
                        [(for)
                         (let* ([body (cddr s)]
                                [shadow (cons (car (cadr s)) (append (binders body) (assigned-vars body)))]
                                [bounds (cdr (cadr s))])
                           ;; bounds: reuse only; the body sees nothing that depends on what it changes
                           (let ([spec (cons (car (cadr s)) (map (lambda (b) (rewrite b #f)) bounds))])
                             `(for ,spec ,@(nested body shadow))))]
                        [(let)
                         (if (null? (cadr s))
                             `(let () ,@(nested (cddr s) '()))
                             (let* ([b (car (cadr s))] [e2 (rewrite (cadr b) #t)])
                           `(let ((,(car b) ,e2)) ,@(nested (cddr s) (list (car b))))))]
                        [(if)
                         (let ([c (rewrite (cadr s) #t)])
                           `(if ,c ,@(map (lambda (x) (single (nested (list x) '()))) (cddr s))))]
                        [else
                         (map-stmt s (lambda (e) (rewrite e #t)) (lambda (b) (nested b '())))])]
                     [after (kill avail* (append bound-here (if (eq? (car s) 'set!) (list (cadr s)) '()) (assigned-vars (list s))))]
                     [rest (cse-block (cdr stmts) after)])
                (if (null? pending)
                    (cons s* rest)
                    (list `(let ,(reverse pending) ,s* ,@rest)))))))
      (cse-block stmts '())))

;; ---- inline ------------------------------------------------------------------------------------

  ;; CSE names a value for every candidate that occurs twice somewhere, including where an
  ;; assignment makes the second occurrence a different value; the names used only once are put
  ;; back (unless the use sits in a deeper loop than the binding: that is the point of a name).
  (define (cse-name? n)
    (and (symbol? n)
         (let ([s (symbol->string n)]) (and (> (string-length s) 2) (string=? (substring s 0 2) "$c")))))

  (define (inline-single-use stmts)
    (let ([uses (make-eq-hashtable)] [depth-of-use (make-eq-hashtable)] [env (make-eq-hashtable)])
      ;; walks statements calling (fe expr depth) on every expression
      (define (walk stmts depth fe)
        (for-each
          (lambda (s)
            (let ([d (if (eq? (car s) 'for) (+ depth 1) depth)])
              (map-stmt s (lambda (e) (fe e d) e) (lambda (b) (walk b d fe) b))))
          stmts))
      (define counted
        (walk stmts 0
            (lambda (e d)
              (for-each (lambda (x)
                          (when (cse-name? x)
                            (hashtable-update! uses x (lambda (n) (+ n 1)) 0)
                            (hashtable-set! depth-of-use x d)))
                        (mentions e)))))
      (define (subst e)
        (cond [(symbol? e) (hashtable-ref env e e)]
              [(not (pair? e)) e]
              [(eq? (car e) 'load) (list 'load (cadr e) (subst (caddr e)))]
              [(eq? (car e) 'let) `(let ,(map (lambda (b) (list (car b) (subst (cadr b)))) (cadr e)) ,(subst (caddr e)))]
              [else (cons (car e) (map subst (cdr e)))]))
      (define (go stmts depth)
        (map (lambda (s)
               (let ([d (if (eq? (car s) 'for) (+ depth 1) depth)])
                 (if (eq? (car s) 'let)
                     (let loop ([bs (cadr s)] [keep '()])
                       (if (null? bs)
                           `(let ,(reverse keep) ,@(go (cddr s) depth))
                           (let* ([n (caar bs)] [e (subst (cadar bs))])
                             (cond
                               [(and (cse-name? n) (= (hashtable-ref uses n 0) 0)) (loop (cdr bs) keep)]
                               [(and (cse-name? n) (= (hashtable-ref uses n 0) 1) (= (hashtable-ref depth-of-use n -1) depth))
                                (hashtable-set! env n e)
                                (loop (cdr bs) keep)]
                               [else (loop (cdr bs) (cons (list n e) keep))]))))
                     (map-stmt s subst (lambda (b) (go b d))))))
             stmts))
      (go stmts 0)))

  (define (pass-cse k) (map-body (lambda (stmts) (inline-single-use (cse-stmts stmts))) k))

  ;; ---- dce ----------------------------------------------------------------------------------------

  ;; Symbols read anywhere (the targets of set! and the names a binding introduces are not reads).
  (define (reads stmts)
    (let ([t (make-eq-hashtable)])
      (define (note e) (for-each (lambda (x) (hashtable-set! t x #t)) (mentions e)))
      (define (walk s)
        (when (and (eq? (car s) 'set!) (not (pure? (caddr s)))) (hashtable-set! t (cadr s) #t))
        (map-stmt s (lambda (e) (note e) e) (lambda (b) (for-each walk b) b)))
      (for-each walk stmts)
      t))

  (define (dce-once stmts)
    (let ([used (reads stmts)] [changed #f])
      (define (read? n) (hashtable-ref used n #f))
      (define (go stmts)
        (let loop ([ss stmts] [acc '()])
          (if (null? ss)
              (reverse acc)
              (let ([s (car ss)])
                (cond
                  [(and (eq? (car s) 'var) (not (read? (cadr s))) (pure? (cadddr s)))
                   (set! changed #t)
                   (loop (cdr ss) acc)]
                  [(and (eq? (car s) 'set!) (not (read? (cadr s))) (pure? (caddr s)))
                   (set! changed #t)
                   (loop (cdr ss) acc)]
                  [(eq? (car s) 'let)
                   (let* ([bs (cadr s)]
                          [keep (filter (lambda (b) (or (read? (car b)) (not (pure? (cadr b))))) bs)])
                     (unless (= (length keep) (length bs)) (set! changed #t))
                     (loop (cdr ss)
                           (cons (map-stmt `(let ,keep ,@(cddr s)) (lambda (e) e) go) acc)))]
                  [else (loop (cdr ss) (cons (map-stmt s (lambda (e) e) go) acc))])))))
      (let ([r (go stmts)]) (cons changed r))))

  (define (dce-stmts stmts)
    (let loop ([ss stmts] [n 0])
      (let ([r (dce-once ss)])
        (if (and (car r) (< n 6)) (loop (cdr r) (+ n 1)) (cdr r)))))

  (define (pass-dce k) (map-body dce-stmts k))

  ;; ---- pass manager ---------------------------------------------------------------------------------

  (define registry
    (list (cons 'fold pass-fold) (cons 'licm pass-licm) (cons 'cse pass-cse) (cons 'dce pass-dce)))

  (define default-passes '(fold licm cse dce))

  (define (register-pass! name proc)
    (set! registry (cons (cons name proc) (filter (lambda (e) (not (eq? (car e) name))) registry))))

  (define (pass-names) (map car registry))

  (define (run-pass name k)
    (let ([p (assq name registry)])
      (unless p (error 'optimize "no such pass" name))
      ((cdr p) k)))

  (define optimize
    (case-lambda
      [(k) (optimize k default-passes)]
      ;; kernels with cooperative matrices are left alone: the passes do not look inside cm-* statements
      [(k passes) (if (has-cmat? k) k (fold-left (lambda (k p) (run-pass p k)) k passes))]))

  (define (has-cmat? x)
    (cond [(pair? x) (or (eq? (car x) 'cmat) (has-cmat? (car x)) (has-cmat? (cdr x)))]
          [else #f]))
)
