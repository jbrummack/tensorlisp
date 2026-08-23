(let
  (
    (ctx get-global-ctx)
    (input
        (create-random-tensor (1 3 224 224)
    )
    (conv1
        (get-tensor ctx "conv1")
    )
    (tensor-add
        (fn
            (a b)
            (ggml_add ctx a b)
        )
    )
  )
  (tensor-add conv1 input)
  )
)
