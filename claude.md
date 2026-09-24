# tensorlisp
tensorlisp
## info
chezscheme is loaded with `git clone https://github.com/cisco/ChezScheme.git --filter=blob:none` to avoid loading unneccessary loaders

## lisp
I evaluated several LISPs, even tried implementing a lisp by hand but i tried already known lisps and settled on ChezScheme.

## other file formats
i looked at other file formats/backends where support may later be added.

## layout
Cargo workspace, see README.md. `crates/ggml-sys` builds vendored ggml (`vendor/ggml`) + bindgen into OUT_DIR; `crates/tensorlisp` is the core (gguf loading, later the Chez host); `crates/importers` has onnx/coreml readers; `experiments/handrolled-lisp` is the archived hand-written lisp attempts.
ChezScheme is the submodule `vendor/ChezScheme`, built with `scripts/build-chez.sh` into `vendor/chez` (gitignored).
