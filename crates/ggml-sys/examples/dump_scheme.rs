//! Prints the generated Scheme bindings: `cargo run -p ggml-sys --example dump_scheme`.
use ggml_sys::ffi::scheme;

fn main() {
    println!(";; raw\n{}\n;; ops\n{}\n;; constants\n{}", scheme::RAW, scheme::OPS, scheme::CONSTANTS);
}
