//! Regenerates `crates/tensorlisp/src/scheme/aot/mil/language.ss` from coreml-rs's own
//! generated MIL op catalogue (`coreml_rs::mil::graph::MIL_GRAPH_OPS`/
//! `MIL_GRAPH_OP_NAMES`, produced by that crate's `build.rs` from
//! `spec/mil_ops.json`): coreml-rs already ports its MIL builder pattern to
//! Scheme itself (one `(define (mil-<op> ...) ...)` per op, building the
//! same `(<op> (<arg> <val>) ...)` S-expression its own Rust
//! `cg_leafnode::builder::Builder` would, as plain graph data -- see that
//! crate's `build.rs`'s `gen_graph_builders` doc comment), so this tool's
//! only job is wrapping that already-generated source in a proper `(tl aot
//! mil)` library (export clause, `mil-op-names`, a doc header) the same way
//! `ggml-codegen`'s generated `OPS`/`CONSTANTS` get spliced into
//! `scheme/core.ss`'s `library_source()`.
//!
//! Run whenever coreml-rs's op catalogue changes (a coremltools/spec bump):
//!
//! ```sh
//! cargo run -p tensorlisp-aot --example gen_mil_language \
//!   > crates/tensorlisp/src/scheme/aot/mil/language.ss
//! ```
use coreml_rs::mil::graph::{MIL_GRAPH_OP_NAMES, MIL_GRAPH_OPS};

/// Wraps `names` into lines of at most `width` columns, each already
/// prefixed with `indent`.
fn wrap(names: &[&str], width: usize, indent: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cur = String::new();
    for n in names {
        if !cur.is_empty() && indent.len() + cur.len() + 1 + n.len() > width {
            lines.push(format!("{indent}{cur}"));
            cur.clear();
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(n);
    }
    if !cur.is_empty() {
        lines.push(format!("{indent}{cur}"));
    }
    lines
}

fn main() {
    let names: Vec<&str> = MIL_GRAPH_OP_NAMES.split_whitespace().collect();
    let export_lines = wrap(&names, 78, "          ").join("\n");
    let names_lines = wrap(&names, 76, "     ").join("\n");

    println!(
        ";;; scheme/aot/mil/language.ss
;;;
;;; The CoreML MIL graph-builder vocabulary -- the first of the passes
;;; under scheme/aot/mil/ (see trace.ss, shadow.ss, compile.ss next to this
;;; file for the highlevel -> MIL nanopass built on top of it). Generated
;;; (see `crates/tensorlisp-aot/examples/gen_mil_language.rs`) from coreml-rs's own
;;; `coreml_rs::mil::graph::MIL_GRAPH_OPS` -- that crate already ports its MIL
;;; builder pattern to Scheme itself (its `build.rs`, from the same
;;; `spec/mil_ops.json` that generates its Rust `MilOp`/`OpSpec` catalog and its
;;; `cg_leafnode::builder::Builder`-based Rust graph functions), so this file is
;;; a thin, regeneratable wrapper, not a hand-written port.
;;;
;;; Each `(mil-<op> <arg> ...)` below takes the op's own declared arguments
;;; positionally (Internal-kind ones, e.g. a `cond`/`while_loop` block body,
;;; are Python-side plumbing and don't appear here) and returns the same
;;; `(<op> (<arg> <val>) ...)` S-expression coreml-rs's Rust
;;; `cg_leafnode::builder::Builder::op` would build for an identical call --
;;; plain graph data, no execution, no validation (real op semantics/type
;;; checking stay `coreml_rs::mil::builder`'s job once a pass hands it a
;;; concrete `Function`). This is `(tl aot highlevel)`'s MIL-side counterpart:
;;; a second nanopass *target* language, next to `reference_compiler.ss`'s
;;; ggml target -- see `core.ss`'s existing (hand-written, symbolic-trace) MIL
;;; pass for how tensorlisp already renders a MIL program from Scheme; a
;;; highlevel -> MIL pass over *this* vocabulary would build the same kind of
;;; IR node this library returns, one node per op, instead of a text trace.
;;;
;;; To regenerate after a coreml-rs op-catalogue change:
;;;   cargo run -p tensorlisp-aot --example gen_mil_language \\
;;;     > crates/tensorlisp/src/scheme/aot/mil/language.ss
(library (tl aot mil language)
  (export mil-op-names
{export_lines})
  (import (rnrs))

  (define mil-op-names
    '({names_lines}))

{ops})",
        ops = MIL_GRAPH_OPS.trim_end(),
    );
}
