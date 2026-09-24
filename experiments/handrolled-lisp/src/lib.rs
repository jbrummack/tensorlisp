//! Archived experiments from before settling on ChezScheme: several attempts at
//! a hand-written lisp (arena lists, hash-consed cells, a small VM). Kept for
//! reference only; nothing in the main crates depends on this.

pub mod hashcons;
pub mod input;
pub mod ops;
pub mod parser;
pub mod runtime;
pub mod stupid_lisp;
