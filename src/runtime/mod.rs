use lasso::Spur;

use crate::runtime::memory::MediumList;

pub mod memory;
pub mod string;
pub enum Expr {
    Float(f32),
    Int(i32),
    Medium(u32),
    Atom(Spur),
    String(u32),
}

pub struct Runtime {
    _mediums: Vec<MediumList<Expr>>,
}
