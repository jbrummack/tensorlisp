use lasso::Spur;

use crate::{
    ops::{Math, Op},
    runtime::memory::FreeMap,
};
pub mod machine;
pub mod memory;
pub mod string;
//8 or 16 fit into one cache line

pub struct LargeInt([u8; 7]);
impl From<u64> for LargeInt {
    fn from(value: u64) -> Self {
        let [a, b, c, d, e, f, g, _] = value.to_ne_bytes();
        Self([a, b, c, d, e, f, g])
    }
}
impl From<LargeInt> for u64 {
    fn from(value: LargeInt) -> Self {
        let [a, b, c, d, e, f, g] = value.0;
        Self::from_ne_bytes([a, b, c, d, e, f, g, 0])
    }
}
impl From<i64> for LargeInt {
    fn from(value: i64) -> Self {
        let [a, b, c, d, e, f, g, _] = value.to_ne_bytes();
        Self([a, b, c, d, e, f, g])
    }
}
impl From<LargeInt> for i64 {
    fn from(value: LargeInt) -> Self {
        let [a, b, c, d, e, f, g] = value.0;
        Self::from_ne_bytes([a, b, c, d, e, f, g, 0])
    }
}
pub enum Expr {
    Nil,
    Medium { cell: u32, cell_offset: u8 },
    Atom(Spur),
    String(u32),
    Bool(bool),
    Float(f32),
    Int(LargeInt),
    Context(LargeInt), //*mut ggml_ctx
    Tensor(LargeInt),  //*mut ggml_tensor
    Op(Op),
}
pub struct Runtime {
    pub freemap: FreeMap,
    pub mediums: Vec<[Expr; 4]>,
}
impl Runtime {}
