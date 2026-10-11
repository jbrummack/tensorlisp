//! A small SPIR-V reflector: derives a kernel's [`Signature`] from its code,
//! so any blob (glslc, naga, a generated kernel) is schedulable without a
//! hand-written description. Reads descriptor bindings and their access
//! (`NonWritable` / `NonReadable` decorations), the size of the push-constant
//! block and the literal workgroup size.

use std::collections::HashMap;

use super::blob::{Access, Code, KernelBlob, OffsetMode, Signature, Target};
use crate::{Error, Result};

const MAGIC: u32 = 0x0723_0203;

// Opcodes.
const OP_ENTRY_POINT: u16 = 15;
const OP_EXECUTION_MODE: u16 = 16;
const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_MATRIX: u16 = 24;
const OP_TYPE_ARRAY: u16 = 28;
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;
const OP_CONSTANT: u16 = 43;
const OP_VARIABLE: u16 = 59;
const OP_DECORATE: u16 = 71;
const OP_MEMBER_DECORATE: u16 = 72;

// Decorations.
const DEC_BINDING: u32 = 33;
const DEC_DESCRIPTOR_SET: u32 = 34;
const DEC_OFFSET: u32 = 35;
const DEC_NON_WRITABLE: u32 = 24;
const DEC_NON_READABLE: u32 = 25;
const DEC_ARRAY_STRIDE: u32 = 6;

// Storage classes.
const SC_PUSH_CONSTANT: u32 = 9;
const SC_UNIFORM: u32 = 2;
const SC_STORAGE_BUFFER: u32 = 12;

const MODE_LOCAL_SIZE: u32 = 17;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reflection {
    /// One entry per binding `0..n` of set 0; unused gaps are `Read`.
    pub bindings: Vec<Access>,
    pub push_bytes: u32,
    /// Literal `LocalSize` of the entry point, `[0; 3]` when it comes from specialization constants.
    pub workgroup: [u32; 3],
    pub entry: String,
}

#[derive(Default)]
struct Module {
    /// Per result id: (decoration, first operand).
    decorations: HashMap<u32, Vec<(u32, u32)>>,
    /// (struct id, member) -> decorations.
    members: HashMap<(u32, u32), Vec<(u32, u32)>>,
    /// Result id -> (opcode, operands).
    types: HashMap<u32, (u16, Vec<u32>)>,
    constants: HashMap<u32, u32>,
    /// (variable id, pointer type id, storage class).
    variables: Vec<(u32, u32, u32)>,
    entry: Option<(u32, String)>,
    local_size: Option<[u32; 3]>,
}

fn parse(words: &[u32]) -> Result<Module> {
    if words.len() < 5 || words[0] != MAGIC {
        return Err(Error::Invalid("not a SPIR-V module (bad magic number)".into()));
    }
    let mut m = Module::default();
    let mut i = 5;
    while i < words.len() {
        let op = (words[i] & 0xffff) as u16;
        let n = (words[i] >> 16) as usize;
        if n == 0 || i + n > words.len() {
            return Err(Error::Invalid("malformed SPIR-V instruction".into()));
        }
        let a = &words[i + 1..i + n];
        match op {
            OP_ENTRY_POINT if m.entry.is_none() && a[0] == 5 => {
                // ExecutionModel GLCompute = 5; name is a NUL-terminated string after the id.
                let bytes: Vec<u8> = a[2..].iter().flat_map(|w| w.to_le_bytes()).take_while(|&b| b != 0).collect();
                m.entry = Some((a[1], String::from_utf8_lossy(&bytes).into_owned()));
            }
            OP_EXECUTION_MODE if a.len() >= 5 && a[1] == MODE_LOCAL_SIZE => {
                if m.entry.as_ref().is_some_and(|e| e.0 == a[0]) {
                    m.local_size = Some([a[2], a[3], a[4]]);
                }
            }
            OP_DECORATE => m.decorations.entry(a[0]).or_default().push((a[1], a.get(2).copied().unwrap_or(0))),
            OP_MEMBER_DECORATE => m.members.entry((a[0], a[1])).or_default().push((a[2], a.get(3).copied().unwrap_or(0))),
            OP_TYPE_INT | OP_TYPE_FLOAT | OP_TYPE_VECTOR | OP_TYPE_MATRIX | OP_TYPE_ARRAY | OP_TYPE_RUNTIME_ARRAY | OP_TYPE_STRUCT | OP_TYPE_POINTER => {
                m.types.insert(a[0], (op, a[1..].to_vec()));
            }
            OP_CONSTANT if a.len() >= 3 => {
                m.constants.insert(a[1], a[2]);
            }
            OP_VARIABLE => m.variables.push((a[1], a[0], a[2])),
            _ => {}
        }
        i += n;
    }
    Ok(m)
}

impl Module {
    fn dec(&self, id: u32, d: u32) -> Option<u32> {
        self.decorations.get(&id)?.iter().find(|x| x.0 == d).map(|x| x.1)
    }

    fn has_dec(&self, id: u32, d: u32) -> bool {
        self.decorations.get(&id).is_some_and(|v| v.iter().any(|x| x.0 == d))
    }

    fn member_dec(&self, s: u32, member: u32, d: u32) -> Option<u32> {
        self.members.get(&(s, member))?.iter().find(|x| x.0 == d).map(|x| x.1)
    }

    /// The struct a pointer ultimately points at, looking through arrays.
    fn pointee_struct(&self, ptr_ty: u32) -> Option<u32> {
        let (op, a) = self.types.get(&ptr_ty)?;
        if *op != OP_TYPE_POINTER {
            return None;
        }
        let mut t = a[1];
        for _ in 0..4 {
            match self.types.get(&t)? {
                (OP_TYPE_STRUCT, _) => return Some(t),
                (OP_TYPE_ARRAY | OP_TYPE_RUNTIME_ARRAY, a) => t = a[0],
                _ => return None,
            }
        }
        None
    }

    /// Size in bytes of a type laid out in a push-constant block.
    fn size_of(&self, t: u32) -> Option<u32> {
        let (op, a) = self.types.get(&t)?;
        Some(match *op {
            OP_TYPE_INT | OP_TYPE_FLOAT => a[0] / 8,
            OP_TYPE_VECTOR => a[1] * self.size_of(a[0])?,
            OP_TYPE_MATRIX => a[1] * self.size_of(a[0])?,
            OP_TYPE_ARRAY => {
                let len = *self.constants.get(&a[1])?;
                let stride = self.dec(t, DEC_ARRAY_STRIDE).unwrap_or(self.size_of(a[0])?);
                len * stride
            }
            OP_TYPE_STRUCT => {
                let mut end = 0;
                for (i, &member) in a.iter().enumerate() {
                    let off = self.member_dec(t, i as u32, DEC_OFFSET)?;
                    end = end.max(off + self.size_of(member)?);
                }
                end
            }
            _ => return None,
        })
    }
}

/// Reflects the first compute entry point of `words`.
pub fn reflect(words: &[u32]) -> Result<Reflection> {
    let m = parse(words)?;
    let (_, entry) = m.entry.clone().ok_or_else(|| Error::Invalid("SPIR-V has no compute entry point".into()))?;

    let mut access: HashMap<u32, Access> = HashMap::new();
    let mut push_bytes = 0;
    for &(var, ptr_ty, class) in &m.variables {
        match class {
            SC_PUSH_CONSTANT => {
                let s = m.pointee_struct(ptr_ty).ok_or_else(|| Error::Invalid("push constant block is not a struct".into()))?;
                push_bytes = push_bytes.max(m.size_of(s).ok_or_else(|| Error::Invalid("cannot size the push constant block".into()))?.next_multiple_of(4));
            }
            SC_STORAGE_BUFFER | SC_UNIFORM => {
                if m.dec(var, DEC_DESCRIPTOR_SET).is_some_and(|s| s != 0) {
                    return Err(Error::Invalid("kernels may only use descriptor set 0".into()));
                }
                let Some(binding) = m.dec(var, DEC_BINDING) else { continue };
                let (can_read, can_write) = match m.pointee_struct(ptr_ty) {
                    Some(s) => {
                        // A block is read-only/write-only when every member says so.
                        let n = match m.types.get(&s) {
                            Some((_, a)) => a.len() as u32,
                            None => 0,
                        };
                        let all = |d: u32| n > 0 && (0..n).all(|i| m.member_dec(s, i, d).is_some()) || m.has_dec(var, d);
                        (!all(DEC_NON_READABLE), !all(DEC_NON_WRITABLE))
                    }
                    None => (true, true),
                };
                // Uniform buffers are read-only by definition.
                let can_write = can_write && class == SC_STORAGE_BUFFER;
                let a = match (can_read, can_write) {
                    (true, false) => Access::Read,
                    (false, true) => Access::Write,
                    _ => Access::ReadWrite,
                };
                // Several variables may alias one binding (A and A_PACKED16): union.
                let merged = match (access.get(&binding), a) {
                    (None, a) => a,
                    (Some(x), y) if *x == y => y,
                    (Some(x), y) => {
                        let (r, w) = (x.reads() || y.reads(), x.writes() || y.writes());
                        match (r, w) {
                            (true, false) => Access::Read,
                            (false, true) => Access::Write,
                            _ => Access::ReadWrite,
                        }
                    }
                };
                access.insert(binding, merged);
            }
            _ => {}
        }
    }
    let n = access.keys().max().map_or(0, |&b| b + 1);
    let bindings = (0..n).map(|b| access.get(&b).copied().unwrap_or(Access::Read)).collect();
    let workgroup = m.local_size.unwrap_or([0; 3]);
    Ok(Reflection { bindings, push_bytes, workgroup, entry })
}

/// A Vulkan blob straight from SPIR-V: the signature comes from reflection.
/// `offsets` says who handles unaligned binding offsets (see [`OffsetMode`]).
pub fn blob(words: Vec<u32>, offsets: OffsetMode, spec: Vec<u32>) -> Result<KernelBlob> {
    let r = reflect(&words)?;
    // With `Prefix`, the backend prepends one residual-offset word per binding to the push
    // constants the launch supplies, so the block holds those plus `sig.push_bytes`.
    let push_bytes = if offsets == OffsetMode::Prefix {
        r.push_bytes
            .checked_sub(4 * r.bindings.len() as u32)
            .ok_or_else(|| Error::Invalid("a kernel with prefix offsets needs a push block starting with one word per binding".into()))?
    } else {
        r.push_bytes
    };
    Ok(KernelBlob {
        target: Target::Vulkan,
        code: Code::Spirv(words),
        entry: r.entry,
        label: String::new(),
        sig: Signature { bindings: r.bindings, push_bytes, workgroup: r.workgroup, offsets },
        spec,
    })
}

#[cfg(all(test, feature = "wgsl"))]
mod tests {
    use super::*;

    #[test]
    fn reflects_a_naga_kernel() {
        let src = r#"
struct PC { oa: u32, ob: u32, n: u32, s: f32 }
var<immediate> pc: PC;
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;
@compute @workgroup_size(32, 2)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    b[id.x] = a[id.x] * pc.s;
    c[id.x] = b[id.x];
}"#;
        let words = super::super::blob::wgsl::spirv(src, "main").unwrap();
        let r = reflect(&words).unwrap();
        assert_eq!(r.entry, "main");
        assert_eq!(r.bindings, vec![Access::Read, Access::ReadWrite, Access::Read, Access::ReadWrite]);
        assert_eq!(r.push_bytes, 16);
        assert_eq!(r.workgroup, [32, 2, 1]);
    }

    #[test]
    fn rejects_garbage() {
        assert!(reflect(&[1, 2, 3, 4, 5]).is_err());
    }
}
