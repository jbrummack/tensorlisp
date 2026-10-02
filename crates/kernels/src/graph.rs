//! Backend-neutral graph IR: what a lowering pass needs to know about a graph
//! that was built elsewhere (today: ggml's cgraph, built by a Scheme program).
//!
//! Op names and `op_params` follow ggml (`"MUL_MAT"`, `"UNARY"` with the
//! unary op in `op_params[0]`, ...), so kernels written for ggml's layouts
//! (`ne`/`nb` in elements/bytes, innermost dimension first) apply unchanged.

/// ggml's `enum ggml_type` numbering.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Ty(pub u32);

impl Ty {
    pub const F32: Ty = Ty(0);
    pub const F16: Ty = Ty(1);
    pub const Q4_0: Ty = Ty(2);
    pub const Q4_1: Ty = Ty(3);
    pub const Q5_0: Ty = Ty(6);
    pub const Q5_1: Ty = Ty(7);
    pub const Q8_0: Ty = Ty(8);
    pub const Q2_K: Ty = Ty(10);
    pub const Q3_K: Ty = Ty(11);
    pub const Q4_K: Ty = Ty(12);
    pub const Q5_K: Ty = Ty(13);
    pub const Q6_K: Ty = Ty(14);
    pub const I32: Ty = Ty(26);
    pub const I64: Ty = Ty(27);
    pub const BF16: Ty = Ty(30);

    /// (name as in `ggml_type_name`, elements per block, bytes per block)
    fn traits(self) -> Option<(&'static str, i64, u64)> {
        Some(match self.0 {
            0 => ("f32", 1, 4),
            1 => ("f16", 1, 2),
            2 => ("q4_0", 32, 18),
            3 => ("q4_1", 32, 20),
            6 => ("q5_0", 32, 22),
            7 => ("q5_1", 32, 24),
            8 => ("q8_0", 32, 34),
            10 => ("q2_K", 256, 84),
            11 => ("q3_K", 256, 110),
            12 => ("q4_K", 256, 144),
            13 => ("q5_K", 256, 176),
            14 => ("q6_K", 256, 210),
            24 => ("i8", 1, 1),
            25 => ("i16", 1, 2),
            26 => ("i32", 1, 4),
            27 => ("i64", 1, 8),
            30 => ("bf16", 1, 2),
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        self.traits().map_or("unknown", |t| t.0)
    }
    /// Elements per block (1 for non-quantized types).
    pub fn blck(self) -> i64 {
        self.traits().unwrap_or_else(|| panic!("unsupported ggml type {}", self.0)).1
    }
    pub fn size(self) -> u64 {
        self.traits().unwrap_or_else(|| panic!("unsupported ggml type {}", self.0)).2
    }
    pub fn is_quantized(self) -> bool {
        self.traits().is_some_and(|t| t.1 > 1)
    }
}

pub type TensorId = usize;

/// Where a (non-view) tensor's bytes live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Storage {
    /// Index into the weight table handed to the planner.
    Weight(usize),
    /// A state tensor (survives runs), by index.
    State(usize),
    /// A graph input, by index (the caller uploads it).
    Input(usize),
    /// Computed in the arena; the planner picks the offset.
    Temp,
}

#[derive(Clone, Debug)]
pub struct Tensor {
    pub ty: Ty,
    /// Elements per dimension, innermost first.
    pub ne: [i64; 4],
    /// Bytes per step in each dimension.
    pub nb: [u64; 4],
    /// ggml op name, `"NONE"` for leaves.
    pub op: String,
    pub op_params: [i32; 16],
    /// Same slots as ggml's `src[]` (up to 10).
    pub src: Vec<Option<TensorId>>,
    /// Views alias the storage of `view_src` (always a root) at `view_offs` bytes.
    pub view_src: Option<TensorId>,
    pub view_offs: u64,
    /// Meaningful for roots (`view_src == None`).
    pub storage: Storage,
    pub name: String,
}

impl Tensor {
    pub fn nelements(&self) -> i64 {
        self.ne.iter().product()
    }

    pub fn nrows(&self) -> i64 {
        self.ne[1] * self.ne[2] * self.ne[3]
    }

    /// Bytes spanned in memory (ggml_nbytes).
    pub fn nbytes(&self) -> u64 {
        if self.ne.iter().any(|&n| n == 0) {
            return 0;
        }
        let blck = self.ty.blck();
        let mut n = if blck == 1 {
            self.ty.size()
        } else {
            (self.ne[0] as u64 * self.nb[0]) / blck as u64
        };
        if blck == 1 {
            n += (self.ne[0] as u64 - 1) * self.nb[0];
        }
        for i in 1..4 {
            n += (self.ne[i] as u64 - 1) * self.nb[i];
        }
        n
    }

    pub fn is_contiguous(&self) -> bool {
        self.is_contiguous_n(0)
    }

    /// ggml_is_contiguous_rows: rows are dense, higher dims may be strided.
    pub fn is_contiguous_rows(&self) -> bool {
        self.nb[0] == self.ty.size() && self.ne[0] as u64 * self.ty.size() / self.ty.blck() as u64 == self.row_size()
    }

    pub fn row_size(&self) -> u64 {
        self.ty.size() * self.ne[0] as u64 / self.ty.blck() as u64
    }

    /// ggml_is_contiguous_0/1/2 (`n` = number of leading dims allowed to be strided).
    pub fn is_contiguous_n(&self, n: usize) -> bool {
        if self.ne.iter().any(|&e| e == 0) {
            return true;
        }
        let blck = self.ty.blck() as u64;
        if self.ne[0] as u64 != blck && self.nb[0] != self.ty.size() {
            return false;
        }
        let mut next = self.ty.size() * self.ne[0] as u64 / blck;
        for i in 1..4 {
            if self.ne[i] != 1 {
                if i > n {
                    if self.nb[i] != next {
                        return false;
                    }
                    next *= self.ne[i] as u64;
                } else {
                    next = self.ne[i] as u64 * self.nb[i];
                }
            }
        }
        true
    }

    /// ggml_is_transposed: the first two strides are out of order.
    pub fn is_transposed(&self) -> bool {
        self.nb[0] > self.nb[1]
    }

    pub fn is_view_op(&self) -> bool {
        matches!(self.op.as_str(), "NONE" | "RESHAPE" | "VIEW" | "TRANSPOSE" | "PERMUTE")
    }

    pub fn src(&self, i: usize) -> Option<TensorId> {
        self.src.get(i).copied().flatten()
    }

    pub fn op_param_f32(&self, i: usize) -> f32 {
        f32::from_bits(self.op_params[i] as u32)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub tensors: Vec<Tensor>,
    /// Execution order (indices into `tensors`); view ops included (no-ops).
    pub nodes: Vec<TensorId>,
    /// Tensors that must stay valid after the last node (outputs, taps).
    pub keep: Vec<TensorId>,
}

impl Graph {
    /// The root tensor owning `id`'s storage, and `id`'s byte offset into it.
    pub fn root(&self, id: TensorId) -> (TensorId, u64) {
        let t = &self.tensors[id];
        match t.view_src {
            Some(r) => (r, t.view_offs),
            None => (id, 0),
        }
    }
}
