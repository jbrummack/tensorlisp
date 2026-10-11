//! Backend-neutral scheduling: kernel blobs, hazard analysis, session/step
//! memory pools, asynchronous submission. See docs/design/scheduler.md.

pub mod backend;
pub mod blob;
pub mod engine;
pub mod handle;
pub mod hazard;
pub mod pool;
pub mod spirv;

pub use backend::{Backend, Limits, MemKind};
pub use blob::{Access, Code, KernelBlob, KernelKey, KernelSource, OffsetMode, Registry, Signature, Target};
pub use engine::{CompiledPlan, Engine, LaunchDesc, ModelMem, Session, Ticket};
pub use handle::{Handle, HandleTable};
pub use hazard::{Footprint, Range, Space};
