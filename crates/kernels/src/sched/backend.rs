//! The contract between the scheduler and a GPU API: allocate, move bytes,
//! make pipelines from blobs, record and submit groups of launches, poll
//! fences. Nothing here knows about tensors, ops or sessions.

use super::blob::{KernelBlob, Target};
use crate::Result;

#[derive(Clone, Copy, Debug, Default)]
pub struct Limits {
    pub max_buffer_bytes: u64,
    /// Granularity of a binding's start offset; the backend rounds down and
    /// hands the residual to the kernel (see docs/design/scheduler.md).
    pub storage_offset_align: u64,
    pub max_bindings: u32,
    pub max_push_bytes: u32,
    pub max_workgroup_invocations: u32,
    pub max_groups: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemKind {
    /// Device-local, for weights and session state.
    Device,
    /// A submission arena: the host writes inputs and reads results often, so
    /// prefer memory both sides can reach.
    Step,
}

/// A byte range of a buffer bound to a kernel.
pub struct Bound<'a, B: Backend + ?Sized> {
    pub buffer: &'a B::Buffer,
    pub offset: u64,
    pub len: u64,
}

pub struct Launch<'a, B: Backend + ?Sized> {
    pub pipeline: &'a B::Pipeline,
    /// In signature order.
    pub bindings: Vec<Bound<'a, B>>,
    /// Exactly `sig.push_bytes` long.
    pub push: &'a [u8],
    /// Workgroup counts.
    pub grid: [u32; 3],
}

pub trait Backend: Send + Sync + Sized + 'static {
    type Buffer: Send + Sync;
    type Pipeline: Send + Sync;
    /// A submission in flight. Dropping it must wait for the device, so the
    /// resources it uses (kept alive by the caller) are never freed early.
    type Pending: Send;

    const TARGET: Target;

    fn name(&self) -> String;
    fn limits(&self) -> Limits;

    fn alloc(&self, bytes: u64, kind: MemKind) -> Result<Self::Buffer>;
    fn buffer_len(&self, buf: &Self::Buffer) -> u64;
    /// Blocking host-to-device copy.
    fn write(&self, buf: &Self::Buffer, off: u64, data: &[u8]) -> Result<()>;
    /// Blocking device-to-host copy.
    fn read(&self, buf: &Self::Buffer, off: u64, out: &mut [u8]) -> Result<()>;
    /// Blocking fill with zeros.
    fn zero(&self, buf: &Self::Buffer, off: u64, len: u64) -> Result<()>;

    fn pipeline(&self, blob: &KernelBlob) -> Result<Self::Pipeline>;

    /// Records the groups in order with a barrier between consecutive groups
    /// (and one at the start, ordering this submission after earlier ones),
    /// submits and returns without waiting.
    fn submit(&self, groups: &[Vec<Launch<'_, Self>>]) -> Result<Self::Pending>;
    /// Like [`Backend::submit`], also copying `reads` (buffer, offset, length) to the host in the same
    /// submission, so that reading them after the wait needs no second round trip to the device. A
    /// backend without this ignores the request and `prefetched` says no.
    fn submit_reading(&self, groups: &[Vec<Launch<'_, Self>>], reads: &[(&Self::Buffer, u64, u64)]) -> Result<Self::Pending> {
        let _ = reads;
        self.submit(groups)
    }
    /// After the wait: fills `out` from the copy `submit_reading` made of `buffer` at `off`, if it made one.
    fn prefetched(&self, pending: &Self::Pending, buffer: &Self::Buffer, off: u64, out: &mut [u8]) -> bool {
        let _ = (pending, buffer, off, out);
        false
    }
    /// True when the submission has finished.
    fn poll(&self, pending: &Self::Pending) -> Result<bool>;
    fn wait(&self, pending: &Self::Pending) -> Result<()>;
}
