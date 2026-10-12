//! The Vulkan executor's lowering (`vulkan::lower`, Tree kernels only) on the Metal backend: the same
//! graph passes and launch lists, with MSL blobs from the Tree compiler instead of SPIR-V. The lowering
//! only needs a [`Caps`] describing the device, built here.

use super::backend::{Metal, OFFSET_ALIGN};
use crate::vulkan::exec::Executor;
use crate::vulkan::Caps;
use crate::Result;

impl Metal {
    /// What the Tree lowering needs to know: 32-wide SIMD-groups with arithmetic, half floats, no
    /// packed int8 dot product and (so far) no matrix units, which Tree has no Metal form for yet.
    pub fn tree_caps(&self) -> Caps {
        let limits = <Self as crate::sched::Backend>::limits(self);
        Caps {
            vendor_id: 0x106b,
            subgroup_size: 32,
            subgroup_arithmetic: true,
            fp16: true,
            int_dot: false,
            coopmat: std::env::var_os("TL_TREE_NOCM").is_none(),
            int_mma: false,
            shared_memory: self.device().props().max_threadgroup_memory.min(32 * 1024),
            max_groups: limits.max_groups,
            max_storage_range: limits.max_buffer_bytes,
            storage_offset_align: OFFSET_ALIGN,
            // Not exposed by Metal; sizes split-KV and mat-vec grids (2 x cores workgroups).
            shader_cores: std::env::var("TL_METAL_CORES").ok().and_then(|v| v.parse().ok()).unwrap_or(10),
        }
    }
}

/// An executor that runs every node on Tree kernels (compile them with `TreeKernels::from_blobs`).
pub fn executor(metal: Metal) -> Result<Executor<Metal>> {
    let caps = metal.tree_caps();
    Ok(Executor::with_backend(metal, caps))
}
