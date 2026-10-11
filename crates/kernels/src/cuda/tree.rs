//! The Vulkan executor's lowering (`vulkan::lower`, Tree kernels only) on the CUDA backend: the
//! same graph passes and launch lists, with PTX blobs from the Tree compiler instead of SPIR-V.
//! The lowering only needs a [`Caps`] describing the device, built here from the driver's attributes.

use super::backend::Cuda;
use super::sys;
use crate::vulkan::exec::Executor;
use crate::vulkan::Caps;
use crate::Result;

impl Cuda {
    /// A device attribute (`CU_DEVICE_ATTRIBUTE_*`), or `default` when the driver does not say.
    pub fn attribute(&self, attr: i32, default: u32) -> u32 {
        let Some(drv) = sys::driver() else { return default };
        let mut v = 0;
        let ret = unsafe { (drv.cuDeviceGetAttribute)(&mut v, attr, self.device().dev) };
        if ret == sys::CUDA_SUCCESS && v > 0 { v as u32 } else { default }
    }

    /// What the Tree lowering needs to know: 32-wide warps (subgroups), `dp4a`, and on sm_80+ the
    /// `mma.sync` forms that stand in for cooperative matrices.
    pub fn tree_caps(&self) -> Caps {
        let limits = <Self as crate::sched::Backend>::limits(self);
        Caps {
            vendor_id: 0x10de,
            subgroup_size: 32,
            subgroup_arithmetic: true,
            fp16: true,
            int_dot: true,
            coopmat: self.attribute(sys::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0) >= 8 && std::env::var_os("TL_TREE_NOCM").is_none(),
            int_mma: self.attribute(sys::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0) >= 8 && std::env::var_os("TL_TREE_NOMMA").is_none(),
            shared_memory: self.attribute(sys::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, 48 * 1024).min(48 * 1024),
            max_groups: limits.max_groups,
            max_storage_range: u64::MAX / 2,
            storage_offset_align: 1,
            shader_cores: self.attribute(sys::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 16),
        }
    }
}

/// An executor that runs every node on Tree kernels (compile them with `TreeKernels::from_blobs`).
pub fn executor(cuda: Cuda) -> Result<Executor<Cuda>> {
    let caps = cuda.tree_caps();
    Ok(Executor::with_backend(cuda, caps))
}
