//! Vulkan compute backend for the scheduler (`sched::Backend`).
//!
//! SPIR-V blobs in, asynchronous submissions out. `ash` dlopen's the loader,
//! so this builds with no SDK and `Vulkan::system_default()` reports
//! [`Error::NoDevice`] when there is no driver. Conventions every kernel
//! follows (docs/design/scheduler.md):
//!
//! * set 0, bindings `0..n` are storage buffers in signature order;
//! * push constants are `u32 offs[n]` (residual byte offset of each binding)
//!   followed by the kernel's own parameters.

use std::collections::HashMap;
use std::ffi::{c_void, CStr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ash::vk;

pub mod exec;
pub mod fuse;
pub mod fuse_tree;
pub mod ggml;
pub mod glsl;
pub mod lower;
pub mod tree;

use crate::sched::backend::{Backend, Launch, Limits, MemKind};
use crate::sched::blob::{Code, KernelBlob, OffsetMode, Target};
use crate::{Error, Result};

/// Errors reported by the validation layer since the process started.
static VALIDATION_ERRORS: AtomicU64 = AtomicU64::new(0);

/// How the device is opened.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    /// Enable `VK_LAYER_KHRONOS_validation` (with synchronization validation)
    /// when the layer is installed; messages go to stderr and errors are
    /// counted in [`validation_errors`]. Defaults from `TL_VULKAN_VALIDATE=1`
    /// in [`Vulkan::system_default`].
    pub validate: bool,
}

/// Validation errors seen so far in this process.
pub fn validation_errors() -> u64 {
    VALIDATION_ERRORS.load(Ordering::SeqCst)
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    let msg = if data.is_null() || unsafe { (*data).p_message.is_null() } {
        String::from("(no message)")
    } else {
        unsafe { CStr::from_ptr((*data).p_message) }.to_string_lossy().into_owned()
    };
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        VALIDATION_ERRORS.fetch_add(1, Ordering::SeqCst);
        eprintln!("[vulkan validation error] {msg}");
    } else {
        eprintln!("[vulkan validation warning] {msg}");
    }
    vk::FALSE
}

fn vk_err<T>(r: ash::prelude::VkResult<T>, what: &str) -> Result<T> {
    r.map_err(|e| Error::Execution(format!("vulkan {what}: {e}")))
}

/// Staging buffer chunk for copies to and from device-only memory.
const STAGING_BYTES: u64 = 16 << 20;

/// Command buffer, fence and descriptor pool, recycled per submission.
struct Ctx {
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    dpool: vk::DescriptorPool,
    dsets: u32,
    ddescs: u32,
    /// Host-visible memory the recording copies requested results into (`Backend::submit_reading`).
    rb: Option<Staging>,
}

/// A finished submission whose command buffer can be submitted again: the key is everything that was
/// recorded (pipelines, buffers, offsets, push constants, grids), so an equal key means an equal command buffer.
struct Replay {
    key: Vec<u64>,
    ctx: Ctx,
}

/// How many recordings are kept for replay.
const REPLAYS: usize = 8;

struct Staging {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    size: u64,
}

unsafe impl Send for Staging {}

/// Optional shader features that were enabled on the device.
#[derive(Clone, Copy, Debug, Default)]
pub struct Features {
    pub shader_int16: bool,
    pub shader_int64: bool,
    /// 16-bit types in storage buffers.
    pub storage16: bool,
    /// 8-bit types in storage buffers.
    pub storage8: bool,
    pub float16: bool,
    pub int8: bool,
    /// 16-bit/8-bit types in subgroup operations.
    pub subgroup_extended_types: bool,
    pub buffer_address: bool,
    /// `OpSDot` on packed 4x8-bit integers (VK_KHR_shader_integer_dot_product), the dp4a of PTX.
    pub int_dot: bool,
}

/// What the kernel selection logic needs to know about the device.
#[derive(Clone, Copy, Debug, Default)]
pub struct Caps {
    pub vendor_id: u32,
    pub subgroup_size: u32,
    /// Subgroup add/reduce in compute shaders.
    pub subgroup_arithmetic: bool,
    /// 16-bit floats in shaders (`shaderFloat16`).
    pub fp16: bool,
    /// Packed int8 dot products in shaders (`OpSDot`).
    pub int_dot: bool,
    pub shared_memory: u32,
    pub max_groups: [u32; 3],
    /// `maxStorageBufferRange`: the largest binding.
    pub max_storage_range: u64,
    pub storage_offset_align: u64,
    /// Streaming multiprocessors / compute units when the driver reports them, else 16.
    pub shader_cores: u32,
}

struct Inner {
    _entry: ash::Entry,
    instance: ash::Instance,
    phys: vk::PhysicalDevice,
    device: ash::Device,
    queue: Mutex<vk::Queue>,
    family: u32,
    mem: vk::PhysicalDeviceMemoryProperties,
    limits: Limits,
    max_range: u64,
    name: String,
    features: Features,
    subgroup_size: u32,
    caps: Caps,
    /// Memory type for step arenas that both sides can map, when the device
    /// has a big enough heap of it (resizable BAR / unified memory).
    mapped_device_type: Option<u32>,
    ctxs: Mutex<Vec<Ctx>>,
    /// Recorded submissions that finished, to be replayed when the same one comes again (see `Vulkan::submit`).
    replays: Mutex<Vec<Replay>>,
    /// Host-to-device staging (write-combined memory is fine) and device-to-host staging (needs
    /// cached memory: the CPU reads it back, and uncached reads crawl at a few hundred MB/s).
    staging: Mutex<Option<Staging>>,
    staging_read: Mutex<Option<Staging>>,
    debug: Option<(ash::ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    validated: bool,
    /// `TL_VULKAN_PROFILE`: timestamp every launch and print a per-kernel table when the device closes.
    profile: bool,
    timestamp_period: f32,
    prof: Mutex<HashMap<String, Agg>>,
}

/// Accumulated GPU time of one kernel label.
#[derive(Default, Clone)]
struct Agg {
    launches: u64,
    /// Nanoseconds from the launch's start stamp to its end stamp.
    busy_ns: f64,
    /// Nanoseconds between the previous launch's end and this launch's start (barrier + drain).
    gap_ns: f64,
}

/// Timestamps recorded into one submission: two per launch, in submission order.
struct Prof {
    pool: vk::QueryPool,
    labels: Vec<String>,
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Inner {
    fn find_type(&self, bits: u32, want: vk::MemoryPropertyFlags, avoid: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.mem.memory_type_count).find(|&i| {
            let f = self.mem.memory_types[i as usize].property_flags;
            bits & (1 << i) != 0 && f.contains(want) && !f.intersects(avoid)
        })
    }

    fn new_ctx(&self) -> Result<Ctx> {
        unsafe {
            let pool = vk_err(
                self.device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(self.family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                ),
                "create_command_pool",
            )?;
            let cmd = vk_err(
                self.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(1),
                ),
                "allocate_command_buffers",
            )?[0];
            let fence = vk_err(self.device.create_fence(&vk::FenceCreateInfo::default(), None), "create_fence")?;
            Ok(Ctx { pool, cmd, fence, dpool: vk::DescriptorPool::null(), dsets: 0, ddescs: 0, rb: None })
        }
    }

    fn take_ctx(&self) -> Result<Ctx> {
        let ctx = self.ctxs.lock().unwrap().pop();
        match ctx {
            Some(c) => Ok(c),
            None => self.new_ctx(),
        }
    }

    /// Returns a finished ctx to the pool, reset for the next submission.
    fn recycle(&self, ctx: Ctx) {
        unsafe {
            let _ = self.device.reset_command_buffer(ctx.cmd, vk::CommandBufferResetFlags::empty());
            let _ = self.device.reset_fences(&[ctx.fence]);
            if ctx.dpool != vk::DescriptorPool::null() {
                let _ = self.device.reset_descriptor_pool(ctx.dpool, vk::DescriptorPoolResetFlags::empty());
            }
        }
        self.ctxs.lock().unwrap().push(ctx);
    }

    fn take_replay(&self, key: &[u64]) -> Option<Ctx> {
        let mut replays = self.replays.lock().unwrap();
        let at = replays.iter().position(|r| r.key == key)?;
        Some(replays.remove(at).ctx)
    }

    /// Keeps a finished, recorded ctx for replay (the oldest recording goes when there are too many).
    fn store_replay(&self, key: Vec<u64>, ctx: Ctx) {
        unsafe {
            let _ = self.device.reset_fences(&[ctx.fence]);
        }
        let evicted = {
            let mut replays = self.replays.lock().unwrap();
            replays.push(Replay { key, ctx });
            if replays.len() > REPLAYS { Some(replays.remove(0)) } else { None }
        };
        if let Some(r) = evicted {
            self.destroy_ctx(&r.ctx);
        }
    }

    /// Forgets every recording: a destroyed buffer may be named again by a new one.
    fn clear_replays(&self) {
        let all: Vec<Replay> = self.replays.lock().unwrap().drain(..).collect();
        for r in all {
            self.destroy_ctx(&r.ctx);
        }
    }

    fn destroy_ctx(&self, ctx: &Ctx) {
        unsafe {
            if ctx.dpool != vk::DescriptorPool::null() {
                self.device.destroy_descriptor_pool(ctx.dpool, None);
            }
            self.device.destroy_fence(ctx.fence, None);
            self.device.destroy_command_pool(ctx.pool, None);
            if let Some(rb) = &ctx.rb {
                self.device.destroy_buffer(rb.buffer, None);
                self.device.free_memory(rb.memory, None);
            }
        }
    }

    /// Reads the timestamps of a finished submission into the per-kernel totals.
    fn collect_profile(&self, p: Prof) {
        let n = p.labels.len() * 2;
        let mut stamps = vec![0u64; n];
        unsafe {
            if n > 0 {
                let r = self.device.get_query_pool_results(p.pool, 0, &mut stamps, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT);
                if r.is_ok() {
                    let period = self.timestamp_period as f64;
                    let mut prof = self.prof.lock().unwrap();
                    let mut prev_end = stamps[0];
                    for (i, label) in p.labels.iter().enumerate() {
                        let (start, end) = (stamps[2 * i], stamps[2 * i + 1]);
                        let a = prof.entry(label.clone()).or_default();
                        a.launches += 1;
                        a.busy_ns += end.saturating_sub(start) as f64 * period;
                        a.gap_ns += start.saturating_sub(prev_end) as f64 * period;
                        prev_end = prev_end.max(end);
                    }
                }
            }
            self.device.destroy_query_pool(p.pool, None);
        }
    }

    /// The per-kernel table, heaviest first.
    fn profile_report(&self) -> String {
        let prof = self.prof.lock().unwrap();
        let mut rows: Vec<(&String, &Agg)> = prof.iter().collect();
        rows.sort_by(|a, b| (b.1.busy_ns + b.1.gap_ns).total_cmp(&(a.1.busy_ns + a.1.gap_ns)));
        let (busy, gap): (f64, f64) = rows.iter().fold((0.0, 0.0), |t, r| (t.0 + r.1.busy_ns, t.1 + r.1.gap_ns));
        let mut out = format!("{:<44} {:>9} {:>11} {:>11} {:>9}\n", "kernel", "launches", "busy ms", "gap ms", "busy us/l");
        for (name, a) in rows.iter().take(40) {
            out += &format!("{:<44} {:>9} {:>11.3} {:>11.3} {:>9.1}\n", name, a.launches, a.busy_ns / 1e6, a.gap_ns / 1e6, a.busy_ns / 1e3 / a.launches as f64);
        }
        out += &format!("{:<44} {:>9} {:>11.3} {:>11.3}\n", "total", rows.iter().map(|r| r.1.launches).sum::<u64>(), busy / 1e6, gap / 1e6);
        out
    }

    /// A global memory barrier: everything earlier in the queue before everything later.
    fn global_barrier(&self, cmd: vk::CommandBuffer) {
        let stages = vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER;
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
            );
        unsafe { self.device.cmd_pipeline_barrier(cmd, stages, stages, vk::DependencyFlags::empty(), &[barrier], &[], &[]) };
    }

    /// Records with `f`, submits and waits.
    fn one_shot(&self, f: impl FnOnce(vk::CommandBuffer)) -> Result<()> {
        let ctx = self.take_ctx()?;
        let result = (|| unsafe {
            vk_err(
                self.device.begin_command_buffer(ctx.cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)),
                "begin_command_buffer",
            )?;
            self.global_barrier(ctx.cmd);
            f(ctx.cmd);
            self.global_barrier(ctx.cmd);
            vk_err(self.device.end_command_buffer(ctx.cmd), "end_command_buffer")?;
            let cmds = [ctx.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            {
                let q = self.queue.lock().unwrap();
                vk_err(self.device.queue_submit(*q, &[submit], ctx.fence), "queue_submit")?;
            }
            vk_err(self.device.wait_for_fences(&[ctx.fence], true, u64::MAX), "wait_for_fences")
        })();
        self.recycle(ctx);
        result
    }

    fn staging_buffer(&self, read: bool) -> Result<std::sync::MutexGuard<'_, Option<Staging>>> {
        let mut guard = if read { self.staging_read.lock().unwrap() } else { self.staging.lock().unwrap() };
        if guard.is_none() {
            *guard = Some(self.make_staging(STAGING_BYTES, read)?);
        }
        Ok(guard)
    }

    /// Host-visible memory of `size` bytes: cached system memory when the CPU reads it.
    fn make_staging(&self, size: u64, read: bool) -> Result<Staging> {
        unsafe {
            let buffer = vk_err(
                self.device.create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(size)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                ),
                "create_buffer (staging)",
            )?;
            let req = self.device.get_buffer_memory_requirements(buffer);
            let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let cached = host | vk::MemoryPropertyFlags::HOST_CACHED;
            let local = vk::MemoryPropertyFlags::DEVICE_LOCAL;
            let ty = if read {
                // The CPU reads this back: cached system memory.
                self.find_type(req.memory_type_bits, cached, local).or_else(|| self.find_type(req.memory_type_bits, cached, vk::MemoryPropertyFlags::empty()))
            } else {
                None
            }
            .or_else(|| self.find_type(req.memory_type_bits, host, local | vk::MemoryPropertyFlags::HOST_CACHED))
            .or_else(|| self.find_type(req.memory_type_bits, host, local))
            .or_else(|| self.find_type(req.memory_type_bits, host, vk::MemoryPropertyFlags::empty()))
            .ok_or_else(|| Error::Invalid("no host-visible memory type for staging".into()))?;
            let memory = vk_err(
                self.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None),
                "allocate_memory (staging)",
            )?;
            vk_err(self.device.bind_buffer_memory(buffer, memory, 0), "bind_buffer_memory (staging)")?;
            let ptr = vk_err(self.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()), "map_memory (staging)")? as *mut u8;
            Ok(Staging { buffer, memory, ptr, size })
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        if self.profile {
            eprintln!("vulkan profile (GPU time by kernel, {}):\n{}", self.name, self.profile_report());
        }
        unsafe {
            let _ = self.device.device_wait_idle();
            for c in self.ctxs.lock().unwrap().drain(..) {
                self.destroy_ctx(&c);
            }
            for r in self.replays.lock().unwrap().drain(..) {
                self.destroy_ctx(&r.ctx);
            }
            for slot in [&self.staging, &self.staging_read] {
                if let Some(s) = slot.lock().unwrap().take() {
                    self.device.destroy_buffer(s.buffer, None);
                    self.device.free_memory(s.memory, None);
                }
            }
            self.device.destroy_device(None);
            if let Some((loader, messenger)) = self.debug.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

/// The Vulkan device: one compute queue, shared by every submission.
#[derive(Clone)]
pub struct Vulkan {
    inner: Arc<Inner>,
}

pub struct Buffer {
    inner: Arc<Inner>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    /// Set when the memory is host visible.
    ptr: Option<*mut u8>,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        self.inner.clear_replays();
        unsafe {
            self.inner.device.destroy_buffer(self.buffer, None);
            self.inner.device.free_memory(self.memory, None);
        }
    }
}

pub struct Pipeline {
    inner: Arc<Inner>,
    module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    bindings: usize,
    offsets: OffsetMode,
    label: String,
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            let d = &self.inner.device;
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.layout, None);
            d.destroy_descriptor_set_layout(self.set_layout, None);
            d.destroy_shader_module(self.module, None);
        }
    }
}

/// A submission in flight: its fence and the resources recorded into it.
pub struct Pending {
    inner: Arc<Inner>,
    ctx: Option<Ctx>,
    prof: Option<Prof>,
    /// Set when the recording can be replayed: it goes back to the cache, not to the pool.
    replay: Option<Vec<u64>>,
    /// Copies the recording made for the host: (source buffer, offset, length, offset in `ctx.rb`).
    reads: Vec<(vk::Buffer, u64, u64, u64)>,
}

impl Drop for Pending {
    fn drop(&mut self) {
        if let Some(ctx) = self.ctx.take() {
            unsafe {
                let _ = self.inner.device.wait_for_fences(&[ctx.fence], true, u64::MAX);
            }
            if let Some(p) = self.prof.take() {
                self.inner.collect_profile(p);
            }
            match self.replay.take() {
                Some(key) => self.inner.store_replay(key, ctx),
                None => self.inner.recycle(ctx),
            }
        }
    }
}

impl Vulkan {
    /// The best device the loader offers (`TL_VULKAN_DEVICE=<index>` picks one).
    pub fn system_default() -> Result<Vulkan> {
        Self::open(Options { validate: std::env::var("TL_VULKAN_VALIDATE").is_ok_and(|v| v != "0") })
    }

    /// Like [`Vulkan::system_default`] with explicit options.
    pub fn open(options: Options) -> Result<Vulkan> {
        let entry = unsafe { ash::Entry::load() }.map_err(|_| Error::NoDevice)?;
        let supported = unsafe { entry.try_enumerate_instance_version() }.ok().flatten().unwrap_or(vk::API_VERSION_1_0);
        let api = supported.min(vk::API_VERSION_1_2);
        let app = vk::ApplicationInfo::default().application_name(c"tensorlisp").api_version(api);
        // Validation layer, when asked for and installed.
        let layers = unsafe { entry.enumerate_instance_layer_properties() }.unwrap_or_default();
        let layer_name = c"VK_LAYER_KHRONOS_validation";
        let have_layer = layers.iter().any(|l| unsafe { CStr::from_ptr(l.layer_name.as_ptr()) } == layer_name);
        let exts = unsafe { entry.enumerate_instance_extension_properties(None) }.unwrap_or_default();
        let have_ext = |name: &CStr| exts.iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == name);
        let validate = options.validate && have_layer && have_ext(ash::ext::debug_utils::NAME);
        if options.validate && !validate {
            eprintln!("vulkan: validation requested but VK_LAYER_KHRONOS_validation is not available");
        }
        // Layer settings (VK_EXT_layer_settings, provided by the layer itself, so
        // it only shows up when the layer is named in the query). Synchronization
        // validation only looks at compute dispatches when shader accesses are
        // taken into account, which is off by default.
        let layer_exts = unsafe { entry.enumerate_instance_extension_properties(Some(layer_name)) }.unwrap_or_default();
        let settings_ext = validate && layer_exts.iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == ash::ext::layer_settings::NAME);
        let mut layer_ptrs: Vec<*const std::ffi::c_char> = Vec::new();
        let mut ext_ptrs: Vec<*const std::ffi::c_char> = Vec::new();
        if validate {
            layer_ptrs.push(layer_name.as_ptr());
            ext_ptrs.push(ash::ext::debug_utils::NAME.as_ptr());
            if settings_ext {
                ext_ptrs.push(ash::ext::layer_settings::NAME.as_ptr());
            }
        }
        static TRUE: u32 = vk::TRUE;
        let setting = |name: &'static CStr| {
            let mut s = vk::LayerSettingEXT::default().layer_name(layer_name).setting_name(name).ty(vk::LayerSettingTypeEXT::BOOL32);
            s.value_count = 1;
            s.p_values = &TRUE as *const u32 as *const c_void;
            s
        };
        let settings = [setting(c"validate_sync"), setting(c"syncval_shader_accesses_heuristic")];
        let mut layer_settings = vk::LayerSettingsCreateInfoEXT::default().settings(&settings);
        let mut info = vk::InstanceCreateInfo::default().application_info(&app).enabled_layer_names(&layer_ptrs).enabled_extension_names(&ext_ptrs);
        if settings_ext {
            info = info.push_next(&mut layer_settings);
        }
        let instance = unsafe { entry.create_instance(&info, None) }.map_err(|_| Error::NoDevice)?;
        let debug = if validate {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let create = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION | vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(debug_callback));
            let messenger = unsafe { loader.create_debug_utils_messenger(&create, None) }.ok();
            messenger.map(|m| (loader, m))
        } else {
            None
        };

        let phys_devices = unsafe { instance.enumerate_physical_devices() }.map_err(|_| Error::NoDevice)?;
        let forced = std::env::var("TL_VULKAN_DEVICE").ok().and_then(|s| s.parse::<usize>().ok());
        let mut best: Option<(i32, vk::PhysicalDevice, u32)> = None;
        for (i, &p) in phys_devices.iter().enumerate() {
            let props = unsafe { instance.get_physical_device_properties(p) };
            let families = unsafe { instance.get_physical_device_queue_family_properties(p) };
            // Prefer a compute-only family (async compute), else any compute family.
            let family = families
                .iter()
                .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE) && !f.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .or_else(|| families.iter().position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE)));
            let Some(family) = family else { continue };
            let mut score = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 30,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 20,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 10,
                _ => 0,
            };
            if forced == Some(i) {
                score += 1000;
            }
            if best.as_ref().is_none_or(|b| score > b.0) {
                best = Some((score, p, family as u32));
            }
        }
        let Some((_, phys, family)) = best else {
            unsafe { instance.destroy_instance(None) };
            return Err(Error::NoDevice);
        };

        let props = unsafe { instance.get_physical_device_properties(phys) };
        let mem = unsafe { instance.get_physical_device_memory_properties(phys) };
        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];

        // Enable the shader features kernels commonly declare (ggml's SPIR-V
        // uses 16/8-bit storage, float16, int8/16/64), but only where supported:
        // robustBufferAccess and friends stay off.
        let mut has = Features::default();
        if props.api_version >= vk::API_VERSION_1_2 && api >= vk::API_VERSION_1_2 {
            let mut f11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut f11).push_next(&mut f12);
            unsafe { instance.get_physical_device_features2(phys, &mut f2) };
            let base = f2.features;
            has = Features {
                shader_int16: base.shader_int16 == vk::TRUE,
                shader_int64: base.shader_int64 == vk::TRUE,
                storage16: f11.storage_buffer16_bit_access == vk::TRUE,
                storage8: f12.storage_buffer8_bit_access == vk::TRUE,
                float16: f12.shader_float16 == vk::TRUE,
                int8: f12.shader_int8 == vk::TRUE,
                subgroup_extended_types: f12.shader_subgroup_extended_types == vk::TRUE,
                buffer_address: false,
                int_dot: false,
            };
            // Integer dot products live in an extension struct (core in Vulkan 1.3).
            let ext_ok = unsafe { instance.enumerate_device_extension_properties(phys) }.is_ok_and(|exts| {
                exts.iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == ash::khr::shader_integer_dot_product::NAME)
            });
            if ext_ok {
                let mut dot = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default();
                let mut f2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut dot);
                unsafe { instance.get_physical_device_features2(phys, &mut f2) };
                has.int_dot = dot.shader_integer_dot_product == vk::TRUE;
            }
        }
        let mut e11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(has.storage16);
        let mut e12 = vk::PhysicalDeviceVulkan12Features::default()
            .storage_buffer8_bit_access(has.storage8)
            .shader_float16(has.float16)
            .shader_int8(has.int8)
            .shader_subgroup_extended_types(has.subgroup_extended_types);
        let mut e2 = vk::PhysicalDeviceFeatures2::default().features(
            vk::PhysicalDeviceFeatures::default().shader_int16(has.shader_int16).shader_int64(has.shader_int64),
        );
        let mut edot = vk::PhysicalDeviceShaderIntegerDotProductFeatures::default().shader_integer_dot_product(has.int_dot);
        let dot_ext = [ash::khr::shader_integer_dot_product::NAME.as_ptr()];
        let mut create = vk::DeviceCreateInfo::default().queue_create_infos(&queue_info);
        if props.api_version >= vk::API_VERSION_1_2 && api >= vk::API_VERSION_1_2 {
            e2 = e2.push_next(&mut e11).push_next(&mut e12);
            if has.int_dot {
                e2 = e2.push_next(&mut edot);
                create = create.enabled_extension_names(&dot_ext);
            }
            create = create.push_next(&mut e2);
        }
        let device = unsafe { instance.create_device(phys, &create, None) }
            .map_err(|e| Error::Execution(format!("vulkan create_device: {e}")))?;
        let queue = unsafe { device.get_device_queue(family, 0) };

        let l = props.limits;
        let limits = Limits {
            max_buffer_bytes: l.max_storage_buffer_range as u64,
            storage_offset_align: l.min_storage_buffer_offset_alignment.max(4),
            max_bindings: l.max_per_stage_descriptor_storage_buffers.min(32),
            max_push_bytes: l.max_push_constants_size,
            max_workgroup_invocations: l.max_compute_work_group_invocations,
            max_groups: l.max_compute_work_group_count,
        };
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }.to_string_lossy().into_owned();

        // Step arenas: host-visible device-local memory only if a heap is big
        // enough to hold real arenas (a 256 MiB BAR window is not).
        let want = vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let mapped_device_type = (0..mem.memory_type_count).find(|&i| {
            let t = mem.memory_types[i as usize];
            t.property_flags.contains(want) && mem.memory_heaps[t.heap_index as usize].size >= (4 << 30)
        });

        let (subgroup_size, subgroup_arithmetic) = {
            let mut sg = vk::PhysicalDeviceSubgroupProperties::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sg);
            unsafe { instance.get_physical_device_properties2(phys, &mut p2) };
            (
                sg.subgroup_size,
                sg.supported_operations.contains(vk::SubgroupFeatureFlags::ARITHMETIC)
                    && sg.supported_stages.contains(vk::ShaderStageFlags::COMPUTE),
            )
        };
        // NVIDIA reports its SM count through VK_NV_shader_sm_builtins (a property
        // query only needs the extension to be supported, not enabled).
        let shader_cores = {
            let exts = unsafe { instance.enumerate_device_extension_properties(phys) }.unwrap_or_default();
            let has_sm = exts.iter().any(|e| unsafe { CStr::from_ptr(e.extension_name.as_ptr()) } == ash::nv::shader_sm_builtins::NAME);
            if has_sm {
                let mut sm = vk::PhysicalDeviceShaderSMBuiltinsPropertiesNV::default();
                let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut sm);
                unsafe { instance.get_physical_device_properties2(phys, &mut p2) };
                sm.shader_sm_count.max(1)
            } else {
                16
            }
        };
        let caps = Caps {
            vendor_id: props.vendor_id,
            subgroup_size,
            subgroup_arithmetic,
            fp16: has.float16,
            int_dot: has.int_dot,
            shared_memory: l.max_compute_shared_memory_size,
            max_groups: l.max_compute_work_group_count,
            max_storage_range: l.max_storage_buffer_range as u64,
            storage_offset_align: limits.storage_offset_align,
            shader_cores,
        };

        Ok(Vulkan {
            inner: Arc::new(Inner {
                _entry: entry,
                instance,
                phys,
                device,
                queue: Mutex::new(queue),
                family,
                mem,
                limits,
                max_range: l.max_storage_buffer_range as u64,
                name,
                features: has,
                subgroup_size,
                caps,
                mapped_device_type,
                ctxs: Mutex::new(Vec::new()),
                replays: Mutex::new(Vec::new()),
                staging: Mutex::new(None),
                staging_read: Mutex::new(None),
                debug,
                validated: validate,
                profile: std::env::var_os("TL_VULKAN_PROFILE").is_some_and(|v| v != "0"),
                timestamp_period: l.timestamp_period,
                prof: Mutex::new(HashMap::new()),
            }),
        })
    }

    /// The per-kernel GPU time table (`TL_VULKAN_PROFILE=1`), heaviest first; empty rows otherwise.
    pub fn profile_report(&self) -> String {
        self.inner.profile_report()
    }

    /// Forgets the accumulated profile (between benchmark cases).
    pub fn profile_clear(&self) {
        self.inner.prof.lock().unwrap().clear();
    }

    /// Takes the accumulated profile: (label, launches, busy nanoseconds), clearing it.
    pub fn profile_take(&self) -> Vec<(String, u64, f64)> {
        let mut prof = self.inner.prof.lock().unwrap();
        let out = prof.iter().map(|(k, a)| (k.clone(), a.launches, a.busy_ns)).collect();
        prof.clear();
        out
    }

    /// Shader features enabled on this device.
    pub fn features(&self) -> Features {
        self.inner.features
    }

    pub fn caps(&self) -> Caps {
        self.inner.caps
    }

    pub fn subgroup_size(&self) -> u32 {
        self.inner.subgroup_size
    }

    /// True when the validation layer is active on this device.
    pub fn validated(&self) -> bool {
        self.inner.validated
    }

    pub fn device_name(&self) -> &str {
        &self.inner.name
    }

    /// True when step arenas are plain host-visible VRAM (no staging copies).
    pub fn has_mapped_device_memory(&self) -> bool {
        self.inner.mapped_device_type.is_some()
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.inner.phys
    }
}

/// Everything a recording depends on, flattened: equal keys record the same command buffer.
fn replay_key(groups: &[Vec<Launch<'_, Vulkan>>]) -> Vec<u64> {
    use ash::vk::Handle;
    let mut key = Vec::with_capacity(groups.iter().map(|g| g.len() * 12).sum::<usize>() + groups.len());
    for g in groups {
        key.push(u64::MAX);
        for l in g {
            key.push(l.pipeline.pipeline.as_raw());
            for b in &l.bindings {
                key.push(b.buffer.buffer.as_raw());
                key.push(b.offset);
                key.push(b.len);
            }
            key.extend(l.push.chunks(8).map(|c| {
                let mut w = [0u8; 8];
                w[..c.len()].copy_from_slice(c);
                u64::from_le_bytes(w)
            }));
            key.push(l.grid[0] as u64 | (l.grid[1] as u64) << 32);
            key.push(l.grid[2] as u64);
            key.push(u64::MAX - 1);
        }
    }
    key
}

impl Backend for Vulkan {
    type Buffer = Buffer;
    type Pipeline = Pipeline;
    type Pending = Pending;

    const TARGET: Target = Target::Vulkan;

    fn name(&self) -> String {
        format!("vulkan {}", self.inner.name)
    }

    fn limits(&self) -> Limits {
        self.inner.limits
    }

    fn alloc(&self, bytes: u64, kind: MemKind) -> Result<Buffer> {
        let inner = &self.inner;
        let size = bytes.max(4);
        let usage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST;
        unsafe {
            let buffer = vk_err(
                inner.device.create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE), None),
                "create_buffer",
            )?;
            let req = inner.device.get_buffer_memory_requirements(buffer);
            let mut candidates: Vec<(u32, bool)> = Vec::new(); // (type, host visible)
            if kind == MemKind::Step {
                if let Some(t) = inner.mapped_device_type {
                    if req.memory_type_bits & (1 << t) != 0 {
                        candidates.push((t, true));
                    }
                }
            }
            if let Some(t) = inner.find_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryPropertyFlags::empty()) {
                // Prefer a purely device-local type over one that is also host visible.
                let pure = inner
                    .find_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL, vk::MemoryPropertyFlags::HOST_VISIBLE)
                    .unwrap_or(t);
                candidates.push((pure, false));
            }
            let mut last = Error::Alloc(size as usize);
            for (ty, host) in candidates {
                match inner.device.allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty), None) {
                    Ok(memory) => {
                        vk_err(inner.device.bind_buffer_memory(buffer, memory, 0), "bind_buffer_memory")?;
                        let ptr = if host {
                            Some(vk_err(inner.device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()), "map_memory")? as *mut u8)
                        } else {
                            None
                        };
                        return Ok(Buffer { inner: inner.clone(), buffer, memory, size, ptr });
                    }
                    Err(_) => last = Error::Alloc(size as usize),
                }
            }
            inner.device.destroy_buffer(buffer, None);
            Err(last)
        }
    }

    fn buffer_len(&self, buf: &Buffer) -> u64 {
        buf.size
    }

    fn write(&self, buf: &Buffer, off: u64, data: &[u8]) -> Result<()> {
        if off + data.len() as u64 > buf.size {
            return Err(Error::Invalid(format!("write of {} bytes at {off} is past a buffer of {}", data.len(), buf.size)));
        }
        if let Some(ptr) = buf.ptr {
            unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(off as usize), data.len()) };
            return Ok(());
        }
        let inner = &self.inner;
        let staging = inner.staging_buffer(false)?;
        let s = staging.as_ref().unwrap();
        let mut at = off;
        for chunk in data.chunks(STAGING_BYTES as usize) {
            unsafe { std::ptr::copy_nonoverlapping(chunk.as_ptr(), s.ptr, chunk.len()) };
            let region = vk::BufferCopy { src_offset: 0, dst_offset: at, size: chunk.len() as u64 };
            inner.one_shot(|cmd| unsafe { inner.device.cmd_copy_buffer(cmd, s.buffer, buf.buffer, &[region]) })?;
            at += chunk.len() as u64;
        }
        Ok(())
    }

    fn read(&self, buf: &Buffer, off: u64, out: &mut [u8]) -> Result<()> {
        if off + out.len() as u64 > buf.size {
            return Err(Error::Invalid(format!("read of {} bytes at {off} is past a buffer of {}", out.len(), buf.size)));
        }
        if let Some(ptr) = buf.ptr {
            unsafe { std::ptr::copy_nonoverlapping(ptr.add(off as usize), out.as_mut_ptr(), out.len()) };
            return Ok(());
        }
        let inner = &self.inner;
        let staging = inner.staging_buffer(true)?;
        let s = staging.as_ref().unwrap();
        let mut at = off;
        for chunk in out.chunks_mut(STAGING_BYTES as usize) {
            let region = vk::BufferCopy { src_offset: at, dst_offset: 0, size: chunk.len() as u64 };
            inner.one_shot(|cmd| unsafe { inner.device.cmd_copy_buffer(cmd, buf.buffer, s.buffer, &[region]) })?;
            unsafe { std::ptr::copy_nonoverlapping(s.ptr, chunk.as_mut_ptr(), chunk.len()) };
            at += chunk.len() as u64;
        }
        Ok(())
    }

    fn zero(&self, buf: &Buffer, off: u64, len: u64) -> Result<()> {
        if off % 4 != 0 || len % 4 != 0 || off + len > buf.size {
            return Err(Error::Invalid(format!("zero of {len} bytes at {off} must be 4-byte aligned and inside the buffer ({})", buf.size)));
        }
        if len == 0 {
            return Ok(());
        }
        let inner = &self.inner;
        inner.one_shot(|cmd| unsafe { inner.device.cmd_fill_buffer(cmd, buf.buffer, off, len, 0) })
    }

    fn pipeline(&self, blob: &KernelBlob) -> Result<Pipeline> {
        let inner = &self.inner;
        let Code::Spirv(words) = &blob.code else {
            return Err(Error::Invalid(format!("the Vulkan backend takes SPIR-V blobs, `{}` is {:?}", blob.entry, blob.target)));
        };
        let n = blob.sig.bindings.len();
        let prefix = if blob.sig.offsets == OffsetMode::Prefix { 4 * n as u32 } else { 0 };
        let push_total = prefix + blob.sig.push_bytes;
        if blob.sig.push_bytes % 4 != 0 {
            return Err(Error::Invalid(format!("`{}`: push constants must be a multiple of 4 bytes", blob.entry)));
        }
        if n as u32 > inner.limits.max_bindings {
            return Err(Error::Invalid(format!("`{}` binds {n} buffers, the device allows {}", blob.entry, inner.limits.max_bindings)));
        }
        if push_total > inner.limits.max_push_bytes {
            return Err(Error::Invalid(format!(
                "`{}`: {} offset bytes + {} parameter bytes exceed the device's {} push constant bytes",
                blob.entry,
                prefix,
                blob.sig.push_bytes,
                inner.limits.max_push_bytes
            )));
        }
        let entry = std::ffi::CString::new(blob.entry.clone()).map_err(|_| Error::Invalid("entry name contains a NUL".into()))?;
        unsafe {
            let module = vk_err(inner.device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None), "create_shader_module")
                .map_err(|e| Error::Compile { lib: blob.entry.clone(), msg: e.to_string() })?;
            let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n as u32)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let set_layout = vk_err(
                inner.device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None),
                "create_descriptor_set_layout",
            )?;
            let layouts = [set_layout];
            let ranges = [vk::PushConstantRange { stage_flags: vk::ShaderStageFlags::COMPUTE, offset: 0, size: push_total }];
            let mut layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
            if push_total > 0 {
                layout_info = layout_info.push_constant_ranges(&ranges);
            }
            let layout = vk_err(inner.device.create_pipeline_layout(&layout_info, None), "create_pipeline_layout")?;

            let map: Vec<vk::SpecializationMapEntry> =
                (0..blob.spec.len() as u32).map(|i| vk::SpecializationMapEntry { constant_id: i, offset: 4 * i, size: 4 }).collect();
            let data: Vec<u8> = blob.spec.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let spec = vk::SpecializationInfo::default().map_entries(&map).data(&data);
            let mut stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(&entry);
            if !blob.spec.is_empty() {
                stage = stage.specialization_info(&spec);
            }
            let info = vk::ComputePipelineCreateInfo::default().stage(stage).layout(layout);
            let pipeline = inner
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[info], None)
                .map_err(|(_, e)| Error::Pipeline { entry: blob.entry.clone(), msg: e.to_string() })?[0];
            Ok(Pipeline { inner: inner.clone(), module, set_layout, layout, pipeline, bindings: n, offsets: blob.sig.offsets, label: if blob.label.is_empty() { blob.entry.clone() } else { blob.label.clone() } })
        }
    }

    fn submit(&self, groups: &[Vec<Launch<'_, Self>>]) -> Result<Pending> {
        self.submit_reading(groups, &[])
    }

    fn prefetched(&self, pending: &Pending, buffer: &Buffer, off: u64, out: &mut [u8]) -> bool {
        let Some(ctx) = &pending.ctx else { return false };
        let Some(rb) = &ctx.rb else { return false };
        for &(src, at, len, stage) in &pending.reads {
            if src == buffer.buffer && off >= at && off + out.len() as u64 <= at + len {
                unsafe { std::ptr::copy_nonoverlapping(rb.ptr.add((stage + off - at) as usize), out.as_mut_ptr(), out.len()) };
                return true;
            }
        }
        false
    }

    fn submit_reading(&self, groups: &[Vec<Launch<'_, Self>>], reads: &[(&Buffer, u64, u64)]) -> Result<Pending> {
        let inner = &self.inner;
        // where each requested range lands in the host-visible copy
        let mut total = 0u64;
        let plan_reads: Vec<(vk::Buffer, u64, u64, u64)> = reads
            .iter()
            .filter(|(b, off, len)| *len > 0 && off + len <= b.size)
            .map(|(b, off, len)| {
                let at = total;
                total += len.div_ceil(16) * 16;
                (b.buffer, *off, *len, at)
            })
            .collect();
        let launches: usize = groups.iter().map(Vec::len).sum();
        let descs: usize = groups.iter().flatten().map(|l| l.bindings.len()).sum();
        let align = inner.limits.storage_offset_align;

        // Recording ~900 dispatches costs more than most decode steps' CPU budget; a step that repeats the
        // last one exactly (same buffers, push constants, grids: only the data differs) replays it.
        let key: Option<Vec<u64>> = (!inner.profile).then(|| {
            let mut k = replay_key(groups);
            for &(b, off, len, _) in &plan_reads {
                k.extend([ash::vk::Handle::as_raw(b), off, len]);
            }
            k
        });
        if let Some(k) = &key {
            if let Some(ctx) = inner.take_replay(k) {
                let cmds = [ctx.cmd];
                let submit = vk::SubmitInfo::default().command_buffers(&cmds);
                let r = {
                    let q = inner.queue.lock().unwrap();
                    vk_err(unsafe { inner.device.queue_submit(*q, &[submit], ctx.fence) }, "queue_submit")
                };
                return match r {
                    Ok(()) => Ok(Pending { inner: inner.clone(), ctx: Some(ctx), prof: None, replay: key, reads: plan_reads }),
                    Err(e) => {
                        inner.recycle(ctx);
                        Err(e)
                    }
                };
            }
        }
        let mut ctx = inner.take_ctx()?;
        let mut prof: Option<Prof> = None;
        let result = (|| unsafe {
            // Descriptor pool big enough for every launch of this submission.
            if (ctx.dsets as usize) < launches || (ctx.ddescs as usize) < descs {
                if ctx.dpool != vk::DescriptorPool::null() {
                    inner.device.destroy_descriptor_pool(ctx.dpool, None);
                }
                let sets = (launches.max(1)).max(ctx.dsets as usize * 2).next_power_of_two() as u32;
                let ndesc = (descs.max(1)).max(ctx.ddescs as usize * 2).next_power_of_two() as u32;
                let sizes = [vk::DescriptorPoolSize { ty: vk::DescriptorType::STORAGE_BUFFER, descriptor_count: ndesc }];
                ctx.dpool = vk_err(
                    inner.device.create_descriptor_pool(&vk::DescriptorPoolCreateInfo::default().max_sets(sets).pool_sizes(&sizes), None),
                    "create_descriptor_pool",
                )?;
                ctx.dsets = sets;
                ctx.ddescs = ndesc;
            }
            vk_err(
                inner.device.begin_command_buffer(ctx.cmd, &vk::CommandBufferBeginInfo::default().flags(if key.is_some() { vk::CommandBufferUsageFlags::empty() } else { vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT })),
                "begin_command_buffer",
            )?;
            if inner.profile && launches > 0 {
                let pool = vk_err(
                    inner.device.create_query_pool(&vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(2 * launches as u32), None),
                    "create_query_pool",
                )?;
                inner.device.cmd_reset_query_pool(ctx.cmd, pool, 0, 2 * launches as u32);
                prof = Some(Prof { pool, labels: Vec::with_capacity(launches) });
            }
            // Orders this submission after earlier ones on the queue.
            inner.global_barrier(ctx.cmd);
            let mut bound_pipeline = vk::Pipeline::null();
            for (gi, group) in groups.iter().enumerate() {
                if gi > 0 {
                    inner.global_barrier(ctx.cmd);
                }
                for l in group {
                    let p = l.pipeline;
                    if l.bindings.len() != p.bindings {
                        return Err(Error::Invalid(format!("launch binds {} buffers, its pipeline takes {}", l.bindings.len(), p.bindings)));
                    }
                    let layouts = [p.set_layout];
                    let set = vk_err(
                        inner.device.allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo::default().descriptor_pool(ctx.dpool).set_layouts(&layouts)),
                        "allocate_descriptor_sets",
                    )?[0];
                    let mut infos: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(l.bindings.len());
                    let mut push: Vec<u8> = Vec::with_capacity(4 * l.bindings.len() + l.push.len());
                    for b in &l.bindings {
                        let aligned = b.offset - b.offset % align;
                        let residual = b.offset - aligned;
                        let range = (residual + b.len).max(4).min(b.buffer.size - aligned);
                        if range > inner.max_range {
                            return Err(Error::Invalid(format!("a binding of {range} bytes is over the device's {} byte storage buffer range", inner.max_range)));
                        }
                        infos.push(vk::DescriptorBufferInfo { buffer: b.buffer.buffer, offset: aligned, range });
                        if p.offsets == OffsetMode::Prefix {
                            push.extend_from_slice(&(residual as u32).to_ne_bytes());
                        }
                    }
                    push.extend_from_slice(l.push);
                    let writes: Vec<vk::WriteDescriptorSet> = infos
                        .iter()
                        .enumerate()
                        .map(|(i, info)| {
                            vk::WriteDescriptorSet::default()
                                .dst_set(set)
                                .dst_binding(i as u32)
                                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                                .buffer_info(std::slice::from_ref(info))
                        })
                        .collect();
                    inner.device.update_descriptor_sets(&writes, &[]);
                    if bound_pipeline != p.pipeline {
                        inner.device.cmd_bind_pipeline(ctx.cmd, vk::PipelineBindPoint::COMPUTE, p.pipeline);
                        bound_pipeline = p.pipeline;
                    }
                    inner.device.cmd_bind_descriptor_sets(ctx.cmd, vk::PipelineBindPoint::COMPUTE, p.layout, 0, &[set], &[]);
                    if !push.is_empty() {
                        inner.device.cmd_push_constants(ctx.cmd, p.layout, vk::ShaderStageFlags::COMPUTE, 0, &push);
                    }
                    if let Some(pr) = prof.as_mut() {
                        let q = 2 * pr.labels.len() as u32;
                        inner.device.cmd_write_timestamp(ctx.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, pr.pool, q);
                        inner.device.cmd_dispatch(ctx.cmd, l.grid[0], l.grid[1], l.grid[2]);
                        inner.device.cmd_write_timestamp(ctx.cmd, vk::PipelineStageFlags::BOTTOM_OF_PIPE, pr.pool, q + 1);
                        pr.labels.push(p.label.clone());
                    } else {
                        inner.device.cmd_dispatch(ctx.cmd, l.grid[0], l.grid[1], l.grid[2]);
                    }
                }
            }
            if !plan_reads.is_empty() {
                if ctx.rb.as_ref().map_or(true, |r| r.size < total) {
                    if let Some(old) = ctx.rb.take() {
                        inner.device.destroy_buffer(old.buffer, None);
                        inner.device.free_memory(old.memory, None);
                    }
                    ctx.rb = Some(inner.make_staging(total.max(4096), true)?);
                }
                let rb = ctx.rb.as_ref().unwrap();
                inner.global_barrier(ctx.cmd);
                for &(src, off, len, at) in &plan_reads {
                    inner.device.cmd_copy_buffer(ctx.cmd, src, rb.buffer, &[vk::BufferCopy { src_offset: off, dst_offset: at, size: len }]);
                }
                let to_host = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::TRANSFER_WRITE).dst_access_mask(vk::AccessFlags::HOST_READ);
                inner.device.cmd_pipeline_barrier(ctx.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::HOST, vk::DependencyFlags::empty(), &[to_host], &[], &[]);
            }
            vk_err(inner.device.end_command_buffer(ctx.cmd), "end_command_buffer")?;
            let cmds = [ctx.cmd];
            let submit = vk::SubmitInfo::default().command_buffers(&cmds);
            let q = inner.queue.lock().unwrap();
            vk_err(inner.device.queue_submit(*q, &[submit], ctx.fence), "queue_submit")
        })();
        match result {
            Ok(()) => Ok(Pending { inner: inner.clone(), ctx: Some(ctx), prof, replay: key, reads: plan_reads }),
            Err(e) => {
                if let Some(p) = prof {
                    unsafe { inner.device.destroy_query_pool(p.pool, None) };
                }
                inner.recycle(ctx);
                Err(e)
            }
        }
    }

    fn poll(&self, pending: &Pending) -> Result<bool> {
        match &pending.ctx {
            Some(ctx) => vk_err(unsafe { self.inner.device.get_fence_status(ctx.fence) }, "get_fence_status"),
            None => Ok(true),
        }
    }

    fn wait(&self, pending: &Pending) -> Result<()> {
        match &pending.ctx {
            Some(ctx) => vk_err(unsafe { self.inner.device.wait_for_fences(&[ctx.fence], true, u64::MAX) }, "wait_for_fences"),
            None => Ok(()),
        }
    }
}
