//! The capability implementations: Server, Model, Session, Run, Upload and TensorRef.
use std::{
    cell::{Cell, RefCell},
    collections::{HashMap, VecDeque},
    path::PathBuf,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use capnp_rpc::{RpcSystem, rpc_twoparty_capnp::Side, twoparty};
use ndarray::ArrayD;
use tensorlisp::{Chunk, Device, Model, ModelState, RawInput, RawKind, RunOptions, Value};
use tokio::{
    sync::{Notify, Semaphore, watch},
    task::spawn_blocking,
};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::{
    convert,
    store::{Incoming, Store},
    tensorlisp_capnp::{
        DType, RunStatus, callbacks, model, model_info, named, param, pipeline_info, run, server as server_capnp, server::upload, session, sink, source, tensor_ref,
        type_spec, value,
    },
};

type CResult<T> = Result<T, capnp::Error>;

fn failed(e: impl std::fmt::Display) -> capnp::Error {
    capnp::Error::failed(format!("{e:#}"))
}

fn parse_device(name: &str) -> CResult<Option<Device>> {
    Ok(Some(match name {
        "" => return Ok(None),
        "auto" => Device::Auto,
        "cpu" => Device::Cpu,
        "gpu" => Device::Gpu,
        "native" => Device::Native,
        other => return Err(failed(format!("unknown device {other:?} (expected auto, cpu, gpu or native)"))),
    }))
}

/// Installed models and the ones currently loaded.
pub struct Registry {
    store: RefCell<Store>,
    device: Device,
    loaded: RefCell<HashMap<String, Model>>,
    load_lock: tokio::sync::Mutex<()>,
    /// Live sessions by token, to turn a Session capability that comes back into its state.
    sessions: RefCell<HashMap<u64, std::rc::Weak<SessionInner>>>,
    token_counter: Cell<u64>,
}

impl Registry {
    pub fn new(store: Store, device: Device) -> Rc<Registry> {
        Rc::new(Registry {
            store: RefCell::new(store),
            device,
            loaded: Default::default(),
            load_lock: Default::default(),
            sessions: Default::default(),
            token_counter: Cell::new(0),
        })
    }

    /// The loaded model, loading it first (on `device`, else the server's default).
    async fn model(&self, hash: &str, device: Option<Device>) -> anyhow::Result<Model> {
        if let Some(m) = self.loaded.borrow().get(hash) {
            return Ok(m.clone());
        }
        let _guard = self.load_lock.lock().await;
        if let Some(m) = self.loaded.borrow().get(hash) {
            return Ok(m.clone());
        }
        let path = self.store.borrow().path(hash);
        let device = device.unwrap_or(self.device);
        let model = spawn_blocking(move || Model::load(path, device)).await??;
        self.loaded.borrow_mut().insert(hash.to_owned(), model.clone());
        Ok(model)
    }

    /// Registers a session under a fresh unguessable token.
    fn add_session(&self, state: ModelState) -> Rc<SessionInner> {
        use std::hash::{BuildHasher, Hasher};
        self.token_counter.set(self.token_counter.get() + 1);
        let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
        hasher.write_u64(self.token_counter.get());
        let token = hasher.finish();
        let inner = Rc::new(SessionInner { state: Some(state), token });
        let mut sessions = self.sessions.borrow_mut();
        sessions.retain(|_, s| s.strong_count() > 0);
        sessions.insert(token, Rc::downgrade(&inner));
        inner
    }

    /// The state behind a Session capability the client passed back.
    async fn session_state(&self, session: session::Client) -> CResult<ModelState> {
        let reply = session.token_request().send().promise.await?;
        let token = reply.get()?.get_token();
        let found = self.sessions.borrow().get(&token).and_then(std::rc::Weak::upgrade);
        found.and_then(|s| s.state.clone()).ok_or_else(|| failed("that session is not one of this server's"))
    }

    fn model_client(self: &Rc<Self>, hash: String) -> model::Client {
        capnp_rpc::new_client(ModelImpl { reg: self.clone(), hash })
    }
}

/// The `Server` capability.
pub(crate) fn server_client(registry: Rc<Registry>) -> server_capnp::Client {
    capnp_rpc::new_client(ServerImpl { reg: registry })
}

/// The bootstrap capability.
pub fn bootstrap(registry: Rc<Registry>) -> capnp::capability::Client {
    server_client(registry).client
}

impl Registry {
    /// What this server tells a scheduler about itself when it registers.
    pub(crate) fn fill_worker_info(&self, mut info: crate::tensorlisp_capnp::worker_info::Builder<'_>, name: &str) {
        info.set_name(name);
        info.set_version(env!("CARGO_PKG_VERSION"));
        let mut devices = info.reborrow().init_devices(1);
        let mut d = devices.reborrow().get(0);
        d.set_name(format!("{:?}", self.device).to_lowercase().as_str());
        d.set_native(matches!(self.device, Device::Native));
        let hashes = self.store.borrow().hashes();
        let mut models = info.init_models(hashes.len() as u32);
        for (i, hash) in hashes.iter().enumerate() {
            let loaded = self.loaded.borrow().get(hash).cloned();
            fill_info(models.reborrow().get(i as u32), self, hash, loaded.as_ref());
        }
    }
}

/// Serves one connection on the current `LocalSet`.
pub fn serve_stream<S>(registry: Rc<Registry>, stream: S)
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + 'static,
{
    let (reader, writer) = tokio::io::split(stream);
    let network = twoparty::VatNetwork::new(reader.compat(), writer.compat_write(), Side::Server, Default::default());
    let rpc = RpcSystem::new(Box::new(network), Some(bootstrap(registry)));
    tokio::task::spawn_local(async move {
        if let Err(e) = rpc.await {
            eprintln!("connection closed with an error: {e}");
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------------------------

enum Out {
    Array(ArrayD<f32>),
    Text(String),
}

async fn fetch_ref(client: tensor_ref::Client) -> CResult<ArrayD<f32>> {
    let info = client.info_request().send().promise.await?;
    let info = info.get()?;
    let dims: Vec<u64> = info.get_dims()?.iter().collect();
    let dtype = info.get_dtype()?;
    let data = client.read_request().send().promise.await?;
    convert::array_from_parts(dtype, &dims, data.get()?.get_data()?)
}

async fn value_array(v: value::Reader<'_>) -> CResult<ArrayD<f32>> {
    match v.which()? {
        value::Which::Tensor(t) => convert::read_tensor(t?),
        value::Which::TensorRef(r) => fetch_ref(r?).await,
        _ => Err(failed("expected a tensor")),
    }
}

async fn named_arrays(list: capnp::struct_list::Reader<'_, named::Owned>) -> CResult<Vec<(String, ArrayD<f32>)>> {
    let mut out = Vec::with_capacity(list.len() as usize);
    for item in list.iter() {
        let name = item.get_name()?.to_str()?.to_owned();
        let array = value_array(item.get_value()?).await.map_err(|e| failed(format!("input {name:?}: {}", e.extra)))?;
        out.push((name, array));
    }
    Ok(out)
}

fn image_from(img: crate::tensorlisp_capnp::image::Reader<'_>) -> CResult<image::DynamicImage> {
    use crate::tensorlisp_capnp::image::format::Which as F;
    let (w, h) = (img.get_width(), img.get_height());
    let pixels = img.get_pixels()?;
    let raw = |channels: usize| {
        if pixels.len() != w as usize * h as usize * channels {
            return Err(failed(format!("a {w}x{h} image with {channels} channels needs {} bytes, got {}", w as usize * h as usize * channels, pixels.len())));
        }
        Ok(pixels.to_vec())
    };
    Ok(match img.get_format().which()? {
        F::Rgb8(()) => image::DynamicImage::ImageRgb8(image::RgbImage::from_raw(w, h, raw(3)?).unwrap()),
        F::Rgba8(()) => image::DynamicImage::ImageRgba8(image::RgbaImage::from_raw(w, h, raw(4)?).unwrap()),
        F::Gray8(()) => image::DynamicImage::ImageLuma8(image::GrayImage::from_raw(w, h, raw(1)?).unwrap()),
        F::Encoded(_) => image::load_from_memory(pixels).map_err(failed)?,
    })
}

fn audio_from(a: crate::tensorlisp_capnp::audio::Reader<'_>) -> CResult<tensorlisp::autopro::audio::Audio> {
    let channels = a.get_channels().max(1) as usize;
    let samples = a.get_samples()?;
    let mono: Vec<f32> = if channels == 1 {
        samples.iter().collect()
    } else {
        let all: Vec<f32> = samples.iter().collect();
        all.chunks(channels).map(|f| f.iter().sum::<f32>() / f.len() as f32).collect()
    };
    Ok(tensorlisp::autopro::audio::Audio::new(mono, a.get_sample_rate()))
}

async fn raw_input(kind: RawKind, v: value::Reader<'_>) -> CResult<RawInput> {
    Ok(match (kind, v.which()?) {
        (RawKind::Text, value::Which::Text(t)) => RawInput::Text(t?.to_str()?.to_owned()),
        (RawKind::Image, value::Which::Image(i)) => RawInput::Image(image_from(i?)?),
        (RawKind::Audio, value::Which::Audio(a)) => RawInput::Audio(audio_from(a?)?),
        (RawKind::Array, _) => RawInput::Array(value_array(v).await?),
        (kind, _) => return Err(failed(format!("expected a {kind:?} value"))),
    })
}

fn write_outputs(
    list: capnp::struct_list::Builder<'_, named::Owned>,
    outs: &[(String, Out)],
    dtype: DType,
    by_ref: bool,
) -> CResult<()> {
    let mut list = list;
    for (i, (name, out)) in outs.iter().enumerate() {
        let mut entry = list.reborrow().get(i as u32);
        entry.set_name(name.as_str());
        let mut v = entry.init_value();
        match out {
            Out::Text(t) => v.set_text(t.as_str()),
            Out::Array(a) if by_ref => {
                let values: Vec<f32> = a.iter().copied().collect();
                let r = TensorRefImpl {
                    dtype,
                    dims: a.shape().iter().map(|&d| d as u64).collect(),
                    bytes: convert::encode(dtype, &values)?,
                };
                v.set_tensor_ref(capnp_rpc::new_client(r));
            }
            Out::Array(a) => convert::write_tensor(v.init_tensor(), a, dtype)?,
        }
    }
    Ok(())
}


/// A stream chunk sent by the client. Only values that are complete in the message are accepted
/// (no `tensorRef`), so pushes are handled in the order they were made.
fn chunk_from(v: value::Reader<'_>) -> CResult<Chunk> {
    Ok(match v.which()? {
        value::Which::Tensor(t) => Chunk::Array(convert::read_tensor(t?)?),
        value::Which::Text(t) => Chunk::Text(t?.to_str()?.to_owned()),
        value::Which::Audio(a) => Chunk::Audio(audio_from(a?)?),
        value::Which::Image(i) => Chunk::Image(image_from(i?)?.to_rgb8()),
        _ => return Err(failed("stream chunks are tensors, text, audio or images")),
    })
}

fn write_chunk(mut v: value::Builder<'_>, chunk: &Chunk, dtype: DType) -> CResult<()> {
    match chunk {
        Chunk::Array(a) => convert::write_tensor(v.init_tensor(), a, dtype)?,
        Chunk::Text(t) => v.set_text(t.as_str()),
        Chunk::Audio(a) => {
            let mut out = v.init_audio();
            out.set_sample_rate(a.sample_rate);
            out.set_channels(1);
            let mut samples = out.init_samples(a.samples.len() as u32);
            for (i, &x) in a.samples.iter().enumerate() {
                samples.set(i as u32, x);
            }
        }
        Chunk::Image(img) => {
            let mut out = v.init_image();
            out.set_width(img.width());
            out.set_height(img.height());
            out.reborrow().init_format().set_rgb8(());
            out.set_pixels(img.as_raw());
        }
    }
    Ok(())
}

/// How many chunks a client may have pushed that the pipeline hasn't taken yet.
const INPUT_WINDOW: usize = 32;

/// The client's end of one input stream of a pipeline.
struct StreamIn {
    tx: RefCell<Option<tensorlisp::stream::StreamSender>>,
    window: Arc<Semaphore>,
}

impl StreamIn {
    fn close(&self) {
        self.tx.borrow_mut().take();
    }
}

struct SinkImpl {
    stream: Rc<StreamIn>,
}

impl sink::Server for SinkImpl {
    async fn push(self: Rc<Self>, params: sink::PushParams) -> CResult<()> {
        // Waiting for a window slot comes first and is first come, first served; everything after
        // it is synchronous, so chunks reach the pipeline in the order they were pushed.
        let permit = self.stream.window.clone().acquire_owned().await.map_err(failed)?;
        let chunk = chunk_from(params.get()?.get_chunk()?)?;
        let tx = self.stream.tx.borrow();
        let tx = tx.as_ref().ok_or_else(|| failed("the stream is closed"))?;
        tx.send(chunk).map_err(|_| failed("the pipeline no longer reads this stream"))?;
        permit.forget();
        Ok(())
    }

    async fn done(self: Rc<Self>, _: sink::DoneParams, _: sink::DoneResults) -> CResult<()> {
        self.stream.close();
        Ok(())
    }
}

/// What a pipeline emitted under one name, waiting to be pulled.
#[derive(Default)]
struct Queue {
    items: RefCell<VecDeque<Chunk>>,
    wake: Notify,
}

/// Output streams of a run. `finished` is set once the pipeline is done and everything it emitted
/// has been queued.
#[derive(Default)]
struct Outputs {
    queues: RefCell<HashMap<String, Rc<Queue>>>,
    finished: Cell<bool>,
}

impl Outputs {
    fn queue(&self, name: &str) -> Rc<Queue> {
        self.queues.borrow_mut().entry(name.to_owned()).or_default().clone()
    }

    fn finish(&self) {
        self.finished.set(true);
        for q in self.queues.borrow().values() {
            q.wake.notify_waiters();
        }
    }
}

struct SourceImpl {
    queue: Rc<Queue>,
    run: Rc<RunState>,
}

impl source::Server for SourceImpl {
    async fn pull(self: Rc<Self>, params: source::PullParams, mut results: source::PullResults) -> CResult<()> {
        let max = params.get()?.get_max().max(1) as usize;
        loop {
            let woken = self.queue.wake.notified();
            tokio::pin!(woken);
            woken.as_mut().enable();
            let taken: Vec<Chunk> = {
                let mut items = self.queue.items.borrow_mut();
                let n = items.len().min(max);
                items.drain(..n).collect()
            };
            if !taken.is_empty() {
                let mut r = results.get();
                let mut list = r.reborrow().init_chunks(taken.len() as u32);
                for (i, chunk) in taken.iter().enumerate() {
                    write_chunk(list.reborrow().get(i as u32), chunk, self.run.dtype)?;
                }
                r.set_eof(false);
                return Ok(());
            }
            if self.run.outputs.finished.get() {
                if let Phase::Failed(e) = &*self.run.rx.borrow() {
                    return Err(failed(e));
                }
                results.get().set_eof(true);
                return Ok(());
            }
            woken.await;
        }
    }

    async fn cancel(self: Rc<Self>, _: source::CancelParams, _: source::CancelResults) -> CResult<()> {
        self.queue.items.borrow_mut().clear();
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// TensorRef
// ---------------------------------------------------------------------------------------------

/// A result held on the server (host memory for now) until the client drops it.
struct TensorRefImpl {
    dtype: DType,
    dims: Vec<u64>,
    bytes: Vec<u8>,
}

impl tensor_ref::Server for TensorRefImpl {
    async fn info(self: Rc<Self>, _: tensor_ref::InfoParams, mut results: tensor_ref::InfoResults) -> CResult<()> {
        let mut r = results.get();
        r.set_dtype(self.dtype);
        let mut dims = r.reborrow().init_dims(self.dims.len() as u32);
        for (i, &d) in self.dims.iter().enumerate() {
            dims.set(i as u32, d);
        }
        r.set_device("host");
        Ok(())
    }

    async fn read(self: Rc<Self>, params: tensor_ref::ReadParams, mut results: tensor_ref::ReadResults) -> CResult<()> {
        let p = params.get()?;
        let len = self.bytes.len() as u64;
        let offset = p.get_offset().min(len);
        let end = if p.get_length() == 0 { len } else { (offset + p.get_length()).min(len) };
        results.get().set_data(&self.bytes[offset as usize..end as usize]);
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------------------------

struct ServerImpl {
    reg: Rc<Registry>,
}

struct Install {
    name: String,
    device: Option<Device>,
    load: bool,
    replace: bool,
}

fn install_options(o: crate::tensorlisp_capnp::install_options::Reader<'_>) -> CResult<Install> {
    if !o.get_pipeline_source()?.is_empty() {
        return Err(capnp::Error::unimplemented("pipelineSource: replacing a model's pipelines on install".into()));
    }
    Ok(Install {
        name: o.get_name()?.to_str()?.to_owned(),
        device: parse_device(o.get_device()?.to_str()?)?,
        load: o.get_load(),
        replace: o.get_replace(),
    })
}

/// Names the stored file `hash` and loads it if asked; on a load failure a model that was just
/// added is removed again.
async fn register(reg: &Rc<Registry>, hash: String, was_new: bool, opts: Install) -> CResult<model::Client> {
    if !opts.name.is_empty() {
        reg.store.borrow_mut().alias(&opts.name, &hash, opts.replace).map_err(failed)?;
    }
    if opts.load
        && let Err(e) = reg.model(&hash, opts.device).await
    {
        if was_new {
            let _ = reg.store.borrow_mut().remove(&hash);
        }
        return Err(failed(format!("loading the model: {e:#}")));
    }
    Ok(reg.model_client(hash))
}

impl server_capnp::Server for ServerImpl {
    async fn info(self: Rc<Self>, _: server_capnp::InfoParams, mut results: server_capnp::InfoResults) -> CResult<()> {
        let mut r = results.get();
        r.set_version(env!("CARGO_PKG_VERSION"));
        let mut devices = r.init_devices(1);
        let mut d = devices.reborrow().get(0);
        d.set_name(format!("{:?}", self.reg.device).to_lowercase().as_str());
        d.set_native(matches!(self.reg.device, Device::Native));
        Ok(())
    }

    async fn list(self: Rc<Self>, _: server_capnp::ListParams, mut results: server_capnp::ListResults) -> CResult<()> {
        let hashes = self.reg.store.borrow().hashes();
        let mut models = results.get().init_models(hashes.len() as u32);
        for (i, hash) in hashes.iter().enumerate() {
            let loaded = self.reg.loaded.borrow().get(hash).cloned();
            fill_info(models.reborrow().get(i as u32), &self.reg, hash, loaded.as_ref());
        }
        Ok(())
    }

    async fn install(
        self: Rc<Self>,
        params: server_capnp::InstallParams,
        mut results: server_capnp::InstallResults,
    ) -> CResult<()> {
        let options = install_options(params.get()?.get_options()?)?;
        let incoming = self.reg.store.borrow().begin().map_err(failed)?;
        let up = UploadImpl { reg: self.reg.clone(), options: RefCell::new(Some(options)), incoming: RefCell::new(Some(incoming)) };
        results.get().set_upload(capnp_rpc::new_client(up));
        Ok(())
    }

    async fn install_from(
        self: Rc<Self>,
        params: server_capnp::InstallFromParams,
        mut results: server_capnp::InstallFromResults,
    ) -> CResult<()> {
        let p = params.get()?;
        let location = p.get_location()?.to_str()?.to_owned();
        if location.starts_with("http://") || location.starts_with("https://") {
            return Err(capnp::Error::unimplemented("installing from a URL".into()));
        }
        let options = install_options(p.get_options()?)?;
        let path = PathBuf::from(location.strip_prefix("file://").unwrap_or(&location));
        let store = self.reg.store.borrow().clone();
        let (hash, new) = spawn_blocking(move || store.import(&path)).await.map_err(failed)?.map_err(failed)?;
        results.get().set_model(register(&self.reg, hash, new, options).await?);
        Ok(())
    }

    async fn open(self: Rc<Self>, params: server_capnp::OpenParams, mut results: server_capnp::OpenResults) -> CResult<()> {
        let name = params.get()?.get_name()?.to_str()?.to_owned();
        let hash = self.reg.store.borrow().resolve(&name).map_err(failed)?;
        results.get().set_model(self.reg.model_client(hash));
        Ok(())
    }

    async fn remove(self: Rc<Self>, params: server_capnp::RemoveParams, _: server_capnp::RemoveResults) -> CResult<()> {
        let name = params.get()?.get_name()?.to_str()?.to_owned();
        let (hash, deleted) = self.reg.store.borrow_mut().remove(&name).map_err(failed)?;
        if deleted {
            self.reg.loaded.borrow_mut().remove(&hash);
        }
        Ok(())
    }
}

struct UploadImpl {
    reg: Rc<Registry>,
    options: RefCell<Option<Install>>,
    incoming: RefCell<Option<Incoming>>,
}

impl upload::Server for UploadImpl {
    async fn push(self: Rc<Self>, params: upload::PushParams) -> CResult<()> {
        let chunk = params.get()?.get_chunk()?;
        match self.incoming.borrow_mut().as_mut() {
            Some(incoming) => incoming.write(chunk).map_err(failed),
            None => Err(failed("the upload is finished")),
        }
    }

    async fn finish(self: Rc<Self>, _: upload::FinishParams, mut results: upload::FinishResults) -> CResult<()> {
        let incoming = self.incoming.borrow_mut().take().ok_or_else(|| failed("the upload is finished"))?;
        let options = self.options.borrow_mut().take().unwrap();
        let (hash, new) = self.reg.store.borrow().commit(incoming).map_err(failed)?;
        results.get().set_model(register(&self.reg, hash, new, options).await?);
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------------------------

struct ModelImpl {
    reg: Rc<Registry>,
    hash: String,
}

fn set_tensor_type(mut t: type_spec::Builder<'_>, dtype: &str, dims: Option<&[Option<i64>]>) {
    let mut g = t.reborrow().init_tensor();
    g.set_dtype(if dtype == "i32" { DType::I32 } else { DType::F32 });
    if let Some(dims) = dims {
        // The program lists dims innermost first (ggml order); the wire is outermost first.
        let mut list = g.init_dims(dims.len() as u32);
        for (i, d) in dims.iter().rev().enumerate() {
            list.reborrow().get(i as u32).set_size(d.unwrap_or(-1));
        }
    }
}

fn set_param(mut p: param::Builder<'_>, name: &str, kind: Option<RawKind>, input: Option<&tensorlisp::InputSpec>) {
    p.set_name(name);
    let mut ty = p.init_type();
    match (kind, input) {
        (Some(RawKind::Text), _) => ty.set_text(()),
        (Some(RawKind::Image), _) => ty.set_image(()),
        (Some(RawKind::Audio), _) => ty.set_audio(()),
        (Some(RawKind::Stream), _) => ty.init_stream().set_any(()),
        (_, Some(i)) => set_tensor_type(ty, &i.dtype, i.dims.as_deref()),
        _ => set_tensor_type(ty, "f32", None),
    }
}

fn fill_info(mut info: model_info::Builder<'_>, reg: &Registry, hash: &str, model: Option<&Model>) {
    info.set_id(hash);
    info.set_name(reg.store.borrow().name_of(hash).as_str());
    info.set_weight_bytes(std::fs::metadata(reg.store.borrow().path(hash)).map(|m| m.len()).unwrap_or(0));
    info.set_loaded(model.is_some());
    let Some(model) = model else { return };
    let mut state = info.reborrow().init_state(model.states().len() as u32);
    for (i, spec) in model.states().iter().enumerate() {
        state_info(spec, state.reborrow().get(i as u32));
    }
    let mut entries = info.reborrow().init_entries(model.entries().len() as u32);
    for (i, spec) in model.entries().iter().enumerate() {
        let mut e = entries.reborrow().get(i as u32);
        e.set_name(spec.name.as_str());
        let mut inputs = e.init_inputs(spec.inputs.len() as u32);
        for (j, input) in spec.inputs.iter().enumerate() {
            set_param(inputs.reborrow().get(j as u32), &input.name, None, Some(input));
        }
    }
    let mut pipelines = info.init_pipelines(model.pipelines().len() as u32);
    for (i, spec) in model.pipelines().iter().enumerate() {
        let mut p: pipeline_info::Builder<'_> = pipelines.reborrow().get(i as u32);
        p.set_name(spec.name.as_str());
        p.set_streaming_in(spec.raw_inputs.iter().any(|r| r.kind == RawKind::Stream));
        let mut inputs = p.init_inputs(spec.raw_inputs.len() as u32);
        for (j, raw) in spec.raw_inputs.iter().enumerate() {
            set_param(inputs.reborrow().get(j as u32), &raw.name, Some(raw.kind), None);
        }
    }
}

impl model::Server for ModelImpl {
    async fn describe(self: Rc<Self>, _: model::DescribeParams, mut results: model::DescribeResults) -> CResult<()> {
        let m = self.reg.model(&self.hash, None).await.map_err(failed)?;
        fill_info(results.get().init_info(), &self.reg, &self.hash, Some(&m));
        Ok(())
    }

    async fn load(self: Rc<Self>, params: model::LoadParams, _: model::LoadResults) -> CResult<()> {
        let device = parse_device(params.get()?.get_device()?.to_str()?)?;
        self.reg.model(&self.hash, device).await.map(|_| ()).map_err(failed)
    }

    async fn unload(self: Rc<Self>, _: model::UnloadParams, _: model::UnloadResults) -> CResult<()> {
        self.reg.loaded.borrow_mut().remove(&self.hash);
        Ok(())
    }

    async fn run(self: Rc<Self>, params: model::RunParams, mut results: model::RunResults) -> CResult<()> {
        let p = params.get()?;
        let entry = p.get_entry()?.to_str()?.to_owned();
        let options = p.get_options()?;
        let (dtype, by_ref) = (options.get_output_dtype()?, options.get_outputs_by_ref());
        let inputs = named_arrays(p.get_inputs()?).await?;
        let state = if p.has_session() { Some(self.reg.session_state(p.get_session()?).await?) } else { None };
        let model = self.reg.model(&self.hash, None).await.map_err(failed)?;
        let output = spawn_blocking(move || {
            let views: Vec<(&str, _)> = inputs.iter().map(|(n, a)| (n.as_str(), a.view())).collect();
            model.run_entry(&entry, &views, &RunOptions { state, ..Default::default() })
        })
        .await
        .map_err(failed)?
        .map_err(failed)?;
        let outs: Vec<(String, Out)> = output.outputs.into_iter().map(|(n, a)| (n, Out::Array(a))).collect();
        write_outputs(results.get().init_outputs(outs.len() as u32), &outs, dtype, by_ref)
    }

    async fn session(self: Rc<Self>, _: model::SessionParams, mut results: model::SessionResults) -> CResult<()> {
        let model = self.reg.model(&self.hash, None).await.map_err(failed)?;
        let created = model.clone();
        let state = spawn_blocking(move || created.new_state()).await.map_err(failed)?.map_err(failed)?;
        let inner = self.reg.add_session(state);
        results.get().set_session(capnp_rpc::new_client(SessionImpl { reg: self.reg.clone(), model, inner }));
        Ok(())
    }

    async fn pipeline(self: Rc<Self>, params: model::PipelineParams, mut results: model::PipelineResults) -> CResult<()> {
        let p = params.get()?;
        let name = p.get_name()?.to_str()?.to_owned();
        let options = p.get_options()?;
        let (dtype, by_ref) = (options.get_output_dtype()?, options.get_outputs_by_ref());
        let model = self.reg.model(&self.hash, None).await.map_err(failed)?;
        let spec = model.pipelines().iter().find(|s| s.name == name).ok_or_else(|| {
            let names: Vec<&str> = model.pipelines().iter().map(|s| s.name.as_str()).collect();
            failed(format!("no pipeline {name:?}; pipelines: {}", names.join(", ")))
        })?;

        let args = p.get_args()?;
        let mut example = Vec::with_capacity(spec.raw_inputs.len());
        let mut inputs = HashMap::new();
        for raw in &spec.raw_inputs {
            if raw.kind == RawKind::Stream {
                // Fed by the client through Run.input(name).
                let window = Arc::new(Semaphore::new(INPUT_WINDOW));
                let released = window.clone();
                let (tx, reader) = tensorlisp::input_stream(move || released.add_permits(1));
                inputs.insert(raw.name.clone(), Rc::new(StreamIn { tx: RefCell::new(Some(tx)), window }));
                example.push((raw.name.clone(), RawInput::Stream(reader)));
                continue;
            }
            let arg = args
                .iter()
                .find(|a| a.get_name().ok().and_then(|n| n.to_str().ok()) == Some(raw.name.as_str()))
                .ok_or_else(|| failed(format!("pipeline {name:?} needs an argument {:?}", raw.name)))?;
            let input = raw_input(raw.kind, arg.get_value()?).await.map_err(|e| failed(format!("argument {:?}: {}", raw.name, e.extra)))?;
            example.push((raw.name.clone(), input));
        }
        let session = if p.has_session() { Some(self.reg.session_state(p.get_session()?).await?) } else { None };
        let timeout = options.get_timeout_millis();

        // What the pipeline emits goes through a channel to a task that sorts it into queues.
        let (emit_tx, mut emit_rx) = tokio::sync::mpsc::unbounded_channel::<(String, Chunk)>();
        let abandoned = Arc::new(AtomicBool::new(false));
        let emit: tensorlisp::Emit = {
            let abandoned = abandoned.clone();
            Arc::new(move |name, chunk| !abandoned.load(Ordering::Relaxed) && emit_tx.send((name.to_owned(), chunk)).is_ok())
        };

        let (tx, rx) = watch::channel(Phase::Queued);
        let state = Rc::new(RunState { tx, rx, dtype, by_ref, outputs: Outputs::default(), callbacks: RefCell::new(None) });
        if p.has_callbacks() {
            *state.callbacks.borrow_mut() = Some(p.get_callbacks()?);
        }
        let (cb_tx, mut cb_rx) = tokio::sync::mpsc::unbounded_channel::<CallbackRequest>();
        let handler: Arc<dyn tensorlisp::Callbacks> = Arc::new(ClientCallbacks { tx: cb_tx });
        let serving = state.clone();
        tokio::task::spawn_local(async move {
            while let Some(request) = cb_rx.recv().await {
                serve_callback(&serving, request).await;
            }
        });
        let forward = state.clone();
        tokio::task::spawn_local(async move {
            while let Some((name, chunk)) = emit_rx.recv().await {
                let queue = forward.outputs.queue(&name);
                queue.items.borrow_mut().push_back(chunk);
                queue.wake.notify_waiters();
            }
            forward.outputs.finish();
        });
        let task = state.clone();
        tokio::task::spawn_local(async move {
            task.set(Phase::Running, |p| matches!(p, Phase::Queued));
            let done = spawn_blocking(move || {
                let options = tensorlisp::PipelineOptions { state: session, emit: Some(emit), callbacks: Some(handler) };
                model.pipeline_with(&name, example, options)
            })
            .await;
            let phase = match done {
                Ok(Ok(values)) => Phase::Done(Rc::new(
                    values
                        .into_iter()
                        .map(|(n, v)| (n, match v {
                            Value::Array(a) => Out::Array(a),
                            Value::Text(t) => Out::Text(t),
                        }))
                        .collect(),
                )),
                Ok(Err(e)) => Phase::Failed(format!("{e:#}")),
                Err(e) => Phase::Failed(format!("pipeline task: {e}")),
            };
            task.set(phase, |p| !p.terminal());
        });
        let run = RunImpl { state, inputs, abandoned };
        if timeout > 0 {
            // The pipeline can't be interrupted, but closing its inputs ends a streaming one.
            let state = run.state.clone();
            let inputs: Vec<Rc<StreamIn>> = run.inputs.values().cloned().collect();
            tokio::task::spawn_local(async move {
                tokio::time::sleep(std::time::Duration::from_millis(timeout as u64)).await;
                if !state.rx.borrow().terminal() {
                    state.set(Phase::Failed(format!("timed out after {timeout} ms")), |p| !p.terminal());
                    inputs.iter().for_each(|s| s.close());
                }
            });
        }
        results.get().set_run(capnp_rpc::new_client(run));
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------------------------

/// A session's state, freed on a blocking thread: dropping it waits for the model's lock, which a
/// running entry holds.
struct SessionInner {
    state: Option<ModelState>,
    token: u64,
}

impl Drop for SessionInner {
    fn drop(&mut self) {
        if let Some(state) = self.state.take() {
            match tokio::runtime::Handle::try_current() {
                Ok(handle) => {
                    handle.spawn_blocking(move || drop(state));
                }
                Err(_) => drop(state),
            }
        }
    }
}

/// The model's `(define-state ...)` tensors for one client (KV caches, encoder context), separate
/// from every other session's.
struct SessionImpl {
    reg: Rc<Registry>,
    model: Model,
    inner: Rc<SessionInner>,
}

fn state_info(spec: &tensorlisp::StateSpec, mut out: crate::tensorlisp_capnp::state_info::Builder<'_>) {
    out.set_name(spec.name.as_str());
    let f16 = spec.dtype == "f16";
    out.set_dtype(if f16 { DType::F16 } else { DType::F32 });
    // Program dims are innermost first; the wire is outermost first.
    let mut dims = out.reborrow().init_dims(spec.dims.len() as u32);
    for (i, &d) in spec.dims.iter().rev().enumerate() {
        dims.set(i as u32, d);
    }
    out.set_bytes(spec.dims.iter().product::<i64>() as u64 * if f16 { 2 } else { 4 });
}

impl session::Server for SessionImpl {
    async fn reset(self: Rc<Self>, params: session::ResetParams, _: session::ResetResults) -> CResult<()> {
        if params.get()?.get_names()?.len() > 0 {
            return Err(capnp::Error::unimplemented("resetting individual state tensors; reset with no names clears all".into()));
        }
        let state = self.inner.state.clone().unwrap();
        spawn_blocking(move || state.reset()).await.map_err(failed)?.map_err(failed)
    }

    async fn info(self: Rc<Self>, _: session::InfoParams, mut results: session::InfoResults) -> CResult<()> {
        let specs = self.model.states();
        let mut r = results.get();
        let mut list = r.reborrow().init_state(specs.len() as u32);
        for (i, spec) in specs.iter().enumerate() {
            state_info(spec, list.reborrow().get(i as u32));
        }
        r.set_bytes_used(specs.iter().map(|s| s.dims.iter().product::<i64>() as u64 * if s.dtype == "f16" { 2 } else { 4 }).sum());
        Ok(())
    }

    async fn fork(self: Rc<Self>, _: session::ForkParams, mut results: session::ForkResults) -> CResult<()> {
        let state = self.inner.state.clone().unwrap();
        let copy = spawn_blocking(move || state.fork()).await.map_err(failed)?.map_err(failed)?;
        // The new session needs the registry that issued this one's token.
        let reg = self.reg.clone();
        let inner = reg.add_session(copy);
        results.get().set_session(capnp_rpc::new_client(SessionImpl { reg, model: self.model.clone(), inner }));
        Ok(())
    }

    async fn token(self: Rc<Self>, _: session::TokenParams, mut results: session::TokenResults) -> CResult<()> {
        results.get().set_token(self.inner.token);
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Callbacks into the client
// ---------------------------------------------------------------------------------------------

/// A request from the pipeline's thread to make a call on the client's capability, which only
/// this thread's event loop may use.
enum CallbackRequest {
    Call { name: String, args: Vec<Chunk>, reply: std::sync::mpsc::Sender<Result<Option<Chunk>, String>> },
    Sample { request: tensorlisp::SampleRequest, reply: std::sync::mpsc::Sender<Result<i32, String>> },
}

/// What the runtime calls on the pipeline's thread; each call waits for the client's answer.
struct ClientCallbacks {
    tx: tokio::sync::mpsc::UnboundedSender<CallbackRequest>,
}

impl tensorlisp::Callbacks for ClientCallbacks {
    fn call(&self, name: &str, args: Vec<Chunk>) -> Result<Option<Chunk>, String> {
        let (reply, answer) = std::sync::mpsc::channel();
        self.tx.send(CallbackRequest::Call { name: name.to_owned(), args, reply }).map_err(|_| "the run is gone")?;
        answer.recv().map_err(|_| "the run is gone".to_string())?
    }

    fn sample(&self, request: tensorlisp::SampleRequest) -> Result<i32, String> {
        let (reply, answer) = std::sync::mpsc::channel();
        self.tx.send(CallbackRequest::Sample { request, reply }).map_err(|_| "the run is gone")?;
        answer.recv().map_err(|_| "the run is gone".to_string())?
    }
}

/// A callback's result value as a chunk (numbers become 0-d arrays, `none` nothing).
fn callback_result(v: value::Reader<'_>) -> CResult<Option<Chunk>> {
    Ok(match v.which()? {
        value::Which::None(()) => None,
        value::Which::Int(i) => Some(Chunk::Array(ndarray::arr0(i as f32).into_dyn())),
        value::Which::Float(f) => Some(Chunk::Array(ndarray::arr0(f as f32).into_dyn())),
        value::Which::Bool(b) => Some(Chunk::Array(ndarray::arr0(b as u8 as f32).into_dyn())),
        _ => Some(chunk_from(v)?),
    })
}

async fn serve_callback(run: &Rc<RunState>, request: CallbackRequest) {
    let client = run.callbacks.borrow().clone();
    let Some(client) = client else {
        let message = "the client set no callbacks for this run".to_string();
        match request {
            CallbackRequest::Call { reply, .. } => drop(reply.send(Err(message))),
            CallbackRequest::Sample { reply, .. } => drop(reply.send(Err(message))),
        }
        return;
    };
    match request {
        CallbackRequest::Call { name, args, reply } => {
            let answer = async {
                let mut req = client.call_request();
                {
                    let mut p = req.get();
                    p.set_name(name.as_str());
                    let mut list = p.init_args(args.len() as u32);
                    for (i, arg) in args.iter().enumerate() {
                        write_chunk(list.reborrow().get(i as u32), arg, run.dtype)?;
                    }
                }
                let response = req.send().promise.await?;
                callback_result(response.get()?.get_result()?)
            }
            .await;
            let _ = reply.send(answer.map_err(|e| format!("client callback {name:?}: {}", e.extra)));
        }
        CallbackRequest::Sample { request, reply } => {
            let answer = async {
                let mut req = client.sample_request();
                {
                    let mut r = req.get().init_request();
                    r.set_step(request.step);
                    let mut ids = r.reborrow().init_candidate_ids(request.candidates.len() as u32);
                    for (i, (id, _)) in request.candidates.iter().enumerate() {
                        ids.set(i as u32, *id);
                    }
                    let mut logits = r.reborrow().init_candidate_logits(request.candidates.len() as u32);
                    for (i, (_, l)) in request.candidates.iter().enumerate() {
                        logits.set(i as u32, *l);
                    }
                    let mut history = r.reborrow().init_history(request.history.len() as u32);
                    for (i, t) in request.history.iter().enumerate() {
                        history.set(i as u32, *t);
                    }
                    if let Some(full) = &request.full {
                        let values: Vec<f32> = full.iter().copied().collect();
                        r.set_full_logits(capnp_rpc::new_client(TensorRefImpl {
                            dtype: DType::F32,
                            dims: full.shape().iter().map(|&d| d as u64).collect(),
                            bytes: convert::encode(DType::F32, &values)?,
                        }));
                    }
                }
                Ok::<i32, capnp::Error>(req.send().promise.await?.get()?.get_token())
            }
            .await;
            let _ = reply.send(answer.map_err(|e| format!("client sampler: {}", e.extra)));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------------------------

#[derive(Clone)]
enum Phase {
    Queued,
    Running,
    Done(Rc<Vec<(String, Out)>>),
    Failed(String),
    Cancelled,
}

impl Phase {
    fn terminal(&self) -> bool {
        matches!(self, Phase::Done(_) | Phase::Failed(_) | Phase::Cancelled)
    }
}

struct RunState {
    tx: watch::Sender<Phase>,
    rx: watch::Receiver<Phase>,
    dtype: DType,
    by_ref: bool,
    outputs: Outputs,
    /// The client's handlers for `(client-call ...)` and `(client-sample ...)`.
    callbacks: RefCell<Option<callbacks::Client>>,
}

impl RunState {
    /// Moves to `phase` if `allowed` holds for the current one.
    fn set(&self, phase: Phase, allowed: impl Fn(&Phase) -> bool) {
        self.tx.send_if_modified(|current| {
            if allowed(current) {
                *current = phase;
                true
            } else {
                false
            }
        });
    }

    async fn finished(&self) -> Phase {
        let mut rx = self.rx.clone();
        let phase = rx.wait_for(|p| p.terminal()).await.expect("the sender lives in the state");
        phase.clone()
    }
}

struct RunImpl {
    state: Rc<RunState>,
    /// The input streams of a streaming pipeline, by argument name.
    inputs: HashMap<String, Rc<StreamIn>>,
    /// Set when the client lets go of the run: the pipeline's next `emit` fails and ends it.
    abandoned: Arc<AtomicBool>,
}

impl Drop for RunImpl {
    fn drop(&mut self) {
        self.abandoned.store(true, Ordering::Relaxed);
        self.inputs.values().for_each(|s| s.close());
    }
}

impl run::Server for RunImpl {
    async fn status(self: Rc<Self>, _: run::StatusParams, mut results: run::StatusResults) -> CResult<()> {
        let mut r = results.get();
        let phase = self.state.rx.borrow().clone();
        r.set_status(match &phase {
            Phase::Queued => RunStatus::Queued,
            Phase::Running => RunStatus::Running,
            Phase::Done(_) => RunStatus::Finished,
            Phase::Failed(_) => RunStatus::Failed,
            Phase::Cancelled => RunStatus::Cancelled,
        });
        if let Phase::Failed(e) = &phase {
            r.set_error(e.as_str());
        }
        Ok(())
    }

    async fn result(self: Rc<Self>, _: run::ResultParams, mut results: run::ResultResults) -> CResult<()> {
        match self.state.finished().await {
            Phase::Done(outs) => write_outputs(results.get().init_outputs(outs.len() as u32), &outs, self.state.dtype, self.state.by_ref),
            Phase::Failed(e) => Err(failed(e)),
            _ => Err(failed("the run was cancelled")),
        }
    }

    async fn cancel(self: Rc<Self>, _: run::CancelParams, _: run::CancelResults) -> CResult<()> {
        // A pipeline that is already executing can't be interrupted; its result is dropped. One
        // that waits for stream input sees the end of the stream, and one that emits stops.
        self.state.set(Phase::Cancelled, |p| !p.terminal());
        self.abandoned.store(true, Ordering::Relaxed);
        self.inputs.values().for_each(|s| s.close());
        Ok(())
    }

    async fn set_callbacks(self: Rc<Self>, params: run::SetCallbacksParams, _: run::SetCallbacksResults) -> CResult<()> {
        *self.state.callbacks.borrow_mut() = Some(params.get()?.get_callbacks()?);
        Ok(())
    }

    async fn input(self: Rc<Self>, params: run::InputParams, mut results: run::InputResults) -> CResult<()> {
        let name = params.get()?.get_name()?.to_str()?.to_owned();
        let stream = self.inputs.get(&name).ok_or_else(|| {
            let mut names: Vec<&str> = self.inputs.keys().map(String::as_str).collect();
            names.sort();
            failed(format!("this pipeline has no input stream {name:?}; streams: {}", names.join(", ")))
        })?;
        results.get().set_sink(capnp_rpc::new_client(SinkImpl { stream: stream.clone() }));
        Ok(())
    }

    async fn output(self: Rc<Self>, params: run::OutputParams, mut results: run::OutputResults) -> CResult<()> {
        let name = params.get()?.get_name()?.to_str()?.to_owned();
        let queue = self.state.outputs.queue(&name);
        results.get().set_source(capnp_rpc::new_client(SourceImpl { queue, run: self.state.clone() }));
        Ok(())
    }
}

