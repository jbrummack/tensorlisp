//! Reverse mode: the worker dials the scheduler, offers its Server over that connection and registers;
//! the scheduler then uses it like any client, and the worker dials again after a dropped connection.
use std::{cell::RefCell, path::PathBuf, rc::Rc, time::Duration};

use capnp_rpc::{RpcSystem, rpc_twoparty_capnp::Side, twoparty};
use tensorlisp::{Device, Program, gguf::GgufWriter};
use tensorlisp_serve::{
    Address, Registry, connect_loop,
    store::Store,
    tensorlisp_capnp::{DType, scheduler, server, value},
};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

/// The scheduler's side: remembers each registered worker and what it said about itself.
struct Fleet {
    registered: tokio::sync::mpsc::UnboundedSender<(server::Client, String, Vec<String>)>,
    count: RefCell<u32>,
}

impl scheduler::Server for Fleet {
    async fn register(
        self: capnp::capability::Rc<Self>,
        params: scheduler::RegisterParams,
        mut results: scheduler::RegisterResults,
    ) -> Result<(), capnp::Error> {
        let p = params.get()?;
        let info = p.get_info()?;
        let name = info.get_name()?.to_str()?.to_owned();
        let models = info.get_models()?.iter().map(|m| m.get_name().unwrap().to_str().unwrap().to_owned()).collect();
        *self.count.borrow_mut() += 1;
        let id = format!("worker-{}", self.count.borrow());
        let _ = self.registered.send((p.get_worker()?, name, models));
        results.get().set_id(id.as_str());
        Ok(())
    }
}

type Connections = Rc<RefCell<Vec<tokio::task::JoinHandle<Result<(), capnp::Error>>>>>;

/// A scheduler listening on `socket`: each connection gets the Fleet as its bootstrap capability.
fn start_scheduler(
    socket: &std::path::Path,
) -> (tokio::sync::mpsc::UnboundedReceiver<(server::Client, String, Vec<String>)>, Connections) {
    let listener = tokio::net::UnixListener::bind(socket).unwrap();
    let (tx, registered) = tokio::sync::mpsc::unbounded_channel();
    let fleet: scheduler::Client = capnp_rpc::new_client(Fleet { registered: tx, count: RefCell::new(0) });
    let connections: Connections = Default::default();
    {
        let connections = connections.clone();
        tokio::task::spawn_local(async move {
            loop {
                let (stream, _) = listener.accept().await.unwrap();
                let (r, w) = tokio::io::split(stream);
                let network = twoparty::VatNetwork::new(r.compat(), w.compat_write(), Side::Server, Default::default());
                let rpc = RpcSystem::new(Box::new(network), Some(fleet.clone().client));
                connections.borrow_mut().push(tokio::task::spawn_local(rpc));
            }
        });
    }
    (registered, connections)
}

async fn run_double(model: tensorlisp_serve::tensorlisp_capnp::model::Client) -> Vec<f32> {
    let mut req = model.run_request();
    {
        let mut p = req.get();
        p.set_entry("double");
        let mut inputs = p.init_inputs(1);
        let mut i0 = inputs.reborrow().get(0);
        i0.set_name("x");
        let mut t = i0.init_value().init_tensor();
        t.set_dtype(DType::F32);
        let mut dims = t.reborrow().init_dims(2);
        dims.set(0, 1);
        dims.set(1, 4);
        t.set_data(&[1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    }
    let res = req.send().promise.await.unwrap();
    let outs = res.get().unwrap().get_outputs().unwrap();
    let value::Which::Tensor(t) = outs.get(0).get_value().unwrap().which().unwrap() else { panic!() };
    let t = t.unwrap();
    tensorlisp_serve::convert::decode(t.get_dtype().unwrap(), t.get_data().unwrap()).unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn a_worker_dials_registers_and_comes_back() {
    let dir: PathBuf = std::env::temp_dir().join(format!("tl-reverse-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut w = GgufWriter::new();
    w.set_program(&Program::Text("(model double (inputs [x f32 (4 batch)]) (outputs [y (ggml-scale x 2.0) 2]))".into())).unwrap();
    w.write(dir.join("m.gguf")).unwrap();
    let gguf = std::fs::read(dir.join("m.gguf")).unwrap();

    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            // The scheduler listens; the worker only dials.
            let socket = dir.join("s.sock");
            let (mut registered, connections) = start_scheduler(&socket);

            let registry = Registry::new(Store::open(dir.join("store")).unwrap(), Device::Cpu);
            tokio::task::spawn_local(connect_loop(registry, Address::Unix(socket.clone()), "gpu-box-7".into(), Duration::from_secs(2)));

            // 1. It registers, with its name and nothing installed.
            let (worker, name, models) = tokio::time::timeout(Duration::from_secs(20), registered.recv()).await.expect("a registration").unwrap();
            assert_eq!((name.as_str(), models.len()), ("gpu-box-7", 0));

            // 2. The scheduler drives it like any client.
            let info = worker.info_request().send().promise.await.unwrap();
            assert_eq!(info.get().unwrap().get_devices().unwrap().get(0).get_name().unwrap().to_str().unwrap(), "cpu");
            let mut req = worker.install_request();
            req.get().init_options().set_name("doubler");
            let upload = req.send().promise.await.unwrap().get().unwrap().get_upload().unwrap();
            let mut push = upload.push_request();
            push.get().set_chunk(&gguf);
            push.send().await.unwrap();
            let model = upload.finish_request().send().promise.await.unwrap().get().unwrap().get_model().unwrap();
            assert_eq!(run_double(model).await, [2.0, 4.0, 6.0, 8.0]);

            // 3. The scheduler drops the connection; the worker dials again and tells what it now holds.
            let first = connections.borrow_mut().remove(0);
            first.abort();
            let _ = first.await;
            drop(worker);
            let (worker2, name2, models2) = tokio::time::timeout(Duration::from_secs(20), registered.recv()).await.expect("a second registration").unwrap();
            assert_eq!((name2.as_str(), models2), ("gpu-box-7", vec!["doubler".to_string()]));
            let mut open = worker2.open_request();
            open.get().set_name("doubler");
            let model = open.send().promise.await.unwrap().get().unwrap().get_model().unwrap();
            assert_eq!(run_double(model).await, [2.0, 4.0, 6.0, 8.0]);
        })
        .await;
    let _ = std::fs::remove_dir_all(dir);
}

/// The real binary: `tl-serve --connect` in a child process dials the scheduler of this test.
#[tokio::test(flavor = "current_thread")]
async fn the_binary_dials_in() {
    let dir: PathBuf = std::env::temp_dir().join(format!("tl-reverse-bin-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut w = GgufWriter::new();
    w.set_program(&Program::Text("(model double (inputs [x f32 (4 batch)]) (outputs [y (ggml-scale x 2.0) 2]))".into())).unwrap();
    w.write(dir.join("m.gguf")).unwrap();
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async {
            let socket = dir.join("s.sock");
            let (mut registered, _connections) = start_scheduler(&socket);
            let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_tl-serve"))
                .args(["--connect", &format!("unix:{}", socket.display()), "--name", "child-1", "--device", "cpu", "--store"])
                .arg(dir.join("store"))
                .stderr(std::process::Stdio::null())
                .spawn()
                .unwrap();
            let (worker, name, _) = tokio::time::timeout(Duration::from_secs(30), registered.recv()).await.expect("the child registers").unwrap();
            assert_eq!(name, "child-1");
            // Install from a path the worker can read, then run.
            let mut req = worker.install_from_request();
            req.get().set_location(dir.join("m.gguf").to_str().unwrap());
            let model = req.send().promise.await.unwrap().get().unwrap().get_model().unwrap();
            assert_eq!(run_double(model).await, [2.0, 4.0, 6.0, 8.0]);
            child.kill().unwrap();
            child.wait().unwrap();
        })
        .await;
    let _ = std::fs::remove_dir_all(dir);
}
