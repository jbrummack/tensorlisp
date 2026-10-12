//! Reverse mode: the worker dials a scheduler instead of waiting to be called.
//!
//! The worker opens the connection, offers its [`Server`](crate::tensorlisp_capnp::server) as the
//! bootstrap capability of that same connection, and calls `Scheduler.register` on whatever the
//! scheduler offers. From then on the scheduler holds a capability to the worker and drives it
//! through the ordinary interface (install models, run entries, start pipelines, stream), so a
//! worker behind NAT or a firewall needs no inbound port. The registration lasts as long as the
//! connection: when it drops the worker dials again, with a growing delay, and registers again.
use std::{path::PathBuf, rc::Rc, str::FromStr, time::Duration};

use anyhow::{Context, Result, bail};
use capnp_rpc::{RpcSystem, rpc_twoparty_capnp::Side, twoparty};
use tokio_util::compat::{TokioAsyncReadCompatExt, TokioAsyncWriteCompatExt};

use crate::{
    server::{Registry, server_client},
    tensorlisp_capnp::scheduler,
};

/// `tcp:HOST:PORT` or `unix:PATH`.
#[derive(Debug, Clone, PartialEq)]
pub enum Address {
    Tcp(String),
    Unix(PathBuf),
}

impl FromStr for Address {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Address> {
        if let Some(addr) = s.strip_prefix("tcp:") {
            Ok(Address::Tcp(addr.to_owned()))
        } else if let Some(path) = s.strip_prefix("unix:") {
            Ok(Address::Unix(path.into()))
        } else {
            bail!("expected tcp:HOST:PORT or unix:PATH, got {s:?}")
        }
    }
}

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Address::Tcp(a) => write!(f, "tcp:{a}"),
            Address::Unix(p) => write!(f, "unix:{}", p.display()),
        }
    }
}

/// One connection: dial, register, and wait until it ends. Ok(true) if the registration went through.
pub async fn connect_once(registry: Rc<Registry>, address: &Address, name: &str) -> Result<bool> {
    match address {
        Address::Tcp(a) => {
            let stream = tokio::net::TcpStream::connect(a).await.with_context(|| format!("connecting to {address}"))?;
            stream.set_nodelay(true)?;
            run(registry, stream, name).await
        }
        Address::Unix(p) => {
            let stream = tokio::net::UnixStream::connect(p).await.with_context(|| format!("connecting to {address}"))?;
            run(registry, stream, name).await
        }
    }
}

async fn run<S>(registry: Rc<Registry>, stream: S, name: &str) -> Result<bool>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin + 'static,
{
    let (reader, writer) = tokio::io::split(stream);
    let network = twoparty::VatNetwork::new(reader.compat(), writer.compat_write(), Side::Client, Default::default());
    let mut rpc = RpcSystem::new(Box::new(network), Some(crate::server::bootstrap(registry.clone())));
    let scheduler: scheduler::Client = rpc.bootstrap(Side::Server);
    let connection = tokio::task::spawn_local(rpc);

    let mut request = scheduler.register_request();
    {
        let mut p = request.get();
        p.set_worker(server_client(registry.clone()));
        registry.fill_worker_info(p.init_info(), name);
    }
    let registered = match request.send().promise.await {
        Ok(reply) => {
            eprintln!("registered as {:?}", reply.get()?.get_id()?.to_str()?);
            true
        }
        Err(e) => {
            connection.abort();
            return Err(anyhow::anyhow!("registering: {e}"));
        }
    };
    // Until the scheduler goes away.
    match connection.await {
        Ok(Ok(())) | Err(_) => {}
        Ok(Err(e)) => eprintln!("connection ended: {e}"),
    }
    Ok(registered)
}

/// Keeps a registration alive forever: dial, serve until the connection ends, wait, dial again. The
/// wait starts at one second and doubles up to `max_wait` while dialing or registering fails, and
/// starts over after a connection that registered.
pub async fn connect_loop(registry: Rc<Registry>, address: Address, name: String, max_wait: Duration) {
    let mut wait = Duration::from_secs(1);
    loop {
        match connect_once(registry.clone(), &address, &name).await {
            Ok(true) => {
                eprintln!("disconnected from {address}");
                wait = Duration::from_secs(1);
            }
            Ok(false) => {}
            Err(e) => eprintln!("{e:#}"),
        }
        tokio::time::sleep(wait).await;
        wait = (wait * 2).min(max_wait);
    }
}
