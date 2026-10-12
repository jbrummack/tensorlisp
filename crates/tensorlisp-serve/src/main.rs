use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use tensorlisp::Device;
use tensorlisp_serve::{Registry, serve_stream, store::Store};

/// Serves tensorlisp models over Cap'n Proto RPC (schema/tensorlisp.capnp).
#[derive(Parser)]
#[command(name = "tl-serve")]
struct Args {
    /// Wait for clients on `unix:PATH` or `tcp:HOST:PORT` (default tcp:127.0.0.1:7700 unless --connect is given).
    #[arg(long)]
    listen: Option<String>,
    /// Dial a scheduler (`unix:PATH` or `tcp:HOST:PORT`, repeatable), offer this server over the connection and
    /// register with it; redial when the connection ends. Needs no inbound port on this side.
    #[arg(long)]
    connect: Vec<String>,
    /// The name this worker registers under (default: the host name).
    #[arg(long)]
    name: Option<String>,
    /// Where installed models live (default: ~/.cache/tensorlisp/models).
    #[arg(long)]
    store: Option<PathBuf>,
    /// Device for models that don't ask for one: auto, cpu, gpu or native.
    #[arg(long, default_value = "auto")]
    device: String,
}

fn device(name: &str) -> Result<Device> {
    Ok(match name {
        "auto" => Device::Auto,
        "cpu" => Device::Cpu,
        "gpu" => Device::Gpu,
        "native" => Device::Native,
        other => bail!("unknown device {other:?} (expected auto, cpu, gpu or native)"),
    })
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    let store_dir = match args.store {
        Some(dir) => dir,
        None => PathBuf::from(std::env::var_os("HOME").context("HOME is not set; pass --store")?)
            .join(".cache/tensorlisp/models"),
    };
    let registry = Registry::new(Store::open(store_dir)?, device(&args.device)?);
    let name = args.name.clone().unwrap_or_else(|| {
        std::process::Command::new("hostname")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "worker".into())
    });
    let connect: Vec<tensorlisp_serve::Address> = args.connect.iter().map(|a| a.parse()).collect::<Result<_>>()?;
    let listen = match (&args.listen, connect.is_empty()) {
        (Some(l), _) => Some(l.clone()),
        (None, true) => Some("tcp:127.0.0.1:7700".to_string()),
        (None, false) => None,
    };
    let local = tokio::task::LocalSet::new();
    local
        .run_until(async move {
            for address in connect {
                eprintln!("dialing {address} as {name:?}");
                tokio::task::spawn_local(tensorlisp_serve::connect_loop(
                    registry.clone(),
                    address,
                    name.clone(),
                    std::time::Duration::from_secs(30),
                ));
            }
            let Some(listen) = listen else { return std::future::pending::<Result<()>>().await };
            if let Some(path) = listen.strip_prefix("unix:") {
                let _ = std::fs::remove_file(path);
                let listener = tokio::net::UnixListener::bind(path).with_context(|| format!("binding {path}"))?;
                eprintln!("tl-serve listening on unix:{path}");
                loop {
                    let (stream, _) = listener.accept().await?;
                    serve_stream(registry.clone(), stream);
                }
            } else if let Some(addr) = listen.strip_prefix("tcp:") {
                let listener = tokio::net::TcpListener::bind(addr).await.with_context(|| format!("binding {addr}"))?;
                eprintln!("tl-serve listening on tcp:{}", listener.local_addr()?);
                loop {
                    let (stream, _) = listener.accept().await?;
                    stream.set_nodelay(true)?;
                    serve_stream(registry.clone(), stream);
                }
            } else {
                bail!("--listen expects unix:PATH or tcp:HOST:PORT")
            }
        })
        .await
}
