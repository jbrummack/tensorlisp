//! Cap'n Proto RPC server for tensorlisp (`schema/tensorlisp.capnp`).
//!
//! Everything runs on one thread (capnp-rpc is single-threaded): capabilities are `Rc`s on a
//! `tokio::task::LocalSet`, and the blocking calls into the runtime go through `spawn_blocking`.
pub mod connect;
pub mod convert;
pub mod server;
pub mod store;

#[allow(clippy::all, dead_code)]
pub mod tensorlisp_capnp {
    include!(concat!(env!("OUT_DIR"), "/tensorlisp_capnp.rs"));
}

pub use connect::{Address, connect_loop, connect_once};
pub use server::{Registry, bootstrap, serve_stream};
