fn main() {
    println!("cargo:rerun-if-changed=schema/tensorlisp.capnp");
    capnpc::CompilerCommand::new()
        .src_prefix("schema")
        .file("schema/tensorlisp.capnp")
        .run()
        .expect("compiling schema/tensorlisp.capnp (needs the `capnp` tool on PATH)");
}
