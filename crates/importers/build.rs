fn main() -> std::io::Result<()> {
    println!("cargo:rerun-if-changed=proto");
    // Generated code goes to OUT_DIR and is pulled in with include! in onnx/mod.rs and mil/mod.rs.
    prost_build::compile_protos(&["proto/onnx.proto", "proto/Model.proto"], &["proto/"])
}
