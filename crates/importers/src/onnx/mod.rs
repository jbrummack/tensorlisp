pub mod loader;
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}
