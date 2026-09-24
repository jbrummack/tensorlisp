fn main() -> anyhow::Result<()> {
    importers::onnx::loader::load_onnx()
}
