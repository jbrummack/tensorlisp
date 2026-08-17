fn main() -> std::io::Result<()> {
    println!("cargo:rerun-if-changed=backend/include/ggml.h");
    let bindings = bindgen::Builder::default()
        .header("./backend/include/ggml.h")
        .rustified_enum(".*")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    let out_path = std::path::PathBuf::from("./src/backend/bindings.rs");
    // 1. Get raw string from bindgen
    let code = bindings.to_string();

    // 2. Wrap the generated items directly in the module text
    let wrapped_code = format!(
        "#[allow(non_camel_case_types,non_snake_case,non_upper_case_globals)]#[codegen::parse_ggml]\npub mod ffi {{\n{}\n}}",
        code
    );

    std::fs::write(out_path, wrapped_code).expect("Couldn't write bindings!");
    Ok(())
}
