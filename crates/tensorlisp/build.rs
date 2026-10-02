fn main() {
    println!("cargo::rustc-check-cfg=cfg(native_device)");
    println!("cargo::rerun-if-changed=build.rs");
    let mac = std::env::var("CARGO_CFG_TARGET_OS").is_ok_and(|os| os == "macos");
    let cuda_native = std::env::var_os("CARGO_FEATURE_NATIVE_CUDA").is_some();
    if mac || cuda_native {
        println!("cargo::rustc-cfg=native_device");
    }
}
