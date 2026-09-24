//! The embedded Chez REPL: `cargo run -p chez --example repl`.
fn main() {
    let scheme = chez::Scheme::new().unwrap();
    std::process::exit(scheme.repl());
}
