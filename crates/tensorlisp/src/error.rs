#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("gguf: {0}")]
    Gguf(String),
    #[error("program: {0}")]
    Program(String),
    #[error("scheme: {0}")]
    Scheme(#[from] chez::Error),
    #[error("input: {0}")]
    Input(String),
    #[error("backend: {0}")]
    Backend(String),
    #[error("ggml assertion failed: {0}")]
    Ggml(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
