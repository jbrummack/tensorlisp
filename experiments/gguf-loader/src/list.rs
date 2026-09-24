pub trait List {
    fn fetch_arg<T: GetFromList>(&mut self, fname: &'static str) -> Result<T::SelfType, ListError>;
}
//pub trait VecList {}
//pub struct TypeDef {}

#[derive(Debug, thiserror::Error)]
pub enum ListError {
    #[error("Invalid conversion from {from} to {to}")]
    InvalidConversion {
        from: &'static str,
        to: &'static str,
        fname: &'static str,
    },
    #[error("Reached list end at {at}")]
    ReachedEnd { at: &'static str },
}
pub enum ListCell {
    GgmlContext,
}

pub trait GetFromList {
    type SelfType;
}
