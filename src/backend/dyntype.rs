use std::{rc::Rc, sync::LazyLock};

use crate::backend::ffi::ggml_tensor;
#[derive(Debug, thiserror::Error)]
pub enum DynTypeError {
    #[error("Invalid conversion from {from} to {to}")]
    InvalidConversion {
        from: &'static str,
        to: &'static str,
    },
}
pub struct Lambda {
    result: LazyLock<Dyntype>,
}
#[derive(Debug, Clone)]
pub enum Dyntype {
    Tensor(*mut ggml_tensor),
    Integer(i64),
    Float(f64),
    Bool(bool),
    List(Vec<Self>),
    Nil,
}

/*impl Iterator<Item = Dyntype> for Dyntype {
    type Item;

    fn next(&mut self) -> Option<Self::Item> {
        match self {

            Dyntype::Cons(cons) => {
                let cdr = cons.0;
                let car = cons.1;
                None
            },
            Dyntype::Nil => None,
            other => Some(Self::Cons(Rc::new(value)))
        }
    }
}*/
impl Dyntype {
    pub fn src_name(&self) -> &'static str {
        match self {
            Dyntype::Tensor(_) => "Tensor",
            Dyntype::Integer(_) => "Integer",
            Dyntype::Float(_) => "Float",
            Dyntype::Bool(_) => "Bool",
            Dyntype::Nil => "Nil",
            Dyntype::List(_) => "List",
        }
    }
}

impl TryInto<f32> for Dyntype {
    type Error = DynTypeError;

    fn try_into(self) -> Result<f32, Self::Error> {
        if let Dyntype::Float(f) = self {
            Ok(f as f32)
        } else {
            Err(DynTypeError::InvalidConversion {
                from: self.src_name(),
                to: "f32",
            })
        }
    }
}
impl TryInto<f64> for Dyntype {
    type Error = DynTypeError;

    fn try_into(self) -> Result<f64, Self::Error> {
        if let Dyntype::Float(f) = self {
            Ok(f as f64)
        } else {
            Err(DynTypeError::InvalidConversion {
                from: self.src_name(),
                to: "f64",
            })
        }
    }
}
impl TryInto<*mut ggml_tensor> for Dyntype {
    type Error = DynTypeError;

    fn try_into(self) -> Result<*mut ggml_tensor, Self::Error> {
        if let Dyntype::Tensor(f) = self {
            Ok(f)
        } else {
            Err(DynTypeError::InvalidConversion {
                from: self.src_name(),
                to: "ggml_tensor",
            })
        }
    }
}
