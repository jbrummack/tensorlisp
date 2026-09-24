//! Raw bindings to the Chez C API (scheme.h) plus `chez_*` wrappers for its macros.
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals, dead_code)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
