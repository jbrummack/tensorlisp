//! Generates the Scheme side of the ggml bindings from the bindgen module:
//!
//! - `symbols()`: (name, address) of every function, for `Sregister_symbol`,
//!   since statically linked functions are invisible to `foreign-procedure`.
//! - `RAW`: `(define ggml_add (foreign-procedure "ggml_add" (uptr uptr uptr) uptr))`
//!   for every function whose signature maps onto Chez foreign types.
//! - `OPS`: model-facing graph ops with an implicit context and tensor records,
//!   e.g. `(define (ggml-add a b) ...)`, for functions that take the context
//!   first and return a tensor, with only tensor and scalar arguments.
//! - `CONSTANTS`: `(define GGML_TYPE_F32 0)` for every ggml enum variant.
//!
//! OPS refers to `%current-ctx`, `%unwrap` (who, parameter name, value) and
//! `%wrap`, which the tensorlisp
//! runtime library defines.
use std::collections::HashMap;

use quote::{format_ident, quote};
use syn::{Expr, ForeignItem, GenericArgument, Item, PathArguments, ReturnType, Type};

/// Graph ops that write tensor data. Graphs are built in no_alloc contexts,
/// so these would dereference NULL.
const DATA_WRITING_OPS: &[&str] = &["ggml_new_i32", "ggml_new_f32"];

/// Declared in the headers but not implemented anywhere in ggml.
const UNIMPLEMENTED: &[&str] = &["ggml_threadpool_get_n_threads"];

/// How a C type crosses into Scheme.
#[derive(Clone, Debug, PartialEq)]
enum Kind {
    Void,
    Tensor,
    Context,
    /// C string argument or return (`const char *`), converted by Chez.
    Str,
    /// Any other pointer: passed as an address.
    Pointer,
    /// Numeric or boolean; the Chez foreign type name.
    Scalar(&'static str),
}

impl Kind {
    fn foreign_type(&self) -> &'static str {
        match self {
            Kind::Void => "void",
            Kind::Tensor | Kind::Context | Kind::Pointer => "uptr",
            Kind::Str => "string",
            Kind::Scalar(t) => t,
        }
    }
}

struct Types<'a> {
    aliases: HashMap<String, &'a Type>,
    enums: HashMap<String, &'static str>,
}

impl<'a> Types<'a> {
    fn collect(items: &'a [Item]) -> Self {
        let mut aliases = HashMap::new();
        let mut enums = HashMap::new();
        for item in items {
            match item {
                Item::Type(alias) => {
                    aliases.insert(alias.ident.to_string(), &*alias.ty);
                }
                Item::Enum(e) => {
                    let unsigned = e.attrs.iter().any(|a| {
                        a.path().is_ident("repr") && quote!(#a).to_string().contains("u32")
                    });
                    enums.insert(e.ident.to_string(), if unsigned { "unsigned-32" } else { "int" });
                }
                _ => {}
            }
        }
        Types { aliases, enums }
    }

    fn kind(&self, ty: &Type) -> Option<Kind> {
        match ty {
            Type::Ptr(p) => {
                let pointee = last_ident(&p.elem);
                Some(match pointee.as_deref() {
                    Some("ggml_tensor") => Kind::Tensor,
                    Some("ggml_context") => Kind::Context,
                    Some("c_char") if p.const_token.is_some() => Kind::Str,
                    _ => Kind::Pointer,
                })
            }
            Type::Path(path) => {
                let seg = path.path.segments.last()?;
                let name = seg.ident.to_string();
                // Function pointers: Option<unsafe extern "C" fn(..)>
                if name == "Option" {
                    if let PathArguments::AngleBracketed(args) = &seg.arguments {
                        if let Some(GenericArgument::Type(Type::BareFn(_))) = args.args.first() {
                            return Some(Kind::Pointer);
                        }
                    }
                    return None;
                }
                if let Some(t) = scalar(&name) {
                    return Some(Kind::Scalar(t));
                }
                if let Some(t) = self.enums.get(&name) {
                    return Some(Kind::Scalar(t));
                }
                // Structs passed by value have no alias and are unsupported.
                self.aliases.get(&name).and_then(|t| self.kind(t))
            }
            Type::Tuple(t) if t.elems.is_empty() => Some(Kind::Void),
            _ => None,
        }
    }
}

fn scalar(name: &str) -> Option<&'static str> {
    Some(match name {
        "bool" => "stdbool",
        "i8" | "c_schar" | "c_char" => "integer-8",
        "u8" | "c_uchar" => "unsigned-8",
        "i16" | "c_short" => "integer-16",
        "u16" | "c_ushort" => "unsigned-16",
        "i32" | "c_int" => "integer-32",
        "u32" | "c_uint" => "unsigned-32",
        "i64" | "c_longlong" => "integer-64",
        "u64" | "c_ulonglong" => "unsigned-64",
        "c_long" | "isize" => "iptr",
        "c_ulong" | "usize" => "uptr",
        "f32" | "c_float" => "float",
        "f64" | "c_double" => "double",
        _ => return None,
    })
}

fn last_ident(ty: &Type) -> Option<String> {
    match ty {
        Type::Path(p) => p.path.segments.last().map(|s| s.ident.to_string()),
        _ => None,
    }
}

struct Function {
    name: String,
    params: Vec<(String, Kind)>,
    ret: Kind,
}

impl Function {
    fn raw_definition(&self) -> String {
        let params: Vec<_> = self.params.iter().map(|(_, k)| k.foreign_type()).collect();
        format!(
            "(define {0} (foreign-procedure \"{0}\" ({1}) {2}))\n",
            self.name,
            params.join(" "),
            self.ret.foreign_type()
        )
    }

    /// Context first, tensor out, everything else a tensor or scalar.
    fn is_graph_op(&self) -> bool {
        matches!(self.params.first(), Some((_, Kind::Context)))
            && self.ret == Kind::Tensor
            && self.params[1..].iter().all(|(_, k)| matches!(k, Kind::Tensor | Kind::Scalar(_)))
            && !DATA_WRITING_OPS.contains(&self.name.as_str())
    }

    fn op_name(&self) -> String {
        self.name.replace('_', "-")
    }

    /// Tensor arguments are unwrapped in declaration order and recorded by
    /// name, so an assertion inside ggml can report them.
    fn op_definition(&self) -> String {
        let op = self.op_name();
        let params: Vec<_> = self.params[1..].iter().map(|(n, _)| n.as_str()).collect();
        let bindings: String = self.params[1..]
            .iter()
            .filter(|(_, k)| *k == Kind::Tensor)
            .map(|(n, _)| format!(" [{n} (%unwrap '{op} '{n} {n})]"))
            .collect();
        format!(
            "(define ({op}{}{}) (let* ([%c (%current-ctx '{op})]{bindings}) (%wrap '{op} ({} %c{}{}))))\n",
            if params.is_empty() { "" } else { " " },
            params.join(" "),
            self.name,
            if params.is_empty() { "" } else { " " },
            params.join(" "),
        )
    }
}

fn param_name(pat: &syn::Pat, index: usize) -> String {
    match pat {
        // bindgen appends `_` to Rust keywords (type_, fn_); Scheme doesn't need that.
        syn::Pat::Ident(i) => i.ident.to_string().trim_end_matches('_').replace('_', "-"),
        _ => format!("arg{index}"),
    }
}

fn functions(items: &[Item], types: &Types) -> Vec<Function> {
    let mut out = Vec::new();
    for item in items {
        let Item::ForeignMod(foreign) = item else { continue };
        for item in &foreign.items {
            let ForeignItem::Fn(f) = item else { continue };
            let sig = &f.sig;
            let name = sig.ident.to_string();
            if !(name.starts_with("ggml_") || name.starts_with("gguf_"))
                || sig.variadic.is_some()
                || UNIMPLEMENTED.contains(&name.as_str())
            {
                continue;
            }
            let params: Option<Vec<_>> = sig
                .inputs
                .iter()
                .enumerate()
                .map(|(i, arg)| match arg {
                    syn::FnArg::Typed(t) => types.kind(&t.ty).map(|k| (param_name(&t.pat, i), k)),
                    syn::FnArg::Receiver(_) => None,
                })
                .collect();
            let ret = match &sig.output {
                ReturnType::Default => Some(Kind::Void),
                ReturnType::Type(_, t) => types.kind(t),
            };
            if let (Some(params), Some(ret)) = (params, ret) {
                out.push(Function { name, params, ret });
            }
        }
    }
    out
}

fn constants(items: &[Item]) -> Vec<(String, i64)> {
    let mut out = Vec::new();
    for item in items {
        let Item::Enum(e) = item else { continue };
        for v in &e.variants {
            let name = v.ident.to_string();
            let Some((_, expr)) = &v.discriminant else { continue };
            if !name.starts_with("GGML_") {
                continue;
            }
            let value = match expr {
                Expr::Lit(l) => quote!(#l).to_string().parse::<i64>().ok(),
                Expr::Unary(u) => quote!(#u).to_string().replace(' ', "").parse::<i64>().ok(),
                _ => None,
            };
            if let Some(value) = value {
                out.push((name, value));
            }
        }
    }
    out
}

pub fn generate(items: &[Item]) -> proc_macro2::TokenStream {
    let types = Types::collect(items);
    let functions = functions(items, &types);
    let constants = constants(items);

    let raw: String = functions.iter().map(Function::raw_definition).collect();
    let ops: Vec<_> = functions.iter().filter(|f| f.is_graph_op()).collect();
    let op_defs: String = ops.iter().map(|f| f.op_definition()).collect();
    let op_names = ops.iter().map(|f| f.op_name()).collect::<Vec<_>>().join(" ");
    let const_defs: String = constants.iter().map(|(n, v)| format!("(define {n} {v})\n")).collect();
    let const_names = constants.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(" ");

    let symbol_names = functions.iter().map(|f| f.name.as_str());
    let symbol_fns = functions.iter().map(|f| format_ident!("{}", f.name));

    quote! {
        /// Scheme bindings generated from this module by ggml-codegen.
        pub mod scheme {
            /// `(define ggml_x (foreign-procedure "ggml_x" (..) ..))` for every supported function.
            pub const RAW: &str = #raw;
            /// Graph ops with an implicit context, e.g. `(ggml-add a b)`.
            pub const OPS: &str = #op_defs;
            /// Space-separated names defined by [`OPS`].
            pub const OP_NAMES: &str = #op_names;
            /// `(define GGML_... n)` for every ggml enum variant.
            pub const CONSTANTS: &str = #const_defs;
            /// Space-separated names defined by [`CONSTANTS`].
            pub const CONSTANT_NAMES: &str = #const_names;

            /// Addresses of every function in [`RAW`], to register with Chez.
            pub fn symbols() -> ::std::vec::Vec<(&'static str, *const ::std::ffi::c_void)> {
                ::std::vec![#((#symbol_names, super::#symbol_fns as *const ::std::ffi::c_void)),*]
            }
        }
    }
}
