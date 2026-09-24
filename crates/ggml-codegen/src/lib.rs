use std::{collections::HashSet, sync::LazyLock};

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{ItemMod, Pat, PatStruct, Path, Signature, Type, TypePath, TypePtr, parse_macro_input};
static _TENSOR_ADJACENT_TYPES: LazyLock<HashSet<&'static str>> =
    LazyLock::new(|| ["ggml_tensor", "ggml_context", "usize", "f32", "c_int"].into());
#[allow(unused)]
#[derive(Debug)]
enum ExtType {
    Float,
    USize,
    CInt,
    CChar,
    File,
    Op,
    Guid,
    PoolOp,
    GluOp,
    SortOrder,
    Void,
    //UInt64,
    Int64,
    FType,
    Double,
    UInt32,
    Bf16,
    Fp16,
    Int32,
    Object,
    Bool,
    Status,
    QuantType,
    Unop,
    TriType,
    InitParams,
    Binop,
    Precision,
    Context,
    ScaleMode,
    Tensor,
    CGraph,
    OpHint,
    Ptr(Box<Self>),
    Unknown(String),
}

trait NameExt {
    fn get_name(&self) -> Option<String>;
}
impl NameExt for Path {
    fn get_name(&self) -> Option<String> {
        self.segments.iter().last().map(|i| i.ident.to_string())
    }
}
impl NameExt for TypePath {
    fn get_name(&self) -> Option<String> {
        self.path.get_name()
        //self.segments.iter().last().map(|i| i.ident.to_string())
    }
}

impl NameExt for Type {
    fn get_name(&self) -> Option<String> {
        match self {
            Type::Path(type_path) => type_path.get_name(),
            Type::Ptr(type_ptr) => type_ptr.get_name(),
            _ => None,
        }
    }
}
impl NameExt for TypePtr {
    fn get_name(&self) -> Option<String> {
        self.elem.get_name()
    }
}
impl NameExt for PatStruct {
    fn get_name(&self) -> Option<String> {
        self.path.get_name()
    }
}
impl NameExt for Pat {
    fn get_name(&self) -> Option<String> {
        match self {
            Pat::Ident(pat_ident) => Some(pat_ident.ident.to_string()),

            Pat::Path(expr_path) => expr_path.path.get_name(),

            Pat::Struct(pat_struct) => pat_struct.get_name(),

            _ => None,
        }
    }
}
impl ExtType {
    fn from_ty(t: &Type) -> Option<Self> {
        let name = t.get_name()?;
        let s = Self::from_str(&name);
        if let &Type::Ptr(_) = t {
            Some(Self::Ptr(Box::new(s)))
        } else {
            Some(s)
        }
    }
    fn from_str(s: impl AsRef<str>) -> Self {
        match s.as_ref() {
            "ggml_tensor" => Self::Tensor,
            "ggml_context" => Self::Context,
            "c_int" => Self::CInt,
            "c_void" => Self::Void,
            "ggml_prec" => Self::Precision,
            "FILE" => Self::File,
            "ggml_status" => Self::Status,
            "ggml_guid_t" => Self::Guid,
            "ggml_bf16_t" => Self::Bf16,
            "ggml_fp16_t" => Self::Fp16,
            "ggml_op_pool" => Self::PoolOp,
            "ggml_sort_order" => Self::SortOrder,
            "ggml_tri_type" => Self::TriType,
            "u64" => Self::Int64,
            "i64" => Self::Int64,
            "i32" => Self::Int32,
            "ggml_unary_op" => Self::Unop,
            "ggml_binary_op" => Self::Binop,
            "c_char" => Self::CChar,
            "f32" => Self::Float,
            "f64" => Self::Double,
            "ggml_object" => Self::Object,
            "ggml_op_hint" => Self::OpHint,
            "ggml_glu_op" => Self::GluOp,
            "ggml_init_params" => Self::InitParams,
            "ggml_ftype" => Self::FType,
            "ggml_op" => Self::Op,
            "u32" => Self::UInt32,
            "usize" => Self::USize,
            "ggml_type" => Self::QuantType,
            "ggml_cgraph" => Self::CGraph,
            "bool" => Self::Bool,
            "ggml_scale_mode" => Self::ScaleMode,
            _ => Self::Unknown(s.as_ref().into()),
        }
    }
}
#[allow(unused)]
#[derive(Debug)]
struct Function {
    name: String,
    arguments: Vec<(String, ExtType)>,
    output: Option<ExtType>,
}
fn extract_sig(sig: &Signature) -> Function {
    let name = sig.ident.to_string();
    let output = match &sig.output {
        syn::ReturnType::Default => None,
        syn::ReturnType::Type(_, tname) => ExtType::from_ty(&tname),
    };
    fn extract_arg(argument: &syn::FnArg) -> Option<(String, ExtType)> {
        match argument {
            //No receivers in c code
            syn::FnArg::Receiver(_receiver) => (),
            syn::FnArg::Typed(pat_type) => {
                let pat = pat_type.pat.get_name();
                let ty = &pat_type.ty;
                let ty = ExtType::from_ty(&ty);
                if let (Some(p), Some(ty)) = (pat, ty) {
                    return Some((p, ty));
                }
            }
        }
        None
    }
    let arguments: Vec<_> = sig.inputs.iter().flat_map(extract_arg).collect();
    Function {
        name,
        arguments,
        output,
    }
}
//gguf_get_val_i8
fn any_int() -> proc_macro2::TokenStream {
    let bits = [8, 16, 32, 64];
    let names: Vec<_> = bits
        .map(|num| [format_ident!("u{num}"), format_ident!("i{num}")])
        .as_flattened()
        .to_vec();

    let stream = names.iter().map(|a| {
        quote! {
            impl From<#a> for AnyInt {
                fn from(value: #a) -> Self {
                    AnyInt(value as i64)
                }
            }
            impl From<AnyInt> for #a {
                fn from(value: AnyInt) -> Self {
                    value.0 as #a
                }
            }
        }
    });
    quote! {
        #[derive(Clone,Copy,Debug)]
        pub struct AnyInt(i64);
        #(#stream)*
    }
    //let signs = ['i','u'];
}

fn dyn_type_impl(ty: &syn::ItemEnum) -> Option<proc_macro2::TokenStream> {
    fn scalar_getter_fn(name: &str) -> Option<String> {
        let suffix = name.split("_").last()?;
        let r = match suffix {
            "BOOL" => "bool".to_string(),
            "ARRAY" => None?,
            "STRING" => None?,
            other if other.contains("UINT") => other.replace("UINT", "u"),
            other if other.contains("INT") => other.replace("INT", "i"),
            other if other.contains("FLOAT") => other.replace("FLOAT", "f"),
            _ => None?,
        };
        Some(r)
    }
    let _int_arms: Vec<_> = ty
        .variants
        .iter()
        .filter_map(|v| {
            let varn_str = v.ident.to_string();
            let sg = scalar_getter_fn(&varn_str)?;

            // Filter for u8..u64 and i8..i64
            if sg.starts_with('i') || sg.starts_with('u') {
                let varn = &v.ident;
                let getter = format_ident!("gguf_get_val_{sg}");
                Some(quote! {
                    gguf_type::#varn => Ok(Kv(|ctx, key_id| {
                        AnyInt::from(unsafe { #getter(ctx, key_id) })
                    })),
                })
            } else {
                None
            }
        })
        .collect();
    let item_name = ty.ident.to_string();
    if item_name.as_str() != "gguf_type" {
        return None;
    };
    let ggml_scalars = ["F32", "F64", "I8", "I16", "I32", "I64"];
    let ggml_scalar_impl = ggml_scalars.iter().map(|s| {
        let ggml_ty = format_ident!("GGML_TYPE_{}", s);
        let r_ty = format_ident!("{}", s.to_lowercase());

        quote! {
            impl GgmlScalar for #r_ty {
                type SelfT = #r_ty;
                const GGML_SCALAR: ggml_type = ggml_type::#ggml_ty;
            }
        }
    });
    let trait_impl = quote! {
        pub trait GgmlScalar {
            type SelfT;
            const GGML_SCALAR: ggml_type;
        }
        pub trait GgufScalar {
            type SelfT;
            const GGUF_SCALAR: gguf_type;
        }
        pub struct Kv<T>(unsafe extern "C" fn(*const gguf_context, i64) -> T);
        impl<T> Kv<T> {
            pub fn exec(&self, ctx: *const gguf_context, key_id: i64) -> T {
                unsafe { (self.0)(ctx, key_id) }
            }
        }
    };
    fn impl_trait(v: &syn::Variant) -> Option<proc_macro2::TokenStream> {
        let varn = v.ident.to_string();
        let sg = scalar_getter_fn(&varn)?;
        let ty = format_ident!("{sg}");

        let varn = format_ident!("{varn}");
        let getter = format_ident!("gguf_get_val_{sg}");

        println!("{varn} {ty:?},{getter:?}");
        let q = quote! {
            impl GgufScalar for #ty {
                type SelfT = #ty;
                const GGUF_SCALAR: gguf_type = gguf_type::#varn;
            }

            impl TryFrom<gguf_type> for Kv<#ty> {
                type Error = gguf_type;
                fn try_from(value: gguf_type) -> Result<Self, Self::Error> {
                    if value == gguf_type::#varn {
                        Ok(Kv(#getter))
                    } else {
                        Err(gguf_type::#varn)
                    }
                }
            }
        };
        //println!("{q:?}");
        Some(q)
    }
    let kv_impl = ty.variants.iter().flat_map(impl_trait);
    /*for v in &ty.variants {
        let varn = v.ident.to_string();
        if let Some(sg) = scalar_getter_fn(&varn) {
            impl_trait(varn, sg);
        }
    }*/
    Some(quote! {
        #trait_impl
        #(#kv_impl)*
        #(#ggml_scalar_impl)*
    })
}
#[proc_macro_attribute]
pub fn parse_ggml(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemMod);
    if input.content.is_none() {
        println!("No content")
    }
    //let mut dyn_type = quote! {};
    fn parse_item(item: &syn::Item) -> proc_macro2::TokenStream {
        //println!("{item:?}");
        match item {
            //syn::Item::Const(item_const) => todo!(),
            syn::Item::Enum(item_enum) => {
                if let Some(dt) = dyn_type_impl(item_enum) {
                    quote! {#item_enum #dt}
                } else {
                    quote! {#item_enum}
                }
                //let item_name = item_enum.ident.to_string();
                //println!("{item_name}");
            }
            syn::Item::ForeignMod(foreign) => {
                for i in &foreign.items {
                    if let syn::ForeignItem::Fn(func) = i {
                        let sig = &func.sig;
                        let func = extract_sig(sig);
                        //println!("{func:?}");
                    }
                    /*let name = match i {
                        syn::ForeignItem::Fn(func) => func.sig.ident.to_string(),
                        syn::ForeignItem::Static(st) => st.ident.to_string(),
                        syn::ForeignItem::Type(ty) => ty.ident.to_string(),
                        _ => String::new(),
                    };
                    println!("{name}")*/
                }
                quote! {#foreign}
            }
            //syn::Item::ExternCrate(item_extern_crate) => todo!(),
            //syn::Item::Fn(_item_fn) => {}
            //syn::Item::ForeignMod(item_foreign_mod) => todo!(),
            //syn::Item::Impl(item_impl) => todo!(),
            //syn::Item::Macro(item_macro) => todo!(),
            //syn::Item::Mod(item_mod) => todo!(),
            //syn::Item::Static(item_static) => todo!(),
            //syn::Item::Struct(item_struct) => todo!(),
            //syn::Item::Trait(item_trait) => todo!(),
            //syn::Item::TraitAlias(item_trait_alias) => todo!(),
            //syn::Item::Type(item_type) => todo!(),
            //syn::Item::Union(item_union) => todo!(),
            // syn::Item::Use(item_use) => todo!(),
            //syn::Item::Verbatim(token_stream) => todo!(),
            other => quote! {#other},
        }
    }
    let expanded = if let Some((_, content)) = &input.content {
        let any_int = any_int(); //
        //println!("Got {} items", content.len());
        let stream = content.iter().map(parse_item);
        quote! {

            #[allow(non_camel_case_types,non_snake_case,non_upper_case_globals)]
            pub mod ffi {
            #any_int
            #(#stream)*
            }
        }
    } else {
        quote! {
            #[allow(non_camel_case_types,non_snake_case,non_upper_case_globals)]
            pub mod ffi {#input}

        }
    };
    /*let name = &input.sig.ident;
    let block = &input.block;
    let vis = &input.vis;
    let sig = &input.sig;*/

    TokenStream::from(expanded)
}
