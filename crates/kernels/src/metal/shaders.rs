//! Vendored Metal sources (see ../../LICENSE-mistral-rs) and the include merger.
//!
//! Runtime compilation can't resolve `#include "x.metal"`, so each module is
//! flattened into one translation unit (the same trick ggml-sys uses for
//! ggml-metal.metal). Template instantiations are *not* in the vendored files:
//! the op layer appends exactly the ones it needs, because compiling the
//! upstream cross product (288 paged-attention variants) takes minutes.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Module {
    PagedAttention,
    ReshapeAndCache,
    CopyBlocks,
    GatherKvCache,
}

impl Module {
    pub fn name(self) -> &'static str {
        match self {
            Module::PagedAttention => "pagedattention",
            Module::ReshapeAndCache => "reshape_and_cache",
            Module::CopyBlocks => "copy_blocks",
            Module::GatherKvCache => "gather_kv_cache",
        }
    }

    fn text(self) -> &'static str {
        match self {
            Module::PagedAttention => include_str!("shaders/pagedattention.metal"),
            Module::ReshapeAndCache => include_str!("shaders/reshape_and_cache.metal"),
            Module::CopyBlocks => include_str!("shaders/copy_blocks.metal"),
            Module::GatherKvCache => include_str!("shaders/gather_kv_cache.metal"),
        }
    }

    /// The flattened source followed by `instantiations`.
    pub fn source(self, instantiations: &str) -> String {
        let mut seen = Vec::new();
        let mut out = String::new();
        flatten(self.text(), &mut seen, &mut out);
        out.push('\n');
        out.push_str(instantiations);
        out
    }
}

fn header(name: &str) -> Option<&'static str> {
    match name {
        "utils.metal" => Some(include_str!("shaders/utils.metal")),
        "float8.metal" => Some(include_str!("shaders/float8.metal")),
        _ => None,
    }
}

fn flatten(text: &str, seen: &mut Vec<String>, out: &mut String) {
    for line in text.lines() {
        let inc = line.trim().strip_prefix("#include \"").and_then(|r| r.strip_suffix('"'));
        match inc {
            Some(name) => {
                if seen.iter().any(|s| s == name) {
                    continue;
                }
                seen.push(name.to_string());
                let h = header(name).unwrap_or_else(|| panic!("unknown shader include {name}"));
                flatten(h, seen, out);
            }
            None => {
                out.push_str(line);
                out.push('\n');
            }
        }
    }
}
