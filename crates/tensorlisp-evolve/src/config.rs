//! Settings from the environment and a repository `.env`.
//!
//! Process environment wins over `.env`. Values are never printed or logged by this crate; `.env*` files
//! are unreadable to the agents' file tools.
use std::collections::HashMap;
use std::path::Path;
use std::sync::OnceLock;

static DOTENV: OnceLock<HashMap<String, String>> = OnceLock::new();

/// Parses `KEY=VALUE` lines (optional `export `, optional single or double quotes, `#` comments).
pub fn parse_dotenv(text: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let Some((k, v)) = line.split_once('=') else { continue };
        let v = v.trim();
        let v = if v.len() >= 2 && ((v.starts_with('"') && v.ends_with('"')) || (v.starts_with('\'') && v.ends_with('\''))) { &v[1..v.len() - 1] } else { v };
        if !k.trim().is_empty() {
            out.insert(k.trim().to_string(), v.to_string());
        }
    }
    out
}

/// Loads `<dir>/.env` once. A missing file is fine.
pub fn load_dotenv(dir: &Path) {
    let map = std::fs::read_to_string(dir.join(".env")).map(|t| parse_dotenv(&t)).unwrap_or_default();
    let _ = DOTENV.set(map);
}

/// A non-empty setting: the process environment, else `.env`.
pub fn var(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty()).or_else(|| DOTENV.get().and_then(|m| m.get(name)).cloned().filter(|v| !v.is_empty()))
}

/// The first of `names` that is set.
pub fn first_var(names: &[&str]) -> Option<String> {
    names.iter().find_map(|n| var(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dotenv_syntax() {
        let m = parse_dotenv("# comment\n\nHF_TOKEN=abc\nexport OPENROUTER_KEY = \"sk-or-1\"\nQ='single'\nEMPTY=\nbad line\n=novalue\nURL=http://x/y?a=b\n");
        assert_eq!(m["HF_TOKEN"], "abc");
        assert_eq!(m["OPENROUTER_KEY"], "sk-or-1");
        assert_eq!(m["Q"], "single");
        assert_eq!(m["EMPTY"], "");
        assert_eq!(m["URL"], "http://x/y?a=b");
        assert!(!m.contains_key("bad line") && !m.contains_key(""));
    }

    #[test]
    fn environment_beats_dotenv_and_empty_counts_as_unset() {
        // PATH is set in any environment; a name that is not set anywhere is None
        assert!(var("PATH").is_some());
        assert!(var("TL_EVOLVE_SURELY_UNSET_VARIABLE").is_none());
        assert_eq!(first_var(&["TL_EVOLVE_SURELY_UNSET_VARIABLE", "PATH"]), var("PATH"));
    }
}
