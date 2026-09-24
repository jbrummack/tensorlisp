//! Minimal .npy reading (numeric dtypes, converted to f32) and writing (f32).
use std::path::Path;

use anyhow::{Context, Result, bail};
use ndarray::{ArrayD, IxDyn, ShapeBuilder};

const MAGIC: &[u8] = b"\x93NUMPY";

struct Header {
    descr: String,
    fortran_order: bool,
    shape: Vec<usize>,
}

/// Value of `'key': value` in the header dict, up to the next top-level comma.
fn dict_value<'a>(dict: &'a str, key: &str) -> Option<&'a str> {
    let start = dict.find(&format!("'{key}'"))? + key.len() + 2;
    let rest = dict[start..].trim_start().strip_prefix(':')?.trim_start();
    let end = if rest.starts_with('(') { rest.find(')')? + 1 } else { rest.find([',', '}'])? };
    Some(rest[..end].trim())
}

fn parse_header(dict: &str) -> Result<Header> {
    let descr = dict_value(dict, "descr").context("npy header has no descr")?.trim_matches(['\'', '"']).to_string();
    let fortran_order = dict_value(dict, "fortran_order").context("npy header has no fortran_order")? == "True";
    let shape = dict_value(dict, "shape")
        .context("npy header has no shape")?
        .trim_matches(['(', ')'])
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().with_context(|| format!("bad npy shape entry {s:?}")))
        .collect::<Result<_>>()?;
    Ok(Header { descr, fortran_order, shape })
}

fn decode(descr: &str, data: &[u8]) -> Result<Vec<f32>> {
    macro_rules! le {
        ($t:ty) => {
            data.chunks_exact(size_of::<$t>()).map(|b| <$t>::from_le_bytes(b.try_into().unwrap()) as f32).collect()
        };
    }
    Ok(match descr {
        "<f4" => le!(f32),
        "<f8" => le!(f64),
        "<f2" => data.chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
        "<i8" => le!(i64),
        "<i4" => le!(i32),
        "<i2" => le!(i16),
        "|i1" => data.iter().map(|&b| b as i8 as f32).collect(),
        "|u1" | "|b1" => data.iter().map(|&b| b as f32).collect(),
        "<u2" => le!(u16),
        "<u4" => le!(u32),
        other => bail!("unsupported npy dtype {other:?} (supported: f2/f4/f8, i1/i2/i4/i8, u1/u2/u4, b1)"),
    })
}

/// Reads a .npy file as f32.
pub fn read(path: &Path) -> Result<ArrayD<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let what = || format!("{} is not a valid .npy file", path.display());
    if !bytes.starts_with(MAGIC) || bytes.len() < 10 {
        bail!(what());
    }
    let (header_len, header_start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, 12),
        v => bail!("{}: unsupported npy version {v}", path.display()),
    };
    let dict = std::str::from_utf8(bytes.get(header_start..header_start + header_len).with_context(what)?)
        .with_context(what)?;
    let header = parse_header(dict).with_context(what)?;
    let values = decode(&header.descr, &bytes[header_start + header_len..])?;
    let n: usize = header.shape.iter().product();
    if values.len() != n {
        bail!("{}: expected {n} values for shape {:?}, found {}", path.display(), header.shape, values.len());
    }
    let shape = IxDyn(&header.shape);
    let array = if header.fortran_order {
        ArrayD::from_shape_vec(shape.f(), values)
    } else {
        ArrayD::from_shape_vec(shape, values)
    };
    Ok(array?)
}

/// Writes an f32 array as a version 1.0 .npy file.
pub fn write(path: &Path, array: &ArrayD<f32>) -> Result<()> {
    let shape = match array.shape() {
        [n] => format!("({n},)"),
        dims => format!("({})", dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape}, }}");
    // Magic (6) + version (2) + length (2) + header + newline, padded to 64 bytes.
    let total = (10 + header.len() + 1).next_multiple_of(64);
    header.push_str(&" ".repeat(total - 10 - header.len() - 1));
    header.push('\n');

    let mut out = Vec::with_capacity(total + array.len() * 4);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in array.as_standard_layout().iter() {
        out.extend_from_slice(&v.to_le_bytes());
    }
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let dir = std::env::temp_dir().join(format!("tl-npy-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for shape in [vec![3], vec![2, 3], vec![1, 2, 3, 2]] {
            let n = shape.iter().product::<usize>();
            let a = ArrayD::from_shape_vec(IxDyn(&shape), (0..n).map(|i| i as f32 * 0.5).collect()).unwrap();
            let path = dir.join("a.npy");
            write(&path, &a).unwrap();
            assert_eq!(read(&path).unwrap(), a);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn header_parsing() {
        let h = parse_header("{'descr': '<f8', 'fortran_order': True, 'shape': (4, 5), }").unwrap();
        assert_eq!((h.descr.as_str(), h.fortran_order, h.shape), ("<f8", true, vec![4, 5]));
        let h = parse_header("{'descr': '<i8', 'fortran_order': False, 'shape': (), }").unwrap();
        assert!(h.shape.is_empty());
    }
}
