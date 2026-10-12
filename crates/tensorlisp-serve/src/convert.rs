//! Wire tensors <-> the runtime's f32 arrays. The runtime computes in f32, so every dtype is
//! converted on the way in; on the way out the client picks f32, f16 or bf16 (`outputDtype`).
use half::{bf16, f16};
use ndarray::ArrayD;

use crate::tensorlisp_capnp::{DType, tensor};

type Result<T> = std::result::Result<T, capnp::Error>;

fn fail<T>(msg: impl Into<String>) -> Result<T> {
    Err(capnp::Error::failed(msg.into()))
}

/// Bytes per element of the dtypes that carry plain scalars.
pub fn elem_size(dtype: DType) -> Option<usize> {
    Some(match dtype {
        DType::F32 | DType::I32 => 4,
        DType::F16 | DType::Bf16 | DType::I16 => 2,
        DType::I64 => 8,
        DType::I8 | DType::U8 | DType::Bool => 1,
        _ => return None,
    })
}

pub fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::Bf16 => "bf16",
        DType::I32 => "i32",
        DType::I64 => "i64",
        DType::I16 => "i16",
        DType::I8 => "i8",
        DType::U8 => "u8",
        DType::Bool => "bool",
        _ => "quantized",
    }
}

/// Decodes little-endian `data` of `dtype` to f32.
pub fn decode(dtype: DType, data: &[u8]) -> Result<Vec<f32>> {
    let Some(size) = elem_size(dtype) else {
        return fail(format!("tensors of dtype {} can't be sent as inputs (use f32, f16, bf16 or an integer type)", dtype_name(dtype)));
    };
    if data.len() % size != 0 {
        return fail(format!("{} bytes is not a whole number of {} elements", data.len(), dtype_name(dtype)));
    }
    let chunks = data.chunks_exact(size);
    Ok(match dtype {
        DType::F32 => chunks.map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
        DType::F16 => chunks.map(|c| f16::from_le_bytes(c.try_into().unwrap()).to_f32()).collect(),
        DType::Bf16 => chunks.map(|c| bf16::from_le_bytes(c.try_into().unwrap()).to_f32()).collect(),
        DType::I32 => chunks.map(|c| i32::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
        DType::I64 => chunks.map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
        DType::I16 => chunks.map(|c| i16::from_le_bytes(c.try_into().unwrap()) as f32).collect(),
        DType::I8 => chunks.map(|c| c[0] as i8 as f32).collect(),
        DType::U8 => chunks.map(|c| c[0] as f32).collect(),
        DType::Bool => chunks.map(|c| (c[0] != 0) as u8 as f32).collect(),
        _ => unreachable!(),
    })
}

/// Encodes `values` as little-endian `dtype`; only the float types are produced.
pub fn encode(dtype: DType, values: &[f32]) -> Result<Vec<u8>> {
    Ok(match dtype {
        DType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        DType::F16 => values.iter().flat_map(|&v| f16::from_f32(v).to_le_bytes()).collect(),
        DType::Bf16 => values.iter().flat_map(|&v| bf16::from_f32(v).to_le_bytes()).collect(),
        other => return fail(format!("results can be sent as f32, f16 or bf16, not {}", dtype_name(other))),
    })
}

pub fn array_from_parts(dtype: DType, dims: &[u64], data: &[u8]) -> Result<ArrayD<f32>> {
    let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
    let values = decode(dtype, data)?;
    let expected = shape.iter().product::<usize>();
    if values.len() != expected {
        return fail(format!("dims {shape:?} need {expected} elements but the data holds {}", values.len()));
    }
    ArrayD::from_shape_vec(shape, values).map_err(|e| capnp::Error::failed(e.to_string()))
}

pub fn read_tensor(t: tensor::Reader<'_>) -> Result<ArrayD<f32>> {
    let dims: Vec<u64> = t.get_dims()?.iter().collect();
    array_from_parts(t.get_dtype()?, &dims, t.get_data()?)
}

pub fn write_tensor(mut out: tensor::Builder<'_>, array: &ArrayD<f32>, dtype: DType) -> Result<()> {
    let values: Vec<f32> = array.iter().copied().collect();
    out.set_dtype(dtype);
    let mut dims = out.reborrow().init_dims(array.ndim() as u32);
    for (i, &d) in array.shape().iter().enumerate() {
        dims.set(i as u32, d as u64);
    }
    out.set_data(&encode(dtype, &values)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip() {
        let values = [0.0f32, 1.0, -2.5, 0.333251953125, 65504.0];
        let bytes = encode(DType::F16, &values).unwrap();
        assert_eq!(bytes.len(), values.len() * 2);
        assert_eq!(decode(DType::F16, &bytes).unwrap(), values);
    }

    #[test]
    fn integers_and_bad_lengths() {
        assert_eq!(decode(DType::I32, &[1, 0, 0, 0, 255, 255, 255, 255]).unwrap(), [1.0, -1.0]);
        assert!(decode(DType::F32, &[0, 0, 0]).is_err());
        assert!(decode(DType::Q80, &[0; 34]).is_err());
        assert!(array_from_parts(DType::U8, &[2, 2], &[1, 2, 3]).is_err());
    }
}
