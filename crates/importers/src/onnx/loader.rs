use std::collections::HashMap;

use prost::Message;

use crate::onnx::onnx::{
    AttributeProto, ModelProto, StringStringEntryProto, TensorProto,
    attribute_proto::AttributeType, tensor_proto::DataType,
};
#[derive(Debug)]
pub enum Attribute {
    Undefined,
    Float(f32),
    FVec(Vec<f32>),
    Int(i64),
    IVec(Vec<i64>),
    String(Vec<u8>),
    Tensor(Option<TensorProto>),
    NotImplemented(AttributeType),
}
impl Attribute {
    pub fn typename(&self) -> String {
        match self {
            Attribute::Undefined => "any".into(),
            Attribute::Float(_) => "Float".into(),
            Attribute::FVec(items) => format!("[Float;{}]", items.len()),
            Attribute::Int(_) => "Int".into(),
            Attribute::IVec(items) => format!("[Int;{}]", items.len()),
            Attribute::String(items) => "String".into(),
            Attribute::Tensor(tensor_proto) => "Tensor".into(),
            Attribute::NotImplemented(attribute_type) => "any".into(),
        }
    }
    pub fn value(&self) -> String {
        match self {
            Attribute::Undefined => "Any".into(),
            Attribute::Float(f) => format!("{f}"),
            Attribute::FVec(items) => format!("{items:?}"),
            Attribute::Int(i) => format!("{i}"),
            Attribute::IVec(items) => format!("{items:?}"),
            Attribute::String(_items) => format!("String"),
            Attribute::Tensor(tensor_proto) => tensor_proto
                .as_ref()
                .map(|i| i.name.clone())
                .unwrap_or(String::from("Tensor")),
            Attribute::NotImplemented(_attribute_type) => "Any".into(),
        }
    }
}

#[derive(Debug)]
pub struct Attributes {
    _inner: HashMap<String, Attributes>,
}

impl AttributeProto {
    pub fn extract(self) -> (String, Attribute) {
        let a = match self.r#type() {
            super::onnx::attribute_proto::AttributeType::Undefined => Attribute::Undefined,
            super::onnx::attribute_proto::AttributeType::Float => Attribute::Float(self.f),
            super::onnx::attribute_proto::AttributeType::Int => Attribute::Int(self.i),
            super::onnx::attribute_proto::AttributeType::String => Attribute::String(self.s),
            super::onnx::attribute_proto::AttributeType::Tensor => Attribute::Tensor(self.t),
            super::onnx::attribute_proto::AttributeType::Floats => Attribute::FVec(self.floats),
            super::onnx::attribute_proto::AttributeType::Ints => Attribute::IVec(self.ints),
            /*super::onnx::attribute_proto::AttributeType::Graph => todo!(),
            super::onnx::attribute_proto::AttributeType::SparseTensor => todo!(),
            super::onnx::attribute_proto::AttributeType::TypeProto => todo!(),

            super::onnx::attribute_proto::AttributeType::Strings => todo!(),
            super::onnx::attribute_proto::AttributeType::Tensors => todo!(),
            super::onnx::attribute_proto::AttributeType::Graphs => todo!(),
            super::onnx::attribute_proto::AttributeType::SparseTensors => todo!(),
            super::onnx::attribute_proto::AttributeType::TypeProtos => todo!(),*/
            other => Attribute::NotImplemented(other),
        };
        let n = self.name;

        (n, a)
    }
}
pub struct Tensor {
    shape: Vec<i64>,
    tdata: TensorData,
}
pub enum TensorData {
    Float(Vec<f32>),
    Raw {
        ty: DataType,
        buf: Vec<u8>,
    },
    Extern {
        loc: Vec<StringStringEntryProto>,
        ty: i32,
    },
}
pub fn load_onnx() -> anyhow::Result<()> {
    let buffer = std::fs::read("./models/model_fp16.onnx")?;
    let proto = ModelProto::decode(buffer.as_slice())?;
    drop(buffer);
    let mut tensors = HashMap::new();
    if let Some(graph) = proto.graph {
        for tensor in graph.initializer {
            match tensor.data_location() {
                super::onnx::tensor_proto::DataLocation::Default => {
                    //match tensor.data_type
                    if let Some(tdata) = match DataType::try_from(tensor.data_type) {
                        Ok(DataType::Float) => Some(TensorData::Float(tensor.float_data)),
                        Ok(other) => {
                            Some(TensorData::Raw {
                                ty: other,
                                buf: tensor.raw_data,
                            })
                            //tensor.raw_data
                        }
                        _ => None,
                    } {
                        let shape = tensor.dims;
                        tensors.insert(tensor.name, Tensor { shape, tdata });
                    };
                }
                super::onnx::tensor_proto::DataLocation::External => {
                    let shape = tensor.dims;
                    tensors.insert(
                        tensor.name,
                        Tensor {
                            shape,
                            tdata: TensorData::Extern {
                                loc: tensor.external_data,
                                ty: tensor.data_type,
                            },
                        },
                    );
                }
            };
        }
        for node in graph.node {
            let mut attributes = HashMap::with_capacity(node.attribute.len());
            for a in node.attribute {
                let (k, v) = a.extract();
                attributes.insert(k, v);
                /*if let Attribute::Tensor(Some(t)) = v {
                    println!("{k}");
                    tensors.insert(
                        k,
                        Tensor {
                            shape: t.dims,
                            tdata: TensorData::Float(Vec::new()),
                        },
                    );
                } else {

                }*/
            }
            let op = node.op_type;
            /*for i in node.input {
                let shape: Option<&[i64]> = tensors.get(&i).map(|s| s.shape.as_ref());
                println!("{op}: {i}({:?})", shape.unwrap_or(&[]));
            }*/
            if let Some(func) = node.output.first() {
                /*let params: Vec<String> = attributes
                .iter()
                .map(|(k, v)| format!("{k}:{}", v.typename()))
                .collect();*/
                let values: Vec<String> = attributes.values().map(|v| v.value()).collect();
                let inputs = node.input.join(" ");

                println!("(let {func} ({op} {inputs} {}))", values.join(" "));
            }
            println!("{op} {attributes:#?}");
        }
    }
    /*for (k, v) in tensors.iter() {
        println!("{k}:{:?}", v.shape);
    }*/
    //println!("{:?}", model.graph);
    Ok(())
}
