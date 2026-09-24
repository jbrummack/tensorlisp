use prost::Message;

use crate::mil::{
    Model,
    mil_spec::{Argument, Block, Operation},
    model::Type,
};

pub fn load_cml() -> anyhow::Result<()> {
    let buffer = std::fs::read("./models/model.mlmodel")?;
    let model = Model::decode(buffer.as_slice())?;
    let desc = model.description.unwrap();

    /*match &ty {
        Type::MlProgram(program) => println!("PROGRAM"),
        Type::SerializedModel(serialized_model) => println!("SERIALISED MODEL"),
        _ => (),
    }*/
    let program = if let Some(Type::MlProgram(prog)) = model.r#type {
        prog
    } else {
        anyhow::bail!("not a program")
    };
    //    println!("{desc:#?}");
    for (k, v) in program.functions {
        println!("{k}");
        //println!("{v:#?}");
        let a = v.block_specializations;
        for (k, bs) in a {
            println!("{k}");
            for o in bs.operations {
                let vname = o.get_outputs();
                let vname = vname.first().as_ref().map(|i| i.as_str()).unwrap_or("");
                println!("(let {vname} ({} {}))", o.r#type, o.get_inputs().join(" "));
            }
        }
        //println!("{a:#?}");
    }

    Ok(())
}
/*impl Argument {
    pub fn format(&self) -> String {

    }
}*/
impl Operation {
    pub fn get_inputs(&self) -> Vec<String> {
        self.inputs.iter().map(|i| i.0.clone()).collect()
    }
    pub fn get_outputs(&self) -> Vec<String> {
        self.outputs.iter().map(|i| i.name.clone()).collect()
    }
}
