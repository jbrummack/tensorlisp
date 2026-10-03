//! LoRA adapters on disk, in PEFT's layout: `adapter_model.safetensors` with the torch-layout
//! tensors `<key_prefix><module>.lora_A.weight` ([r, in]) and `.lora_B.weight` ([out, r]), and an
//! `adapter_config.json`, so an adapter trained here loads in `peft` and the other way round.
//!
//! The program's parameters are named `lora.<module>.A` / `.B` (see `lora-attach!` in core.ss).

use std::{collections::BTreeSet, path::Path};

use ndarray::{ArrayD, IxDyn};
use safetensors::{Dtype, SafeTensors, tensor::TensorView};

use super::Model;
use crate::error::{Error, Result};

/// How adapters are named on disk.
#[derive(Debug, Clone)]
pub struct AdapterFormat {
    /// In front of the module path of every tensor: PEFT's `base_model.model.` plus the base
    /// checkpoint's own prefix (`model.` for T5Gemma 2).
    pub key_prefix: String,
    /// `lora_alpha` of the adapter config (the program's `lora-alpha`).
    pub alpha: f32,
    /// `base_model_name_or_path` of the adapter config.
    pub base_model: String,
}

impl Default for AdapterFormat {
    fn default() -> Self {
        AdapterFormat { key_prefix: "base_model.model.model.".into(), alpha: 16.0, base_model: String::new() }
    }
}

/// `lora.<module>.A` -> (`<module>`, "lora_A").
fn split_param(name: &str) -> Option<(&str, &'static str)> {
    let rest = name.strip_prefix("lora.")?;
    if let Some(m) = rest.strip_suffix(".A") {
        Some((m, "lora_A"))
    } else {
        rest.strip_suffix(".B").map(|m| (m, "lora_B"))
    }
}

impl Model {
    /// Writes the program's LoRA parameters to `dir` as PEFT files.
    pub fn save_adapter(&self, dir: &Path, format: &AdapterFormat) -> Result<()> {
        let mut tensors: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        let mut targets = BTreeSet::new();
        let mut rank = 0;
        for (name, _) in self.params() {
            let Some((module, which)) = split_param(&name) else { continue };
            let value = self.state(&name)?;
            if which == "lora_A" {
                rank = value.shape()[0];
            }
            targets.insert(module.rsplit('.').next().unwrap_or(module).to_string());
            let bytes: Vec<u8> = value.as_standard_layout().iter().flat_map(|v| v.to_le_bytes()).collect();
            tensors.push((format!("{}{module}.{which}.weight", format.key_prefix), value.shape().to_vec(), bytes));
        }
        if tensors.is_empty() {
            return Err(Error::Program("the program has no LoRA parameters (lora.<module>.A / .B)".into()));
        }
        tensors.sort_by(|a, b| a.0.cmp(&b.0));
        let views = tensors
            .iter()
            .map(|(n, shape, bytes)| {
                TensorView::new(Dtype::F32, shape.clone(), bytes).map(|v| (n.clone(), v)).map_err(|e| Error::Backend(e.to_string()))
            })
            .collect::<Result<Vec<_>>>()?;
        std::fs::create_dir_all(dir)?;
        safetensors::serialize_to_file(views, None, &dir.join("adapter_model.safetensors"))
            .map_err(|e| Error::Backend(e.to_string()))?;
        let config = serde_json::json!({
            "peft_type": "LORA",
            "task_type": "SEQ_2SEQ_LM",
            "base_model_name_or_path": format.base_model,
            "r": rank,
            "lora_alpha": format.alpha,
            "lora_dropout": 0.0,
            "bias": "none",
            "target_modules": targets.into_iter().collect::<Vec<_>>(),
            "fan_in_fan_out": false,
            "inference_mode": true,
        });
        std::fs::write(dir.join("adapter_config.json"), serde_json::to_string_pretty(&config).unwrap())?;
        Ok(())
    }

    /// Loads an adapter written by [`Model::save_adapter`] (its directory, or the `.safetensors` file)
    /// into the program's LoRA parameters. Every parameter must be in the file, with its shape.
    /// Returns how many tensors were loaded.
    pub fn load_adapter(&self, path: &Path, format: &AdapterFormat) -> Result<usize> {
        let file = if path.is_dir() { path.join("adapter_model.safetensors") } else { path.to_path_buf() };
        let bytes = std::fs::read(&file)?;
        let tensors = SafeTensors::deserialize(&bytes).map_err(|e| Error::Input(format!("{}: {e}", file.display())))?;
        let mut loaded = 0;
        for (name, _) in self.params() {
            let Some((module, which)) = split_param(&name) else { continue };
            let key = format!("{}{module}.{which}.weight", format.key_prefix);
            let t = tensors.tensor(&key).map_err(|_| Error::Input(format!("{} has no tensor {key}", file.display())))?;
            if t.dtype() != Dtype::F32 {
                return Err(Error::Input(format!("{key} is {:?}, expected F32", t.dtype())));
            }
            let data: Vec<f32> = t.data().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
            let value = ArrayD::from_shape_vec(IxDyn(t.shape()), data).map_err(|e| Error::Input(e.to_string()))?;
            self.set_state(&name, &value)?;
            loaded += 1;
        }
        Ok(loaded)
    }
}
