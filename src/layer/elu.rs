use crate::layer::{Layer, SerializableLayer};
use crate::{Parameter, Tensor};
use bitcode;
use serde::{Deserialize, Serialize};
use std::cell::Cell;

/// Exponential Linear Unit (ELU) activation layer.
///
/// The ELU activation function is defined as:
/// - For x >= 0: f(x) = x
/// - For x < 0: f(x) = alpha * (exp(x) - 1)
/// # Example
/// ```
/// use ml_lib::layer::ELU;
/// let elu = ELU::new(1.0);
/// ```
pub struct ELU {
    alpha: Cell<f32>,
}

impl Layer for ELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        // ELU: x if x >= 0 else alpha * (exp(x) - 1)
        let alpha = self.alpha.get();

        input.map(
            move |x| {
                if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) }
            },
            move |x| {
                if x >= 0.0 { 1.0 } else { alpha * x.exp() }
            },
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }

    fn as_serializable(&self) -> Option<&dyn SerializableLayer> {
        Some(self)
    }
}

impl ELU {
    pub fn new(alpha: f32) -> Self {
        ELU {
            alpha: Cell::new(alpha),
        }
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self::new(1.0)
    }
}

#[derive(Serialize, Deserialize)]
struct ELUConfig {
    pub alpha: f32,
}

impl SerializableLayer for ELU {
    fn layer_type(&self) -> &'static str {
        "ELU"
    }

    fn serialize_config(&self) -> Vec<u8> {
        let cfg = ELUConfig {
            alpha: self.alpha.get(),
        };
        bitcode::serialize(&cfg).unwrap()
    }

    fn load_config(&self, data: &[u8]) {
        let cfg: ELUConfig = bitcode::deserialize(data).unwrap();
        self.alpha.set(cfg.alpha); // works because `alpha` is mutable
    }

    fn parameters_serializable(&self) -> Vec<(String, Tensor)> {
        vec![]
    }
    fn load_parameters(&self, _params: Vec<(String, Tensor)>) {}
}
