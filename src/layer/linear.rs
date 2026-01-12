use crate::layer::initializer::{Initializer, Zeros};
use crate::layer::{Layer, SerializableLayer};
use crate::ops::{add, matmul};
use crate::{Parameter, Tensor};
use bitcode;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;

/// A fully connected linear layer.
/// # Example
/// ```
/// use ml_lib::layer::{Linear, Layer};
/// use ml_lib::layer::initializer::Zeros;
/// use ml_lib::Tensor;
/// let layer = Linear::new(3, 2, &Zeros, &Zeros);
/// let input = Tensor::from_vec_leaf(vec![1.0, 2.0, 3.0], vec![3, 1]);
/// let output = layer.forward(&input);
/// assert_eq!(output.shape()[0], 2);
/// ```
pub struct Linear {
    pub weight: Parameter,
    pub bias: Parameter,
    config: RefCell<LinearConfig>,
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        let weight = self.weight.tensor.borrow().clone();
        let bias = self.bias.tensor.borrow().clone();
        add(&matmul(&weight, input).unwrap(), &bias).unwrap()
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn as_serializable(&self) -> Option<&dyn SerializableLayer> {
        Some(self)
    }
}

impl Linear {
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        weight_init: &dyn Initializer,
        bias_init: &dyn Initializer,
    ) -> Self {
        let weight_shape = vec![output_dim, input_dim];
        let bias_shape = vec![output_dim, 1];

        let weight = Parameter::from_vec(weight_init.init(&weight_shape), weight_shape);

        let bias = Parameter::from_vec(bias_init.init(&bias_shape), bias_shape);

        Self {
            weight,
            bias,
            config: RefCell::new(LinearConfig {
                in_dim: input_dim,
                out_dim: output_dim,
            }),
        }
    }

    pub fn dummy() -> Self {
        Self::new(0, 0, &Zeros, &Zeros)
    }
}

#[derive(Serialize, Deserialize)]
struct LinearConfig {
    in_dim: usize,
    out_dim: usize,
}

impl SerializableLayer for Linear {
    fn layer_type(&self) -> &'static str {
        "Linear"
    }

    fn serialize_config(&self) -> Vec<u8> {
        let cfg = self.config.borrow();
        bitcode::serialize(&*cfg).unwrap()
    }

    fn load_config(&self, data: &[u8]) {
        let cfg: LinearConfig = bitcode::deserialize(data).unwrap();
        *self.config.borrow_mut() = cfg;
    }

    fn parameters_serializable(&self) -> Vec<(String, Tensor)> {
        vec![
            ("weight".into(), self.weight.tensor.borrow().clone()),
            ("bias".into(), self.bias.tensor.borrow().clone()),
        ]
    }

    fn load_parameters(&self, params: Vec<(String, Tensor)>) {
        for (name, t) in params {
            match name.as_str() {
                "weight" => *self.weight.tensor.borrow_mut() = t,
                "bias" => *self.bias.tensor.borrow_mut() = t,
                _ => panic!("Unknown param {}", name),
            }
        }
    }
}
