use crate::layer::{Layer, SerializableLayer};
use crate::ops::{exp, inv, scalar_add, scalar_mul};
use crate::{Parameter, Tensor};

pub struct Sigmoid;

impl Layer for Sigmoid {
    fn forward(&self, input: &Tensor) -> Tensor {
        // Sigmoid function: 1 / (1 + exp(-x))
        let exponent = exp(&scalar_mul(input, -1.0).unwrap()).unwrap();
        let denominator = scalar_add(&exponent, 1.0).unwrap();
        inv(&denominator).unwrap()
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }

    fn as_serializable(&self) -> Option<&dyn SerializableLayer> {
        Some(self)
    }
}

impl SerializableLayer for Sigmoid {
    fn layer_type(&self) -> &'static str {
        "Sigmoid"
    }

    fn serialize_config(&self) -> Vec<u8> {
        vec![]
    }
    fn load_config(&self, _data: &[u8]) {}
    fn parameters_serializable(&self) -> Vec<(String, Tensor)> {
        vec![]
    }

    fn load_parameters(&self, _params: Vec<(String, Tensor)>) {}
}
