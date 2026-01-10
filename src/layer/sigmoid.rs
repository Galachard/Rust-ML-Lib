use crate::layer::Layer;
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
}
