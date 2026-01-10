use crate::layer::Layer;
use crate::layer::initializer::Initializer;
use crate::ops::{add, matmul};
use crate::{Parameter, Tensor};

pub struct Linear {
    pub weight: Parameter,
    pub bias: Parameter,
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

        Self { weight, bias }
    }
}
