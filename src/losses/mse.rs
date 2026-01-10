use crate::Tensor;
use crate::ops::{mul, scalar_mul, sub, sum};

pub fn mse(predicted: &Tensor, target: &Tensor) -> Tensor {
    let diff = sub(predicted, target).unwrap();
    let squared_diff = mul(&diff, &diff).unwrap();
    scalar_mul(&sum(&squared_diff), 1.0 / predicted.data.len() as f32).unwrap()
}
