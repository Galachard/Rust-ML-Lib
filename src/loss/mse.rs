use crate::Tensor;
use crate::ops::{mul, scalar_mul, sub, sum};

/// Computes the Mean Squared Error (MSE) between the predicted and target tensors.
/// # Arguments
/// * `predicted` - A reference to the tensor containing predicted values.
/// * `target` - A reference to the tensor containing target values.
/// # Returns
/// A tensor representing the Mean Squared Error.
pub fn mse(predicted: &Tensor, target: &Tensor) -> Tensor {
    let diff = sub(predicted, target).unwrap();
    let squared_diff = mul(&diff, &diff).unwrap();
    scalar_mul(&sum(&squared_diff), 1.0 / predicted.data.len() as f32).unwrap()
}
