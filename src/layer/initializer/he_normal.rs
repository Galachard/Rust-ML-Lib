use crate::layer::initializer::{Initializer, Normal};

/// He Normal initializer
/// Initializes weights with values drawn from a normal distribution
/// with mean 0 and standard deviation sqrt(2 / fan_in)
pub struct HeNormal;

impl Initializer for HeNormal {
    fn init(&self, shape: &[usize]) -> Vec<f32> {
        let fan_in = shape[1] as f32;
        let std = (2.0 / fan_in).sqrt();

        Normal { mean: 0.0, std }.init(shape)
    }
}
