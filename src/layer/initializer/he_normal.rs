use crate::layer::initializer::{Initializer, Normal};

pub struct HeNormal;

impl Initializer for HeNormal {
    fn init(&self, shape: &[usize]) -> Vec<f32> {
        let fan_in = shape[1] as f32;
        let std = (2.0 / fan_in).sqrt();

        Normal { mean: 0.0, std }.init(shape)
    }
}
