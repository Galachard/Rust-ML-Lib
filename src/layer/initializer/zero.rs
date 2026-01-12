use crate::layer::initializer::Initializer;

/// An initializer that sets all weights to zero.
pub struct Zeros;

impl Initializer for Zeros {
    fn init(&self, shape: &[usize]) -> Vec<f32> {
        let size = shape.iter().product();
        vec![0.0; size]
    }
}
