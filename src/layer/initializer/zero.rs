use crate::layer::initializer::Initializer;

pub struct Zeros;

impl Initializer for Zeros {
    fn init(&self, shape: &[usize]) -> Vec<f32> {
        let size = shape.iter().product();
        vec![0.0; size]
    }
}
