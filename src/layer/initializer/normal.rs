use crate::layer::initializer::Initializer;
use rand::rng;
use rand_distr::{Distribution, Normal as RandNormal};

/// Initializes weights with values drawn from a normal (Gaussian) distribution
/// with specified mean and standard deviation.
pub struct Normal {
    pub mean: f32,
    pub std: f32,
}

impl Initializer for Normal {
    fn init(&self, shape: &[usize]) -> Vec<f32> {
        let size: usize = shape.iter().product();
        let mut rng = rng();

        let normal = RandNormal::new(self.mean, self.std)
            .expect("Invalid parameters for Normal distribution");

        (0..size).map(|_| normal.sample(&mut rng)).collect()
    }
}
