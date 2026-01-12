/// Defines the Initializer trait for initializing weights in neural network layers.
pub trait Initializer {
    /// Initializes weights based on the provided shape.
    /// # Arguments
    /// * `shape` - A slice representing the dimensions of the weights to be initialized.
    /// # Returns
    /// A vector of f32 values representing the initialized weights.
    fn init(&self, shape: &[usize]) -> Vec<f32>;
}
