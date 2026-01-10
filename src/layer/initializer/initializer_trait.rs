pub trait Initializer {
    fn init(&self, shape: &[usize]) -> Vec<f32>;
}
