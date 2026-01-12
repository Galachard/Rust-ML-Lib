/// Defines the Optimizer trait for optimization algorithms.
pub trait Optimizer {
    /// Performs a single optimization step.
    fn step(&mut self);
    /// Resets the gradients of all optimized parameters to zero.
    fn zero_grad(&mut self);
}
