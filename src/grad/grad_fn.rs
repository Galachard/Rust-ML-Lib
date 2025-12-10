use crate::tensor::Tensor;

/// Trait required for gradient functions in the computation graph.
pub trait GradFn {
    /// Backpropagate gradient from output into parent tensors.
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;
}

/// Dummy GradFn for leaf tensors.
pub struct LeafGrad;
impl GradFn for LeafGrad {
    fn backward(&self, _grad_output: &Tensor) -> Vec<Tensor> {
        vec![]
    }
}
