use crate::Tensor;
use crate::grad::GradFn;

/// Dummy GradFn for leaf tensors.
pub struct LeafGrad;
impl GradFn for LeafGrad {
    fn backward(&self, _grad_output: &Tensor) -> Vec<Tensor> {
        vec![]
    }
}
