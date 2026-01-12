use crate::Parameter;
use crate::optimizer::Optimizer;

/// Stochastic Gradient Descent (SGD) optimizer
pub struct SGD {
    params: Vec<Parameter>,
    pub lr: f32,
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for param in &self.params {
            // Extract gradient
            let grad = {
                let tensor = param.tensor.borrow();

                let node = tensor.node.as_ref().expect("Parameter tensor has no node");

                node.borrow().grad.as_ref().cloned()
            };

            let grad = match grad {
                Some(g) => g,
                None => continue,
            };

            // Apply gradient descent update
            let mut tensor = param.tensor.borrow_mut();

            // TODO: Do I want that?
            assert_eq!(
                tensor.data.len(),
                grad.data.len(),
                "Gradient shape mismatch in SGD::step"
            );

            for (w, g) in tensor.data.iter_mut().zip(&grad.data) {
                *w -= self.lr * g;
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.params {
            let tensor = param.tensor.borrow();

            if let Some(node) = &tensor.node {
                node.borrow_mut().grad = None;
            }
        }
    }
}

impl SGD {
    /// Create a new SGD optimizer
    /// # Arguments
    /// * `params` - A vector of parameters to optimize
    /// * `lr` - Learning rate
    /// # Returns
    /// A new SGD optimizer
    pub fn new(params: Vec<Parameter>, lr: f32) -> SGD {
        SGD { params, lr }
    }
}
