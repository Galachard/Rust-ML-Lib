use crate::Tensor;
use std::cell::RefCell;
use std::rc::Rc;

/// A trainable parameter in a neural network.
#[derive(Clone)]
pub struct Parameter {
    pub tensor: Rc<RefCell<Tensor>>,
}

impl Parameter {
    /// Creates a new trainable parameter from a tensor.
    pub fn new(tensor: Tensor) -> Self {
        Parameter {
            tensor: Rc::new(RefCell::new(tensor)),
        }
    }

    /// Creates a new trainable parameter from a vector. Initializes it as a leaf node.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Parameter {
            tensor: Rc::new(RefCell::new(Tensor::from_vec_leaf(data, shape))),
        }
    }
}
