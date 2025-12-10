use crate::grad::GradFn;
use crate::tensor::*;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// A node in the computation graph.
pub struct Node {
    pub parents: Vec<Weak<RefCell<Node>>>,
    pub grad_fn: Box<dyn GradFn>,

    // gradient accumulated during backward pass
    pub grad: Option<Tensor>,
}

impl Node {
    /// Create a new computation graph node. New nodes are wrapped in Rc<RefCell<>> and have None
    /// in their grad field.
    pub fn new(parents: Vec<Weak<RefCell<Node>>>, grad_fn: Box<dyn GradFn>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Self {
            parents,
            grad_fn,
            grad: None,
        }))
    }
}
