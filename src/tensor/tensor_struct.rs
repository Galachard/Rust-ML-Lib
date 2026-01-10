use crate::error::TensorError;
use crate::grad::GradFn;
use crate::grad::LeafGrad;
use crate::graph::Node;
use std::cell::RefCell;
use std::rc::Rc;

/// Tensor struct representing a multidimensional array with optional computation graph node.
/// Shape is represented as a vector of usize, and data is stored in a flat vector of f32.
/// Can optionally link to a computation graph node for automatic differentiation.
/// # Examples
/// ```
/// use ml_lib::tensor::Tensor;
/// let t = Tensor::zeros(vec![2, 3]);
/// assert_eq!(t.shape(), &[2, 3]);
/// assert!(t.data().iter().all(|x| *x == 0.0));
/// ```
/// ```
/// use ml_lib::tensor::Tensor;
/// use ml_lib::grad::LeafGrad;
/// use ml_lib::graph::node::Node;
/// let t = Tensor::from_vec_with_node(vec![1.0, 2.0, 3.0], vec![3], Some(Node::new(vec![], Box::new(LeafGrad {}))));
/// assert_eq!(t.data(), &[1.0, 2.0, 3.0]);
/// ```
#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,

    // Computation graph node (None for leaf tensors)
    pub node: Option<Rc<RefCell<Node>>>,
}

impl Tensor {
    /// Create a new Tensor with given data and shape.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            node: None,
        }
    }

    /// Get a reference to the tensor's data.
    pub fn data(&self) -> &Vec<f32> {
        &self.data
    }

    /// Get a reference to the tensor's shape.
    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    /// Check if the given position is valid for the tensor's shape.
    fn check_position(&self, position: &[usize]) -> Result<(), TensorError> {
        if position.len() != self.shape.len() {
            return Err(TensorError::InvalidIndex {
                index: position.to_owned(),
                shape: self.shape.clone(),
            });
        }
        for i in 0..position.len() {
            if position[i] >= self.shape[i] {
                return Err(TensorError::InvalidIndex {
                    index: position.to_owned(),
                    shape: self.shape.clone(),
                });
            }
        }
        Ok(())
    }

    /// Get the value at the specified position.
    pub fn get(self, position: Vec<usize>) -> Result<f32, TensorError> {
        self.check_position(&position)?;
        let mut index = 0;
        for i in position {
            index += i;
        }
        Ok(self.data[index])
    }

    /// Set the value at the specified position.
    pub fn set(&mut self, position: Vec<usize>, value: f32) -> Result<f32, TensorError> {
        self.check_position(&position)?;
        let mut index = 0;
        for i in position {
            index += i;
        }
        let old_value = self.data[index];
        self.data[index] = value;
        Ok(old_value)
    }

    /// Create a tensor filled with zeros of the specified shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self::new(vec![0.0; size], shape)
    }

    /// Create a tensor filled with ones of the specified shape.
    pub fn ones(shape: Vec<usize>) -> Self {
        let size = shape.iter().product();
        Self::new(vec![1.0; size], shape)
    }

    /// Create a tensor of zeros with the same shape as the given tensor.
    pub fn zeros_like(t: &Tensor) -> Self {
        let shape = t.shape.clone();
        let size = shape.iter().product();
        Self::new(vec![0.0; size], shape)
    }

    /// Create a tensor of ones with the same shape as the given tensor.
    pub fn ones_like(t: &Tensor) -> Self {
        let shape = t.shape.clone();
        let size = shape.iter().product();
        Self::new(vec![1.0; size], shape)
    }

    /// Create a tensor from a vector of data and a shape.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self::new(data, shape)
    }

    /// Create a tensor from a vector of data, a shape, and an optional computation graph node.
    pub fn from_vec_with_node(
        data: Vec<f32>,
        shape: Vec<usize>,
        node: Option<Rc<RefCell<Node>>>,
    ) -> Self {
        Tensor { data, shape, node }
    }

    /// Creates a leaf tensor from a vector of data and a shape.
    pub fn from_vec_leaf(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Tensor {
            data,
            shape,
            node: Some(Node::new(vec![], Box::new(LeafGrad {}))),
        }
    }

    /// Apply a function element-wise to the tensor, with its derivative for backpropagation.
    pub fn map<F, G>(&self, f: F, df: G) -> Tensor
    where
        F: Fn(f32) -> f32 + 'static,
        G: Fn(f32) -> f32 + 'static,
    {
        // Forward pass
        let out_data: Vec<f32> = self.data.iter().copied().map(&f).collect();

        // Backward node
        struct MapBackward<G>
        where
            G: Fn(f32) -> f32,
        {
            input: Tensor,
            df: G,
        }

        impl<G> GradFn for MapBackward<G>
        where
            G: Fn(f32) -> f32 + 'static,
        {
            fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
                let grad_data: Vec<f32> = self
                    .input
                    .data
                    .iter()
                    .zip(&grad_output.data)
                    .map(|(&x, &g)| g * (self.df)(x))
                    .collect();

                vec![Tensor {
                    data: grad_data,
                    shape: self.input.shape.clone(),
                    node: None,
                }]
            }
        }

        let node = Node::new(
            vec![self.node.as_ref().expect("Tensor must have node").clone()],
            Box::new(MapBackward {
                input: self.clone(),
                df,
            }),
        );

        Tensor {
            data: out_data,
            shape: self.shape.clone(),
            node: Some(node),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.data().iter().all(|x| *x == 0.0));
    }

    #[test]
    fn test_tensor_ones() {
        let t = Tensor::ones(vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.data().iter().all(|x| *x == 1.0));
    }

    #[test]
    fn test_tensor_from_vec() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(t.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_tensor_indexing() {
        let mut t = Tensor::zeros(vec![2, 2]);
        _ = t.set(vec![1, 1], 5.0);
        assert_eq!(t.get(vec![1, 1]).unwrap(), 5.0);
    }
}
