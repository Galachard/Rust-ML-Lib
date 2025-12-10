use crate::error::TensorError;
use crate::grad::GradFn;
use crate::graph::Node;
use crate::ops::*;
use crate::tensor::*;
use std::cell::RefCell;
use std::rc::{Rc, Weak};

/// Element-wise addition of two tensors with gradient tracking.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let out_data = raw_add(a, b)?;

    // Graph for grad
    struct AddBackward;
    impl GradFn for AddBackward {
        fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
            // d(a + b)/da = 1
            // d(a + b)/db = 1
            vec![grad_output.clone(), grad_output.clone()]
        }
    }

    let node = Node::new(vec![weak(&a.node), weak(&b.node)], Box::new(AddBackward));

    Ok(Tensor {
        data: out_data.data,
        shape: a.shape.clone(),
        node: Some(node),
    })
}

/// Element-wise multiplication of two tensors with gradient tracking.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let out_data = raw_mul(a, b)?;

    // Graph for grad
    struct MulBackward {
        a: Tensor,
        b: Tensor,
    }
    impl GradFn for MulBackward {
        fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
            // d(ab)/da = db
            // d(ab)/db = da
            let grad_a = raw_mul(grad_output, &self.b).unwrap();

            let grad_b = raw_mul(grad_output, &self.a).unwrap();

            vec![grad_a, grad_b]
        }
    }

    let node = Node::new(
        vec![weak(&a.node), weak(&b.node)],
        Box::new(MulBackward {
            a: a.clone(),
            b: b.clone(),
        }),
    );

    Ok(Tensor {
        data: out_data.data,
        shape: a.shape.clone(),
        node: Some(node),
    })
}

/// Matrix multiplication of two tensors with gradient tracking.
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let out_data = raw_matmul(a, b)?;

    struct MatMulBackward {
        a: Tensor,
        b: Tensor,
    }

    impl GradFn for MatMulBackward {
        fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
            // grad_output has shape (m, n)

            // dA = grad_output * B^T
            let b_transposed = transpose(&self.b).unwrap();
            let da = matmul(grad_output, &b_transposed).unwrap();

            // dB = A^T * grad_output
            let a_transposed = transpose(&self.a).unwrap();
            let db = matmul(&a_transposed, grad_output).unwrap();

            vec![da, db]
        }
    }

    let node = Node::new(
        vec![weak(&a.node), weak(&b.node)],
        Box::new(MatMulBackward {
            a: a.clone(),
            b: b.clone(),
        }),
    );

    Ok(Tensor {
        data: out_data.data,
        shape: out_data.shape,
        node: Some(node),
    })
}

/// Element-wise exponential of a tensor with gradient tracking.
pub fn exp(t: &Tensor) -> Result<Tensor, TensorError> {
    let out_data = raw_exp(t)?;

    // Graph for grad
    struct ExpBackward {
        a: Tensor,
    }
    impl GradFn for ExpBackward {
        fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
            // d(e^a)/da = e^a
            let grad = raw_mul(grad_output, &raw_exp(&self.a).unwrap()).unwrap();
            vec![grad]
        }
    }

    let node = Node::new(vec![weak(&t.node)], Box::new(ExpBackward { a: t.clone() }));

    Ok(Tensor {
        data: out_data.data,
        shape: t.shape.clone(),
        node: Some(node),
    })
}

/// Helper function to convert Option<Rc<RefCell<Node>>> to Weak<RefCell<Node>>.
fn weak(n: &Option<Rc<RefCell<Node>>>) -> Weak<RefCell<Node>> {
    n.as_ref().map(Rc::downgrade).unwrap_or_default()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::grad::grad_fn::LeafGrad;

    #[test]
    fn test_add_backward() {
        let a = Tensor::from_vec_with_node(
            vec![1.0, 2.0],
            vec![2],
            Some(Node::new(vec![], Box::new(LeafGrad {}))),
        );
        let b = Tensor::from_vec_with_node(
            vec![3.0, 4.0],
            vec![2],
            Some(Node::new(vec![], Box::new(LeafGrad {}))),
        );

        let c = add(&a, &b).unwrap();

        c.backward();

        let a_grad = a.node.unwrap().borrow().grad.as_ref().unwrap().clone();
        let b_grad = b.node.unwrap().borrow().grad.as_ref().unwrap().clone();

        // In addition, the derivatives are 1.0
        assert_eq!(a_grad.data, vec![1.0, 1.0]);
        assert_eq!(b_grad.data, vec![1.0, 1.0]);
    }

    #[test]
    fn test_mul_backward() {
        let a = Tensor::from_vec_with_node(
            vec![1.0, 2.0, 3.0],
            vec![1, 3],
            Some(Node::new(vec![], Box::new(LeafGrad {}))),
        );
        let b = Tensor::from_vec_with_node(
            vec![4.0, 5.0, 6.0],
            vec![1, 3],
            Some(Node::new(vec![], Box::new(LeafGrad {}))),
        );

        let c = mul(&a, &b).unwrap();

        c.backward();

        let a_grad = a.node.unwrap().borrow().grad.as_ref().unwrap().clone();
        let b_grad = b.node.unwrap().borrow().grad.as_ref().unwrap().clone();

        // Gradients:
        // d(ab)/da = (4.0, 5.0, 6.0)
        // d(ab)/db = (1.0, 2.0, 3.0)
        assert_eq!(a_grad.data, vec![4.0, 5.0, 6.0]);
        assert_eq!(b_grad.data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_exp_backward() {
        let a = Tensor::from_vec_with_node(
            vec![0.0, 1.0, 2.0],
            vec![3],
            Some(Node::new(vec![], Box::new(LeafGrad {}))),
        );

        let b = exp(&a).unwrap();

        b.backward();

        let a_grad = a.node.unwrap().borrow().grad.as_ref().unwrap().clone();

        // The derivatives are e^a, so 0.0, e, e^2
        let one: f32 = 1.0;
        let two: f32 = 2.0;
        assert_eq!(a_grad.data(), &[1.0, one.exp(), two.exp()]);
    }

    #[test]
    fn test_matmul_backward() {
        let a = Tensor::from_vec_with_node(
            vec![1.0, 2.0],
            vec![1, 2],
            Some(Node::new(vec![], Box::new(LeafGrad {}))),
        );
        let b = Tensor::from_vec_with_node(
            vec![3.0, 4.0],
            vec![2, 1],
            Some(Node::new(vec![], Box::new(LeafGrad {}))),
        );

        let c = matmul(&a, &b).unwrap();

        c.backward();

        let a_grad = a.node.unwrap().borrow().grad.as_ref().unwrap().clone();
        let b_grad = b.node.unwrap().borrow().grad.as_ref().unwrap().clone();

        assert_eq!(a_grad.data(), &[3.0, 4.0]);
        assert_eq!(b_grad.data(), &[1.0, 2.0]);
    }
}
