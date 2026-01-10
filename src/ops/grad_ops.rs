use crate::error::TensorError;
use crate::grad::GradFn;
use crate::graph::Node;
use crate::ops::*;
use crate::tensor::*;
use std::cell::RefCell;
use std::rc::Rc;

/// Element-wise addition of two tensors with gradient tracking.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let out_data = raw_add(a, b)?;

    struct AddBackward;
    impl GradFn for AddBackward {
        fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
            // d(a + b)/da = 1
            // d(a + b)/db = 1
            vec![grad_output.clone(), grad_output.clone()]
        }
    }

    let node = Node::new(vec![node_rc_of(a), node_rc_of(b)], Box::new(AddBackward));

    Ok(Tensor {
        data: out_data.data,
        shape: a.shape.clone(),
        node: Some(node),
    })
}

/// Element-wise subtraction of two tensors with gradient tracking.
pub fn sub(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    let out_data = raw_sub(a, b)?;

    struct SubBackward;
    impl GradFn for SubBackward {
        fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
            // d(a - b)/da = 1
            // d(a - b)/db = -1
            vec![
                grad_output.clone(),
                raw_scalar_mul(grad_output, -1.0).unwrap(),
            ]
        }
    }

    let node = Node::new(vec![node_rc_of(a), node_rc_of(b)], Box::new(SubBackward));

    Ok(Tensor {
        data: out_data.data,
        shape: a.shape.clone(),
        node: Some(node),
    })
}

/// Element-wise addition of a tensor and a scalar with gradient tracking.
pub fn scalar_add(t: &Tensor, scalar: f32) -> Result<Tensor, TensorError> {
    let out_data = raw_scalar_add(t, scalar)?;

    struct ScalarAddBackward;
    impl GradFn for ScalarAddBackward {
        fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
            // d(a + scalar)/da = 1
            vec![grad_output.clone()]
        }
    }
    let node = Node::new(vec![node_rc_of(t)], Box::new(ScalarAddBackward));
    Ok(Tensor {
        data: out_data.data,
        shape: t.shape.clone(),
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
        vec![node_rc_of(a), node_rc_of(b)],
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

/// Element-wise multiplication of a tensor by a scalar with gradient tracking.
pub fn scalar_mul(t: &Tensor, scalar: f32) -> Result<Tensor, TensorError> {
    let out_data = raw_scalar_mul(t, scalar)?;

    // Graph for grad
    struct ScalarMulBackward {
        scalar: f32,
    }
    impl GradFn for ScalarMulBackward {
        fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
            // d(scalar * a)/da = scalar
            let grad = raw_scalar_mul(grad_output, self.scalar).unwrap();
            vec![grad]
        }
    }

    let node = Node::new(vec![node_rc_of(t)], Box::new(ScalarMulBackward { scalar }));

    Ok(Tensor {
        data: out_data.data,
        shape: t.shape.clone(),
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
            let (m, k) = (self.a.shape[0], self.a.shape[1]); // a: m x k
            let (kb, n) = (self.b.shape[0], self.b.shape[1]); // b: k x n
            assert_eq!(k, kb);

            // convenience helpers to index flat arrays
            fn idx(shape: &[usize], r: usize, c: usize) -> usize {
                r * shape[1] + c
            }

            assert_eq!(grad_output.shape, vec![m, n]);

            // dA = grad_output * B^T
            let mut d_a = vec![0.0f32; m * k];
            for i in 0..m {
                for j in 0..k {
                    let mut sum = 0.0f32;
                    for p in 0..n {
                        let dcp = grad_output.data[idx(&grad_output.shape, i, p)];
                        let bpj = self.b.data[idx(&self.b.shape, j, p)];
                        sum += dcp * bpj;
                    }
                    d_a[idx(&[m, k], i, j)] = sum;
                }
            }

            // dB = A^T * grad_output
            let mut d_b = vec![0.0f32; k * n];
            for i in 0..k {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for p in 0..m {
                        let api = self.a.data[idx(&self.a.shape, p, i)];
                        let dpj = grad_output.data[idx(&grad_output.shape, p, j)];
                        sum += api * dpj;
                    }
                    d_b[idx(&[k, n], i, j)] = sum;
                }
            }

            // Return gradient tensors as *leaf* tensors (no node)
            let grad_a = Tensor::from_vec_leaf(d_a, vec![m, k]);
            let grad_b = Tensor::from_vec_leaf(d_b, vec![k, n]);

            vec![grad_a, grad_b]
        }
    }

    let node = Node::new(
        vec![node_rc_of(a), node_rc_of(b)],
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

    let node = Node::new(vec![node_rc_of(t)], Box::new(ExpBackward { a: t.clone() }));

    Ok(Tensor {
        data: out_data.data,
        shape: t.shape.clone(),
        node: Some(node),
    })
}

/// Element-wise inverse of a tensor with gradient tracking.
pub fn inv(t: &Tensor) -> Result<Tensor, TensorError> {
    let out_data = raw_inv(t)?;

    // Graph for grad
    struct InvBackward {
        a: Tensor,
    }
    impl GradFn for InvBackward {
        fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
            // d(1/a)/da = -1/a^2
            let a_squared = raw_mul(&self.a, &self.a).unwrap();
            let a_squared_inv = raw_inv(&a_squared).unwrap();
            let multiplier = raw_scalar_mul(&a_squared_inv, -1.0).unwrap();
            let grad = raw_mul(grad_output, &multiplier).unwrap();
            vec![grad]
        }
    }

    let node = Node::new(vec![node_rc_of(t)], Box::new(InvBackward { a: t.clone() }));

    Ok(Tensor {
        data: out_data.data,
        shape: t.shape.clone(),
        node: Some(node),
    })
}

/// Sum all elements of a tensor with gradient tracking.
pub fn sum(t: &Tensor) -> Tensor {
    let sum_value: f32 = t.data.iter().sum();

    struct SumBackward {
        input_shape: Vec<usize>,
    }

    impl GradFn for SumBackward {
        fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
            // grad_output is scalar (shape [1])
            let g = grad_output.data[0];

            let size: usize = self.input_shape.iter().product();
            let data = vec![g; size];

            vec![Tensor::from_vec_leaf(data, self.input_shape.clone())]
        }
    }

    let node = Node::new(
        vec![node_rc_of(t)],
        Box::new(SumBackward {
            input_shape: t.shape.clone(),
        }),
    );

    Tensor {
        data: vec![sum_value],
        shape: vec![1],
        node: Some(node),
    }
}

/// Helper function to convert Option<Rc<RefCell<Node>>> to Weak<RefCell<Node>>.
// fn weak(n: &Option<Rc<RefCell<Node>>>) -> Weak<RefCell<Node>> {
//     n.as_ref().map(Rc::downgrade).unwrap_or_default()
// }
fn node_rc_of(t: &Tensor) -> Rc<RefCell<Node>> {
    t.node.as_ref().expect("Tensor must have node").clone()
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::grad::LeafGrad;

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
