use ml_lib::grad::LeafGrad;
use ml_lib::graph::node::Node;
use ml_lib::ops::grad_ops;
use ml_lib::tensor::tensor_struct::*;

#[test]
fn test_double_connection() {
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

    let ab = grad_ops::mul(&a, &b).unwrap();
    let res = grad_ops::add(&ab, &a).unwrap();

    res.backward();

    let a_grad = a.node.unwrap().borrow().grad.as_ref().unwrap().clone();
    let b_grad = b.node.unwrap().borrow().grad.as_ref().unwrap().clone();

    assert_eq!(a_grad.data(), &[4.0, 5.0]);
    assert_eq!(b_grad.data(), &[1.0, 2.0]);
}

#[test]
fn test_chain() {
    let a = Tensor::from_vec_with_node(
        vec![0.0, 1.0],
        vec![2, 1],
        Some(Node::new(vec![], Box::new(LeafGrad {}))),
    );
    let b = Tensor::from_vec_with_node(
        vec![2.0, 3.0],
        vec![2, 1],
        Some(Node::new(vec![], Box::new(LeafGrad {}))),
    );
    let c = Tensor::from_vec_with_node(
        vec![4.0, 5.0, 6.0, 7.0],
        vec![2, 2],
        Some(Node::new(vec![], Box::new(LeafGrad {}))),
    );
    let d = Tensor::from_vec_with_node(
        vec![8.0, 9.0, 10.0, 11.0],
        vec![2, 2],
        Some(Node::new(vec![], Box::new(LeafGrad {}))),
    );
    let e = Tensor::from_vec_with_node(
        vec![12.0, 13.0],
        vec![2, 1],
        Some(Node::new(vec![], Box::new(LeafGrad {}))),
    );

    let res1 = grad_ops::exp(&a).unwrap();
    let res2 = grad_ops::mul(&res1, &b).unwrap();
    let res3 = grad_ops::matmul(&c, &d).unwrap();
    let res4 = grad_ops::matmul(&res3, &e).unwrap();
    let res5 = grad_ops::add(&res4, &res2).unwrap();
    let res6 = grad_ops::mul(&res5, &res1).unwrap();
    let res7 = grad_ops::mul(&res6, &a).unwrap();

    let one: f32 = 1.0;
    let exp = one.exp();
    assert_eq!(res1.data(), &[1.0, exp]);
    assert_eq!(res2.data(), &[2.0, 3.0 * exp]);
    assert_eq!(res3.data(), &[82.0, 91.0, 118.0, 131.0]);
    assert_eq!(res4.data(), &[2167.0, 3119.0]);
    assert_eq!(res5.data(), &[2169.0, 3119.0 + 3.0 * exp]);
    assert_eq!(res6.data(), &[2169.0, (3119.0 + 3.0 * exp) * exp]);
    assert_eq!(res7.data(), &[0.0, (3119.0 + 3.0 * exp) * exp]);

    res7.backward();

    let a_grad = a.node.unwrap().borrow().grad.as_ref().unwrap().clone();
    let b_grad = b.node.unwrap().borrow().grad.as_ref().unwrap().clone();
    let c_grad = c.node.unwrap().borrow().grad.as_ref().unwrap().clone();
    let d_grad = d.node.unwrap().borrow().grad.as_ref().unwrap().clone();
    let e_grad = e.node.unwrap().borrow().grad.as_ref().unwrap().clone();

    assert_eq!(a_grad.data(), &[2169.0, 17023.14]);
    assert_eq!(b_grad.data(), &[0.0, 7.3890557]);
    assert_eq!(c_grad.data(), &[0.0, 0.0, 578.994, 714.9081]);
    assert_eq!(d_grad.data(), &[195.71628, 212.02597, 228.33566, 247.36363]);
    assert_eq!(e_grad.data(), &[320.75723, 356.0949]);
}
