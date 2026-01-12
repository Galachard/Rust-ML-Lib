use crate::Tensor;
use crate::grad::GradFn;
use crate::graph::Node;

/// Cross entropy loss: expects probabilities (after Softmax)
/// # Arguments
/// * `pred` - Predicted probabilities tensor
/// * `target` - One-hot encoded target tensor
/// # Returns
/// * `Tensor` - Scalar tensor representing the cross entropy loss
pub fn cross_entropy(pred: &Tensor, target: &Tensor) -> Tensor {
    assert_eq!(pred.data.len(), target.data.len());

    let eps = 1e-9;

    let loss_value: f32 = pred
        .data
        .iter()
        .zip(&target.data)
        .map(|(p, t)| if *t > 0.0 { -t * (p + eps).ln() } else { 0.0 })
        .sum();

    struct CrossEntropyBackward {
        pred: Vec<f32>,
        target: Vec<f32>,
    }

    impl GradFn for CrossEntropyBackward {
        fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
            // dL/dp = -y / p
            let grad: Vec<f32> = self
                .pred
                .iter()
                .zip(&self.target)
                .map(|(p, t)| if *t > 0.0 { -t / (p + 1e-9) } else { 0.0 })
                .map(|g| g * grad_output.data[0])
                .collect();

            vec![Tensor::from_vec_leaf(grad, grad_output.shape.clone())]
        }
    }

    let node = Node::new(
        vec![pred.node.as_ref().unwrap().clone()],
        Box::new(CrossEntropyBackward {
            pred: pred.data.clone(),
            target: target.data.clone(),
        }),
    );

    Tensor {
        data: vec![loss_value],
        shape: vec![1, 1],
        node: Some(node),
    }
}
