use crate::Tensor;
use crate::graph::Node;
use crate::grad::GradFn;

pub fn softmax_cross_entropy(logits: &Tensor, target: &Tensor) -> Tensor {
    let max = logits.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let exp: Vec<f32> = logits.data.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();

    let probs: Vec<f32> = exp.iter().map(|x| x / sum).collect();

    let loss: f32 = probs
        .iter()
        .zip(&target.data)
        .map(|(p, t)| if *t > 0.0 { -p.ln() } else { 0.0 })
        .sum();

    struct SoftmaxCEBackward {
        probs: Vec<f32>,
        target: Vec<f32>,
    }

    impl GradFn for SoftmaxCEBackward {
        fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
            let grad: Vec<f32> = self
                .probs
                .iter()
                .zip(&self.target)
                .map(|(p, t)| (p - t) * grad_output.data[0])
                .collect();

            vec![Tensor::from_vec_leaf(grad, vec![self.target.len(), 1])]
        }
    }

    let node = Node::new(
        vec![logits.node.as_ref().unwrap().clone()],
        Box::new(SoftmaxCEBackward {
            probs,
            target: target.data.clone(),
        }),
    );

    Tensor {
        data: vec![loss],
        shape: vec![1, 1],
        node: Some(node),
    }
}
