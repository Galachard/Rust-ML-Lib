use crate::Tensor;

pub fn accuracy(pred: &Tensor, target: &Tensor) -> f32 {
    assert_eq!(pred.data.len(), target.data.len());

    let pred_class = pred
        .data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    let target_class = target
        .data
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    if pred_class == target_class {
        1.0
    } else {
        0.0
    }
}
