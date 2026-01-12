/// One-hot encodes a label into a vector of given number of classes.
pub fn one_hot(label: u8, num_classes: usize) -> Vec<f32> {
    let mut v = vec![0.0; num_classes];
    v[label as usize] = 1.0;
    v
}
