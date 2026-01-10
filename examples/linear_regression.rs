use ml_lib::layer::Layer;
use ml_lib::layer::initializer::{HeNormal, Zeros};
use ml_lib::{Tensor, layer, optimizer};

fn main() {
    // Specify the model - linear regression, no activation function
    let mut model = layer::Sequential::new();
    model.add(layer::Linear::new(1, 1, &HeNormal {}, &Zeros {}));

    // SGD optimizer
    let mut opt = optimizer::SGD::new(model.parameters(), 0.1);

    // Training data: y = 2x + 1
    let inputs = vec![
        Tensor::from_vec_leaf(vec![0.0], vec![1, 1]),
        Tensor::from_vec_leaf(vec![1.0], vec![1, 1]),
        Tensor::from_vec_leaf(vec![2.0], vec![1, 1]),
        Tensor::from_vec_leaf(vec![3.0], vec![1, 1]),
        Tensor::from_vec_leaf(vec![4.0], vec![1, 1]),
    ];
    let targets = vec![
        Tensor::from_vec_leaf(vec![1.0], vec![1, 1]),
        Tensor::from_vec_leaf(vec![3.0], vec![1, 1]),
        Tensor::from_vec_leaf(vec![5.0], vec![1, 1]),
        Tensor::from_vec_leaf(vec![7.0], vec![1, 1]),
        Tensor::from_vec_leaf(vec![9.0], vec![1, 1]),
    ];

    // Train the model
    layer::Sequential::train(&model, &mut opt, &inputs, &targets, 10);

    // Test the model - for 5.0 the output should be close to 11.0
    let test_input = Tensor::from_vec_leaf(vec![5.0], vec![1, 1]);
    let prediction = model.forward(&test_input);
    println!("Prediction for input 5.0: {:?}", prediction.data);
}
