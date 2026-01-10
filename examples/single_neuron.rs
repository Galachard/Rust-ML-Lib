use ml_lib::layer::Layer;
use ml_lib::layer::initializer::{HeNormal, Zeros};
use ml_lib::{Tensor, layer, optimizer};

fn main() {
    // Specify the model - linear regression, no activation function
    let mut model = layer::Sequential::new();
    model.add(layer::Linear::new(1, 1, &HeNormal {}, &Zeros {}));
    model.add(layer::Sigmoid);

    // SGD optimizer
    let mut opt = optimizer::SGD::new(model.parameters(), 0.1);

    // Training data: y = 2x + 1
    let inputs = vec![
        Tensor::from_vec_leaf(vec![0.0], vec![1, 1]),
        Tensor::from_vec_leaf(vec![1.0], vec![1, 1]),
        Tensor::from_vec_leaf(vec![2.0], vec![1, 1]),
        Tensor::from_vec_leaf(vec![3.0], vec![1, 1]),
    ];
    let targets = vec![
        Tensor::from_vec_leaf(vec![0.0], vec![1, 1]),
        Tensor::from_vec_leaf(vec![0.0], vec![1, 1]),
        Tensor::from_vec_leaf(vec![1.0], vec![1, 1]),
        Tensor::from_vec_leaf(vec![1.0], vec![1, 1]),
    ];

    // Train the model
    layer::Sequential::train(&model, &mut opt, &inputs, &targets, 256);

    // Test the model
    // For -1.0 the output should be close to 0, for 2.5 between 0 and 1 and for 5.0 close to 1
    let test_input = Tensor::from_vec_leaf(vec![-1.0], vec![1, 1]);
    let prediction = model.forward(&test_input);
    println!("Prediction for input -1.0: {:?}", prediction.data);

    let test_input = Tensor::from_vec_leaf(vec![2.5], vec![1, 1]);
    let prediction = model.forward(&test_input);
    println!("Prediction for input 2.5: {:?}", prediction.data);

    let test_input = Tensor::from_vec_leaf(vec![5.0], vec![1, 1]);
    let prediction = model.forward(&test_input);
    println!("Prediction for input 5.0: {:?}", prediction.data);
}
