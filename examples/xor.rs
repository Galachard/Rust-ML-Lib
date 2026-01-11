use ml_lib::layer::Layer;
use ml_lib::layer::initializer::{HeNormal, Zeros};
use ml_lib::loss::mse;
use ml_lib::{Tensor, layer, optimizer};

fn main() {
    // Specify the model - linear regression, no activation function
    let mut model = layer::Sequential::new();
    model.add(layer::Linear::new(2, 4, &HeNormal {}, &Zeros {}));
    model.add(layer::ELU::default());
    model.add(layer::Linear::new(4, 1, &HeNormal {}, &Zeros {}));
    model.add(layer::Sigmoid);

    // SGD optimizer
    let mut opt = optimizer::SGD::new(model.parameters(), 0.8);

    // Training data: y = 2x + 1
    let inputs = vec![
        Tensor::from_vec_leaf(vec![0.0, 0.0], vec![2, 1]),
        Tensor::from_vec_leaf(vec![0.0, 1.0], vec![2, 1]),
        Tensor::from_vec_leaf(vec![1.0, 0.0], vec![2, 1]),
        Tensor::from_vec_leaf(vec![1.0, 1.0], vec![2, 1]),
    ];
    let targets = vec![
        Tensor::from_vec_leaf(vec![0.0], vec![1, 1]),
        Tensor::from_vec_leaf(vec![1.0], vec![1, 1]),
        Tensor::from_vec_leaf(vec![1.0], vec![1, 1]),
        Tensor::from_vec_leaf(vec![0.0], vec![1, 1]),
    ];

    // Train the model
    layer::Sequential::train(&model, &mut opt, &inputs, &targets, 256, 4, mse);

    // Test the model on training data (does it reproduce XOR)
    let test_input = Tensor::from_vec_leaf(vec![0.0, 0.0], vec![2, 1]);
    let prediction = model.forward(&test_input);
    println!("Prediction for input (0.0, 0.0): {:?}", prediction.data);

    let test_input = Tensor::from_vec_leaf(vec![0.0, 1.0], vec![2, 1]);
    let prediction = model.forward(&test_input);
    println!("Prediction for input (0.0, 1.0): {:?}", prediction.data);

    let test_input = Tensor::from_vec_leaf(vec![1.0, 0.0], vec![2, 1]);
    let prediction = model.forward(&test_input);
    println!("Prediction for input (1.0, 0.0): {:?}", prediction.data);

    let test_input = Tensor::from_vec_leaf(vec![1.0, 1.0], vec![2, 1]);
    let prediction = model.forward(&test_input);
    println!("Prediction for input (1.0, 1.0): {:?}", prediction.data);
}
