use ml_lib::data::mnist::load_mnist_as_tensors;
use ml_lib::layer::Layer;
use ml_lib::layer::initializer::{HeNormal, Zeros};
use ml_lib::{Tensor, layer, loss, optimizer};
use std::path::PathBuf;

fn argmax(tensor: &Tensor) -> usize {
    let mut max_index = 0;
    let mut max_value = tensor.data[0];
    for (i, &value) in tensor.data.iter().enumerate() {
        if value > max_value {
            max_value = value;
            max_index = i;
        }
    }
    max_index
}

fn main() {
    // Load the MNIST train dataset
    let mut base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    base_path.push("data/mnist");
    let images = base_path.join("train-images.idx3-ubyte");
    let labels = base_path.join("train-labels.idx1-ubyte");

    let (inputs, targets) =
        load_mnist_as_tensors(images, labels, 10).expect("Failed to load MNIST dataset.");

    let mut model = layer::Sequential::new();
    model.add(layer::Linear::new(784, 64, &HeNormal {}, &Zeros {}));
    model.add(layer::ELU::default());
    model.add(layer::Linear::new(64, 64, &HeNormal {}, &Zeros {}));
    model.add(layer::ELU::default());
    model.add(layer::Linear::new(64, 10, &HeNormal {}, &Zeros {}));

    let mut opt = optimizer::SGD::new(model.parameters(), 1e-3);

    layer::Sequential::train(
        &model,
        &mut opt,
        &inputs,
        &targets,
        4,
        64,
        loss::softmax_cross_entropy,
    );

    // Load MNIST test dataset
    let test_images = base_path.join("t10k-images.idx3-ubyte");
    let test_labels = base_path.join("t10k-labels.idx1-ubyte");

    let (test_inputs, test_targets) = load_mnist_as_tensors(test_images, test_labels, 10)
        .expect("Failed to load MNIST test dataset.");

    let mut correct = 0;
    for i in 0..test_inputs.len() {
        let output = model.forward(&test_inputs[i]);
        let predicted_label = argmax(&output);
        let true_label = argmax(&test_targets[i]);
        if predicted_label == true_label {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / test_inputs.len() as f32;
    println!("Test Accuracy: {:.2}%", accuracy * 100.0);
}
