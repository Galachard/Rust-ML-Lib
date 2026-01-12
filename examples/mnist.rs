use ml_lib::data::mnist::load_mnist_as_tensors;
use ml_lib::layer::Layer;
use ml_lib::layer::initializer::{HeNormal, Zeros};
use ml_lib::{layer, loss, optimizer};
use std::path::PathBuf;

fn main() {
    // Load the MNIST train dataset
    let mut base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    base_path.push("data/mnist");
    let images = base_path.join("train-images.idx3-ubyte");
    let labels = base_path.join("train-labels.idx1-ubyte");

    println!("{}", images.as_path().display());

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
        2,
        64,
        loss::softmax_cross_entropy,
    );
}
