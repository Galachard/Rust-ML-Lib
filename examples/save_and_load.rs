use ml_lib::Tensor;
use ml_lib::layer::*;

fn main() {
    // Create a simple model
    let mut model = Sequential::new();
    model.add(Linear::new(
        2,
        4,
        &initializer::HeNormal {},
        &initializer::Zeros {},
    ));
    model.add(ELU::default());
    model.add(Linear::new(
        4,
        1,
        &initializer::HeNormal {},
        &initializer::Zeros {},
    ));
    model.add(Sigmoid);

    // Forward pass (dummy input)
    let x = Tensor::from_vec_leaf(vec![1.0, 2.0], vec![2, 1]);
    let y_pred = model.forward(&x);
    println!("Before saving: {:?}", y_pred.data);

    // Save model to disk
    model.save("model.ml_lib");

    // Create a new empty model and load
    let mut loaded_model = Sequential::new();
    loaded_model.add(Linear::dummy()); // dummy layers
    loaded_model.add(ELU::default());
    loaded_model.add(Linear::dummy());
    loaded_model.add(Sigmoid);

    loaded_model.load("model.ml_lib", &layer_factory);

    // Forward pass again
    let y_pred2 = loaded_model.forward(&x);
    println!("After loading: {:?}", y_pred2.data);
}

// Layer factory for deserialization
fn layer_factory(name: &str) -> Box<dyn Layer> {
    match name {
        "Linear" => Box::new(Linear::dummy()),
        "ELU" => Box::new(ELU::default()),
        "Sigmoid" => Box::new(Sigmoid),
        _ => panic!("Unknown layer type {}", name),
    }
}
