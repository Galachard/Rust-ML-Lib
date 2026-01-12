# Deep Learning Framework in Rust

This repository contains deep learning framework written from scratch in Rust, done as a final project for Rust course.

It implements a dynamic autograd system, core tensor operations, fully connected neural networks, losses, optimizers, and optional CPU parallelization for raw tensor operations.

---

## Features

- Tensor API
    - N-dimensional tensors
    - Shape checking
    - Explicit data layout
- Dynamic autograd
    - Runtime computation graph
    - Backward propagation via graph traversal
    - Custom gradient functions
- Neural network layers
    - Linear
    - ELU
    - Sigmoid
    - Sequential container
- Loss functions
    - Mean Squared Error (MSE)
    - Cross Entropy
    - Softmax Cross Entropy
- Optimizers
    - Stochastic Gradient Descent (SGD)
- Model serialization
    - Layer-aware save and load
    - Full serialization not implemented - Sequential cannot be serialized if a part of another Sequential model
- Optional CPU parallelization
    - Raw tensor operations parallelization using rayon
    - Enabled via feature flags

---

## Repository Structure

```
src/
├── data
│ ├── mnist.rs
│ └── utils.rs
├── grad
│ ├── grad_fn.rs
│ └── leaf_grad.rs
├── graph
│ └── node.rs
├── layer
│ ├── elu.rs
│ ├── initializer
│ ├── layer_trait.rs
│ ├── linear.rs
│ ├── sequential.rs
│ ├── serializable_layer.rs
│ └── sigmoid.rs
├── loss
│ ├── cross_entropy.rs
│ ├── mse.rs
│ └── softmax_cross_entropy.rs
├── metric
│ └── accuracy.rs
├── ops
│ ├── grad_ops.rs
│ └── raw_ops.rs
├── optimizer
│ ├── optimizer_trait.rs
│ └── sgd.rs
├── tensor
│ ├── backward.rs
│ ├── operators.rs
│ ├── parameter.rs
│ └── tensor_struct.rs
├── error.rs
├── lib.rs

examples/
├── mnist.rs
├── xor.rs
├── linear_regression.rs
├── single_neuron.rs
├── save_and_load.rs
├── parallel_raw_op_speedup.rs
tests/
└── grad_tests.rs
```

---

## Dependencies
This library uses the following dependencies:
- *rand* & *rand_distr* - for random number generation and distributions
- *indicatif* - for training progress bars
- *serde* & *bitcode* - for serialization and deserialization
- *rayon* - for optional parallelization of raw tensor operations

## Getting Started

### Build

```bash
cargo build --release
```

### Run MNIST Example
First, download the MNIST dataset from [Yann LeCun's website](https://yann.lecun.org/exdb/mnist/) or from [a mirror](https://github.com/cvdfoundation/mnist) and place the files in the `data/mnist/` directory:
- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

Then, run the MNIST example:
```bash
cargo run --release --example mnist
```
The network proposed in the file reaches over 95% accuracy after just 4 epochs of training.

### Run Other Examples
```bash
cargo run --release --example xor
cargo run --release --example linear_regression
cargo run --release --example single_neuron
cargo run --release --example save_and_load
cargo run --release --example parallel_raw_op_speedup --features parallel_ops
cargo run --release --example parallel_raw_op_speedup
```

### Example: Training a Model
```
let mut model = Sequential::new();
model.add(Linear::new(784, 64, &HeNormal {}, &Zeros {}));
model.add(ELU::default());
model.add(Linear::new(64, 10, &HeNormal {}, &Zeros {}));


let mut opt = SGD::new(model.parameters(), 1e-4);

Sequential::train(
    &model,
    &mut opt,
    &inputs,
    &targets,
    5,
    64,
    softmax_cross_entropy,
);
```

### Saving and Loading Models

Models are serialized without autograd state.

```
// Save
model.save("model.ml_lib");

// Load
let mut loaded_model = Sequential::new();
loaded_model.add(Linear::dummy()); // dummy layers
loaded_model.add(ELU::default());
loaded_model.add(Linear::dummy());
loaded_model.add(Sigmoid);
loaded_model.load("model.ml_lib", &layer_factory);
```

Each layer defines:
- How its configuration is serialized
- How its parameters are serialized
- How it reconstructs itself from serialized data

## Parallelization
### Raw Tensor Operations

Raw tensor operations have parallelized implementations using rayon.

Enable them with:

```bash
cargo run --release --features parallel_ops
```

### Performance Notes

Parallel raw operations do not always speed up training workloads.

This is expected for models like MNIST where execution is dominated by smaller tensor operations and autograd overhead.

Parallelism is most effective for large standalone tensor operations.

### Tests

Run all tests with:

```bash
cargo test
```

Run tests with parallel raw ops:

```bash
cargo test --features parallel_ops
```
