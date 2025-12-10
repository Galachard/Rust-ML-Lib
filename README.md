# ML Lib

A minimal deep learning library in Rust, featuring tensor operations, automatic differentiation, and computation graphs.

---

## Project Structure

- `src/`
    - `lib.rs` — Library root, exposing public API
    - `grad/`
        - `grad_fn.rs` — `GradFn` trait defining backward operations and `LeafGrad` struct for leaf nodes
    - `grad.rs` — Grad-related re-exports
    - `graph/`
        - `node.rs` — `Node` struct representing computation graph nodes
    - `graph.rs` — Graph utilities and re-exports
    - `ops/`
        - `grad_ops.rs` — Gradient-aware operations
        - `raw_ops.rs` — Raw tensor operations without gradient tracking
    - `ops.rs` — Ops re-exports
    - `tensor/`
        - `tensor_struct.rs` — `Tensor` struct
        - `backward.rs` — Backpropagation logic for `Tensor`
    - `tensor.rs` — Tensor re-exports
    - `error.rs` — Errors

- `tests/`
    - `grad_tests.rs` — integration tests for gradient computation and ops

---

## Features

- **Tensors**: N-dimensional arrays.
- **Raw operations**: Basic operations (addition, multiplication, etc.) on tensors.
- **Grad-aware operations**: Operations that support automatic differentiation.
- **Computation graph**: Nodes to track dependencies for reverse-mode autodiff.
- **Backward propagation**: Compute gradients for all tensors in the graph.
- **Unit and integration tests**: Validate correctness of operations and gradient computations.

---

## Planned Features

- Neural network layers and activations
- Optimizers
- Neural network training loops
- Parallelized operations using Rust concurrency
