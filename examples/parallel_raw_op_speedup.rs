use ml_lib::Tensor;
use ml_lib::layer::initializer::{Initializer, Normal};
use ml_lib::ops::*;

macro_rules! time_op {
    ($desc:expr, $op:expr) => {{
        use std::time::Instant;
        let start = Instant::now();
        let result = $op;
        let duration = start.elapsed();
        println!("{} took: {:?}", $desc, duration);
        result
    }};
}

fn main() {
    let a_shape = vec![1024, 1024];
    let b_shape = vec![1024, 1024];

    println!(
        "Preparing benchmark for {}x{} * {}x{} matrix ops",
        a_shape[0], a_shape[1], b_shape[0], b_shape[1]
    );

    #[cfg(not(feature = "parallel_ops"))]
    {
        println!("Version - sequential operations\n");
    }
    #[cfg(feature = "parallel_ops")]
    {
        println!("Version - parallel operations\n");
    }

    let a_initializer = Normal {
        mean: 1.0,
        std: 1000.0,
    };
    let b_initializer = Normal {
        mean: -1.0,
        std: 100.0,
    };

    let a_data = a_initializer.init(&a_shape);
    let b_data = b_initializer.init(&b_shape);

    let a = Tensor::from_vec(a_data, a_shape);
    let b = Tensor::from_vec(b_data, b_shape);

    time_op!("Matrix multiplication", raw_matmul(&a, &b)).unwrap();
    time_op!("Element-wise matrix multiplication", &a * &b);
    time_op!("Addition", &a + &b);
    time_op!("Exponent", raw_exp(&a)).unwrap();
}
