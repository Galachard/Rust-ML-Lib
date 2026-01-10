use crate::error::TensorError;
use crate::tensor::*;

/// Check if two tensors have the same shape.
pub(super) fn check_shapes(a: &Tensor, b: &Tensor) -> Result<(), TensorError> {
    if a.shape() != b.shape() {
        Err(TensorError::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        })
    } else {
        Ok(())
    }
}

/// Transpose a rank-2 tensor.
pub fn transpose(t: &Tensor) -> Result<Tensor, TensorError> {
    if t.shape.len() != 2 {
        return Err(TensorError::DimensionError(
            "Transpose requires rank-2 tensor".to_string(),
        ));
    }

    let rows = t.shape[0];
    let cols = t.shape[1];

    let mut out = vec![0.0; rows * cols];

    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = t.data[i * cols + j];
        }
    }

    Ok(Tensor::from_vec_with_node(
        out,
        vec![cols, rows],
        t.node.clone(),
    ))
}

/// Element-wise multiplication of two tensors without gradient tracking.
pub fn raw_mul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    check_shapes(a, b)?;

    let mut out = a.data.clone();
    for (i, v) in out.iter_mut().enumerate() {
        *v *= b.data[i];
    }

    Ok(Tensor::new(out, a.shape.clone()))
}

/// Element-wise multiplication of a tensor by a scalar without gradient tracking.
pub fn raw_scalar_mul(t: &Tensor, scalar: f32) -> Result<Tensor, TensorError> {
    let mut out = t.data.clone();
    for v in out.iter_mut() {
        *v *= scalar;
    }
    Ok(Tensor::new(out, t.shape.clone()))
}

/// Element-wise division of two tensors without gradient tracking.
pub fn raw_div(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    check_shapes(a, b)?;

    let mut out = a.data.clone();
    for (i, v) in out.iter_mut().enumerate() {
        if b.data[i] == 0.0 {
            return Err(TensorError::DivisionByZero);
        }
        *v /= b.data[i];
    }

    Ok(Tensor::new(out, a.shape.clone()))
}

/// Element-wise division of a tensor by a scalar without gradient tracking.
pub fn raw_scalar_div(t: &Tensor, scalar: f32) -> Result<Tensor, TensorError> {
    if scalar == 0.0 {
        return Err(TensorError::DivisionByZero);
    }
    let mut out = t.data.clone();
    for v in out.iter_mut() {
        *v /= scalar;
    }
    Ok(Tensor::new(out, t.shape.clone()))
}

/// Matrix multiplication of two rank-2 tensors without gradient tracking.
pub fn raw_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    // Custom shape check due to matrix multiplication
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err(TensorError::DimensionError(
            "matmul requires rank-2 tensors".to_string(),
        ));
    }

    let (m, k1) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);

    if k1 != k2 {
        return Err(TensorError::DimensionError(format!(
            "matmul dimension mismatch: {}x{} * {}x{}",
            m, k1, k2, n
        )));
    }

    let mut out_data = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..k1 {
                sum += a.data[i * k1 + k] * b.data[k * n + j];
            }
            out_data[i * n + j] = sum;
        }
    }
    Ok(Tensor::new(out_data, vec![m, n]))
}

/// Element-wise addition of two tensors without gradient tracking.
pub fn raw_add(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    check_shapes(a, b)?;

    let mut out = a.data.clone();
    for (i, v) in out.iter_mut().enumerate() {
        *v += b.data[i];
    }

    Ok(Tensor::new(out, b.shape.clone()))
}

/// Element-wise subtraction of two tensors without gradient tracking.
pub fn raw_sub(a: &Tensor, b: &Tensor) -> Result<Tensor, TensorError> {
    check_shapes(a, b)?;

    let mut out = a.data.clone();
    for (i, v) in out.iter_mut().enumerate() {
        *v -= b.data[i];
    }

    Ok(Tensor::new(out, b.shape.clone()))
}

/// Element-wise addition of a tensor and a scalar without gradient tracking.
pub fn raw_scalar_add(t: &Tensor, scalar: f32) -> Result<Tensor, TensorError> {
    let mut out = t.data.clone();
    for v in out.iter_mut() {
        *v += scalar;
    }
    Ok(Tensor::new(out, t.shape.clone()))
}

/// Element-wise exponential of a tensor without gradient tracking.
pub fn raw_exp(t: &Tensor) -> Result<Tensor, TensorError> {
    let mut out = t.data.clone();
    for v in out.iter_mut() {
        *v = v.exp();
    }
    Ok(Tensor::new(out, t.shape.clone()))
}

/// Element-wise inversion of a tensor without gradient tracking.
pub fn raw_inv(t: &Tensor) -> Result<Tensor, TensorError> {
    let mut out = t.data.clone();
    for v in out.iter_mut() {
        *v = 1.0 / *v;
    }
    Ok(Tensor::new(out, t.shape.clone()))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_transpose() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let t = transpose(&t).unwrap();
        assert_eq!(t.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_raw_multiplication() {
        let t1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let t2 = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let res = raw_mul(&t1, &t2).unwrap();
        assert_eq!(res.data(), &[1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_raw_scalar_multiplication() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let scalar = 4.0;
        let res = raw_scalar_mul(&t, scalar).unwrap();
        assert_eq!(res.data(), &[4.0, 8.0, 12.0]);
    }

    #[test]
    fn test_raw_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2]);

        let out = raw_add(&a, &b).unwrap();

        assert_eq!(out.data(), &[4.0, 6.0]);
    }

    #[test]
    fn test_raw_exp() {
        let a = Tensor::from_vec(vec![0.0, 1.0, 2.0], vec![3]);
        let out = raw_exp(&a).unwrap();
        let one: f32 = 1.0;
        let two: f32 = 2.0;
        assert_eq!(out.data(), &[1.0, one.exp(), two.exp()]);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![1, 2]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);
        let out = raw_matmul(&a, &b).unwrap();
        assert_eq!(out.data(), &[11.0]);
    }

    #[test]
    fn test_matmul_shape() {
        let a = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![2, 3]);
        let b = Tensor::from_vec(
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![3, 3],
        );
        let c = Tensor::from_vec(
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![3, 4],
        );

        let res1 = raw_matmul(&a, &b).unwrap();
        let res2 = raw_matmul(&b, &c).unwrap();
        let res3 = raw_matmul(&a, &c).unwrap();

        assert_eq!(res1.shape(), &[2, 3]);
        assert_eq!(res2.shape(), &[3, 4]);
        assert_eq!(res3.shape(), &[2, 4]);
    }

    #[test]
    fn test_shape_checking() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let res1 = raw_add(&a, &b);
        let res2 = raw_mul(&a, &b);
        assert!(res1.is_err());
        assert!(res2.is_err());
    }

    #[test]
    fn test_matmul_shape_checking() {
        let a = Tensor::from_vec(vec![1.0, 2.0], vec![2, 1]);
        let b = Tensor::from_vec(vec![3.0, 4.0], vec![2, 1]);
        let res1 = raw_matmul(&a, &b);
        assert!(res1.is_err());

        let c = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 1, 3]);
        let res2 = raw_matmul(&a, &c);
        assert!(res2.is_err());

        let d = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1]);
        let res3 = raw_matmul(&a, &d);
        assert!(res3.is_err());
    }
}
