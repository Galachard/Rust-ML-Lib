use crate::Tensor;
use crate::ops::raw_ops::*;
use std::ops;

impl ops::Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, _rhs: &Tensor) -> Tensor {
        raw_add(self, _rhs).unwrap()
    }
}

impl ops::Add<f32> for &Tensor {
    type Output = Tensor;

    fn add(self, _rhs: f32) -> Tensor {
        raw_scalar_add(self, _rhs).unwrap()
    }
}

impl ops::Sub<&Tensor> for &Tensor {
    type Output = Tensor;

    fn sub(self, _rhs: &Tensor) -> Tensor {
        raw_sub(self, _rhs).unwrap()
    }
}

impl ops::Sub<f32> for &Tensor {
    type Output = Tensor;

    fn sub(self, _rhs: f32) -> Tensor {
        raw_scalar_add(self, -_rhs).unwrap()
    }
}

impl ops::Mul<&Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, _rhs: &Tensor) -> Tensor {
        raw_mul(self, _rhs).unwrap()
    }
}

impl ops::Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, _rhs: f32) -> Tensor {
        raw_scalar_mul(self, _rhs).unwrap()
    }
}

impl ops::Div<&Tensor> for &Tensor {
    type Output = Tensor;

    fn div(self, _rhs: &Tensor) -> Tensor {
        raw_div(self, _rhs).unwrap()
    }
}

impl ops::Div<f32> for &Tensor {
    type Output = Tensor;

    fn div(self, _rhs: f32) -> Tensor {
        raw_scalar_div(self, _rhs).unwrap()
    }
}
