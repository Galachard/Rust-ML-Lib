use std::fmt;

#[derive(Debug)]
/// Enum representing tensor-related errors.
pub enum TensorError {
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    InvalidIndex {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
    GradientMissing,
    DimensionError(String),
    NotImplemented(&'static str),
    DivisionByZero,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            TensorError::InvalidIndex { index, shape } => write!(
                f,
                "Invalid index {:?} for tensor of shape {:?}",
                index, shape
            ),
            TensorError::GradientMissing => write!(f, "Gradient requested but not available"),
            TensorError::DimensionError(msg) => write!(f, "Dimension error: {}", msg),
            TensorError::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
            TensorError::DivisionByZero => write!(f, "Division by zero error"),
        }
    }
}

impl std::error::Error for TensorError {}
