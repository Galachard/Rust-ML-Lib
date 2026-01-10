use crate::layer::Layer;
use crate::{Parameter, Tensor};

pub struct ELU {
    alpha: f32,
}

impl Layer for ELU {
    fn forward(&self, input: &Tensor) -> Tensor {
        // ELU: x if x >= 0 else alpha * (exp(x) - 1)
        let alpha = self.alpha;

        input.map(
            move |x| {
                if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) }
            },
            move |x| {
                if x >= 0.0 { 1.0 } else { alpha * x.exp() }
            },
        )
    }

    fn parameters(&self) -> Vec<Parameter> {
        vec![]
    }
}

impl ELU {
    pub fn new(alpha: f32) -> Self {
        ELU { alpha }
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self::new(1.0)
    }
}
