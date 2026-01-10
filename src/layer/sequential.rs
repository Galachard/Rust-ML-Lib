use crate::layer::Layer;
use crate::losses::mse;
use crate::optimizer::Optimizer;
use crate::{Parameter, Tensor};

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Layer for Sequential {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.layers
            .iter()
            .fold(input.clone(), |x, layer| layer.forward(&x))
    }

    fn parameters(&self) -> Vec<Parameter> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: vec![] }
    }

    pub fn add<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    pub fn backward(&self, loss: &Tensor) {
        loss.backward();
    }

    pub fn step<O: Optimizer>(&self, opt: &mut O) {
        opt.step();
    }

    pub fn zero_grad<O: Optimizer>(&self, opt: &mut O) {
        opt.zero_grad();
    }

    pub fn train(
        model: &Sequential,
        optimizer: &mut impl Optimizer,
        inputs: &[Tensor],
        targets: &[Tensor],
        epochs: usize,
    ) {
        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (x, y) in inputs.iter().zip(targets.iter()) {
                let pred = model.forward(x);
                let loss = mse(&pred, y); // TODO: should allow different losses

                model.backward(&loss);
                model.step(optimizer);
                model.zero_grad(optimizer);

                total_loss += loss.data[0];
            }

            println!("Epoch {epoch}: loss = {}", total_loss / inputs.len() as f32);
        }
    }
}
