use crate::layer::Layer;
use crate::optimizer::Optimizer;
use crate::{Parameter, Tensor};
use indicatif::{ProgressBar, ProgressStyle};

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
        batch_size: usize,
        loss_function: impl Fn(&Tensor, &Tensor) -> Tensor,
    ) {
        assert_eq!(inputs.len(), targets.len());

        let n = inputs.len();

        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            let pb = ProgressBar::new((n / batch_size) as u64);
            pb.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] {bar:40.white/grey} {pos}/{len} loss={msg}",
                )
                .unwrap(),
            );

            let mut i = 0;
            while i < n {
                let end = (i + batch_size).min(n);

                // Zero grad once per batch
                model.zero_grad(optimizer);

                for j in i..end {
                    let pred = model.forward(&inputs[j]);
                    let loss = loss_function(&pred, &targets[j]);

                    model.backward(&loss);
                    total_loss += loss.data[0];
                }
                pb.set_message(format!("{:.4}", total_loss / (end as f32)));
                pb.inc(1);

                // Apply accumulated gradients
                model.step(optimizer);

                i = end;
            }

            pb.finish_with_message(format!("Epoch {}: loss = {}", epoch, total_loss / n as f32));
        }
    }
}
