mod regression;

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};
use burn::nn::loss::{MseLoss, Reduction};
use burn::optim::{SgdConfig, GradientsParams};


#[derive(Module, Debug)]
struct Model {
    layer: Linear,
}

impl Model {
    fn new(device: &Device) -> Self {
        Self { layer: LinearConfig::new(1, 1).init(device) }
    }

    fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        self.layer.forward(x)
    }
}

fn main() {
    let device = Device::wgpu(Default::default()).autodiff();

    let x = Tensor::<2>::from_floats([[1.0], [2.0], [3.0], [4.0]], &device);
    let y = Tensor::<2>::from_floats([[5.1], [7.2], [8.9], [11.0]], &device);

    let mut model = Model::new(&device);
    let mut optimizer = SgdConfig::new().init();
    let lossfn = MseLoss::new();

    for _ in 0..1000 {
        let y_hat = model.forward(x.clone());

        let loss = lossfn.forward(y_hat, y.clone(), Reduction::Mean);

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(0.01, model, grads_params);
    }

    let test_x = Tensor::<2>::from_floats([[5.0]], &device);
    println!("Prediction for x = 5: {}", model.forward(test_x).into_scalar::<f32>());
}
