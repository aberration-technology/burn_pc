# burn_pc

`burn_pc` provides small, backend-generic predictive-coding update primitives for Burn tensors.

The crate intentionally does not own a training loop, model layout, dataset, or optimizer. Downstream
training crates decide which tensors are treated as latents, which local energy is minimized, and how
often inference is applied.

```rust
use burn_pc::{PcInferenceConfig, pc_sgd_update};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

fn correct_latent<B: Backend, const D: usize>(
    latent: Tensor<B, D>,
    grad: Tensor<B, D>,
) -> Tensor<B, D> {
    pc_sgd_update(latent, grad, &PcInferenceConfig::default())
}
```

For a complete backend-resident inference schedule, provide the local energy
gradient as a callback:

```rust
use burn_pc::{PcInferenceConfig, run_pc_inference};
# use burn::tensor::Tensor;
# use burn::tensor::backend::Backend;

fn infer<B: Backend, const D: usize>(latent: Tensor<B, D>) -> Tensor<B, D> {
    let config = PcInferenceConfig {
        steps: 4,
        ..PcInferenceConfig::default()
    };
    run_pc_inference(latent, &config, |state| {
        // Gradient of the downstream model's local energy.
        state.mul_scalar(2.0)
    })
    .latent
}
```

`run_pc_inference` performs no diagnostic readback. The `diagnostic_scalar_f32`
helper is intentionally separate because it synchronizes the selected backend.
The normal dependency does not force a concrete Burn backend.

The primary use in `burn_dragon` is an experimental recurrent-state inference step before normal
TBPTT weight-gradient updates. The state correction is an ablation for shorter effective credit
chains and should remain optional.
