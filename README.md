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

The primary use in `burn_dragon` is an experimental recurrent-state inference step before normal
TBPTT weight-gradient updates. The state correction is an ablation for shorter effective credit
chains and should remain optional.
