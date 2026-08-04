# burn_pc

`burn_pc` provides backend-generic building blocks for predictive-coding and
other local-learning pipelines in Burn. It separates three concerns that are
often conflated:

- a factor graph describes which activities predict which other activities;
- an inference schedule relaxes unclamped activities using local energy VJPs;
- an optimizer transforms already-computed local parameter derivatives.

No API in this crate requires an autodiff backend or traverses a global network
backward graph. An AdamW update here is only a tensor update rule.

```rust
use burn::tensor::{Tensor, backend::Backend};
use burn_pc::{PcInferenceConfig, run_pc_inference};

fn relax_activity<B: Backend>(activity: Tensor<B, 2>) -> Tensor<B, 2> {
    let config = PcInferenceConfig {
        steps: 4,
        ..PcInferenceConfig::default()
    };
    run_pc_inference(activity, &config, |state| {
        // One local factor's energy derivative with respect to this activity.
        state.mul_scalar(2.0)
    })
    .latent
}
```

The crate includes:

- versioned `PcGraphSpec` metadata for activities and local factors;
- equilibrium and incremental schedule plans;
- backend-resident inference, clipping, and diagnostic primitives;
- global, per-sample, and per-row clipping geometries for independent token activities;
- analytic affine Gaussian factor derivatives;
- plain-backend SGD, momentum, and AdamW parameter transforms;
- finite-difference helpers for downstream numerical VJP tests;
- Criterion coverage for inference, local factors, and parameter updates.

Diagnostic scalar readback is deliberately separate from hot-path operations
because it synchronizes the selected backend. Model layout, datasets, runtime
orchestration, and checkpoint policy remain downstream responsibilities.

In `burn_dragon`, the canonical integration builds a layer-local factor graph,
uses analytic plain-backend VJPs, aggregates derivatives from every use of the
shared Dragon weights, and invokes the normal optimizer only as the final
parameter-update transform. The older recurrent-state correction auxiliary is
a distinct global-backprop ablation and is not the local-learning contract.
