use burn::tensor::Tensor;
use burn_ndarray::NdArray;
use burn_pc::{PcInferenceConfig, run_pc_inference};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

type BenchBackend = NdArray<f32>;

fn bench_inference(c: &mut Criterion) {
    let device = Default::default();
    let latent = Tensor::<BenchBackend, 2>::zeros([256, 256], &device);
    for steps in [1, 4, 8] {
        let config = PcInferenceConfig {
            steps,
            max_grad_norm: None,
            ..PcInferenceConfig::default()
        };
        c.bench_function(&format!("pc/inference_256x256_steps_{steps}"), |b| {
            b.iter(|| {
                run_pc_inference(black_box(latent.clone()), &config, |state| {
                    state.mul_scalar(2.0)
                })
            })
        });
    }
}

criterion_group!(benches, bench_inference);
criterion_main!(benches);
