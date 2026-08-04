use burn::tensor::Tensor;
use burn_ndarray::NdArray;
use burn_pc::{
    PcInferenceConfig, PcParameterOptimizerConfig, linear_gaussian_factor, pc_parameter_update,
    run_pc_inference,
};
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

fn bench_local_learning(c: &mut Criterion) {
    let device = Default::default();
    let parent = Tensor::<BenchBackend, 2>::zeros([256, 256], &device);
    let weight = Tensor::<BenchBackend, 2>::zeros([256, 256], &device);
    let bias = Tensor::<BenchBackend, 1>::zeros([256], &device);
    let activity = Tensor::<BenchBackend, 2>::ones([256, 256], &device);
    c.bench_function("pc/linear_factor_256x256", |b| {
        b.iter(|| {
            linear_gaussian_factor(
                black_box(parent.clone()),
                black_box(weight.clone()),
                black_box(bias.clone()),
                black_box(activity.clone()),
                1.0,
            )
        })
    });

    let parameter = Tensor::<BenchBackend, 2>::ones([256, 256], &device);
    let derivative = Tensor::<BenchBackend, 2>::ones([256, 256], &device);
    let config = PcParameterOptimizerConfig::default();
    c.bench_function("pc/adamw_update_256x256", |b| {
        b.iter(|| {
            pc_parameter_update(
                black_box(parameter.clone()),
                black_box(derivative.clone()),
                None,
                &config,
            )
        })
    });
}

criterion_group!(benches, bench_inference, bench_local_learning);
criterion_main!(benches);
