#![forbid(unsafe_code)]

pub mod context;
pub mod graph;
pub mod learning;
pub mod optimizer;
pub mod schedule;
pub mod testkit;

pub use context::{
    PredictiveContextAllocation, PredictiveContextBank, PredictiveContextBankConfig,
    PredictiveContextCalibration, PredictiveContextCandidate, PredictiveContextCapacityPolicy,
    PredictiveContextIdentity, PredictiveContextLifecycle, PredictiveContextMergeConfig,
    PredictiveContextMergeEvidence, PredictiveContextNoveltyGate, PredictiveContextSelection,
};
pub use graph::{PcFactorId, PcFactorSpec, PcGraphSpec, PcNodeId, PcNodeSpec};
pub use learning::{PcLinearFactorDerivatives, activity_energy_gradient, linear_gaussian_factor};
pub use optimizer::{
    PcParameterOptimizerConfig, PcParameterOptimizerKind, PcParameterOptimizerState,
    PcParameterUpdate, pc_parameter_update,
};
pub use schedule::{PcLearningSchedule, PcSchedulePhase, PcSchedulePlan};
pub use testkit::{DerivativeCheck, central_difference};

use anyhow::{Result, anyhow};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

/// Geometry used to normalize independent latent corrections.
///
/// `PerSample` keeps one clipping norm for every leading batch element and is
/// invariant to batch replication. `PerRow` treats the final axis as features
/// and every leading-axis position as an independent observation, which is the
/// appropriate geometry for token activities. `Global` is retained for
/// experiments that intentionally couple every element of a latent tensor.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PcGradientNormScope {
    Global,
    #[default]
    PerSample,
    PerRow,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct PcInferenceConfig {
    pub steps: usize,
    pub step_size: f32,
    pub latent_decay: f32,
    pub max_grad_norm: Option<f32>,
    pub gradient_norm_scope: PcGradientNormScope,
    pub eps: f32,
}

impl Default for PcInferenceConfig {
    fn default() -> Self {
        Self {
            steps: 1,
            step_size: 0.03,
            latent_decay: 0.0,
            max_grad_norm: Some(1.0),
            gradient_norm_scope: PcGradientNormScope::PerSample,
            eps: 1.0e-8,
        }
    }
}

impl PcInferenceConfig {
    pub fn validate(&self, prefix: &str) -> Result<()> {
        if self.steps == 0 {
            return Err(anyhow!("{prefix}.steps must be > 0"));
        }
        if self.step_size <= 0.0 || !self.step_size.is_finite() {
            return Err(anyhow!("{prefix}.step_size must be finite and > 0"));
        }
        if self.latent_decay < 0.0 || !self.latent_decay.is_finite() {
            return Err(anyhow!("{prefix}.latent_decay must be finite and >= 0"));
        }
        if let Some(max_grad_norm) = self.max_grad_norm
            && (max_grad_norm <= 0.0 || !max_grad_norm.is_finite())
        {
            return Err(anyhow!("{prefix}.max_grad_norm must be finite and > 0"));
        }
        if self.eps <= 0.0 || !self.eps.is_finite() {
            return Err(anyhow!("{prefix}.eps must be finite and > 0"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq)]
pub struct PcInferenceMetrics {
    pub energy_before: Option<f64>,
    pub energy_after: Option<f64>,
    pub energy_delta: Option<f64>,
    pub grad_norm_mean: Option<f64>,
    pub grad_norm_max: Option<f64>,
    pub delta_rms_mean: Option<f64>,
    pub clip_fraction_mean: Option<f64>,
    pub steps_run: usize,
    pub chunks_seen: usize,
    pub chunks_corrected: usize,
    pub skipped_empty_state: usize,
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone)]
pub struct PcTensorUpdate<B: Backend, const D: usize> {
    pub tensor: Tensor<B, D>,
    /// Mean clipping-group gradient norm.
    pub grad_norm: Tensor<B, 1>,
    /// Maximum clipping-group gradient norm.
    pub grad_norm_max: Tensor<B, 1>,
    pub delta_rms: Tensor<B, 1>,
    /// Fraction of clipping groups whose norm exceeded the configured limit.
    pub clip_fraction: Tensor<B, 1>,
}

#[derive(Debug, Clone)]
/// Backend-resident result of a complete predictive-coding inference schedule.
pub struct PcInferenceResult<B: Backend, const D: usize> {
    pub latent: Tensor<B, D>,
    pub last_grad_norm: Tensor<B, 1>,
    pub last_delta_rms: Tensor<B, 1>,
    pub steps_run: usize,
}

/// Runs a complete latent-inference schedule without synchronizing to the host.
///
/// The callback computes the local energy gradient for the current latent.
/// Downstream model code remains responsible for defining that energy.
pub fn run_pc_inference<B, const D: usize, F>(
    mut latent: Tensor<B, D>,
    config: &PcInferenceConfig,
    mut energy_gradient: F,
) -> PcInferenceResult<B, D>
where
    B: Backend,
    F: FnMut(Tensor<B, D>) -> Tensor<B, D>,
{
    let device = latent.device();
    let mut last_grad_norm = Tensor::<B, 1>::zeros([1], &device);
    let mut last_delta_rms = Tensor::<B, 1>::zeros([1], &device);
    for _ in 0..config.steps {
        let update =
            pc_sgd_update_with_metrics(latent.clone(), energy_gradient(latent.clone()), config);
        latent = update.tensor;
        last_grad_norm = update.grad_norm;
        last_delta_rms = update.delta_rms;
    }
    PcInferenceResult {
        latent,
        last_grad_norm,
        last_delta_rms,
        steps_run: config.steps,
    }
}

pub fn pc_sgd_update<B: Backend, const D: usize>(
    latent: Tensor<B, D>,
    grad: Tensor<B, D>,
    config: &PcInferenceConfig,
) -> Tensor<B, D> {
    let clipped_grad = if let Some(max_grad_norm) = config.max_grad_norm {
        let grad_norm = tensor_l2_norms(grad.clone(), config.gradient_norm_scope, config.eps);
        let scale = grad_norm
            .clamp_min(config.eps)
            .recip()
            .mul_scalar(max_grad_norm)
            .clamp_max(1.0);
        grad * scale
    } else {
        grad
    };
    let decay_scale = (1.0 - config.step_size * config.latent_decay).max(0.0);
    latent.mul_scalar(decay_scale) + clipped_grad.mul_scalar(-config.step_size)
}

pub fn pc_sgd_update_with_metrics<B: Backend, const D: usize>(
    latent: Tensor<B, D>,
    grad: Tensor<B, D>,
    config: &PcInferenceConfig,
) -> PcTensorUpdate<B, D> {
    let grad_norms = tensor_l2_norms(grad.clone(), config.gradient_norm_scope, config.eps);
    let clipped_grad = if let Some(max_grad_norm) = config.max_grad_norm {
        let scale = grad_norms
            .clone()
            .clamp_min(config.eps)
            .recip()
            .mul_scalar(max_grad_norm)
            .clamp_max(1.0);
        grad.clone() * scale
    } else {
        grad.clone()
    };
    let decay_scale = (1.0 - config.step_size * config.latent_decay).max(0.0);
    let delta = clipped_grad.mul_scalar(-config.step_size);
    let tensor = latent.mul_scalar(decay_scale) + delta.clone();
    PcTensorUpdate {
        tensor,
        grad_norm: grad_norms.clone().mean().reshape([1]),
        grad_norm_max: grad_norms.clone().max(),
        delta_rms: tensor_rms(delta, config.eps),
        clip_fraction: config.max_grad_norm.map_or_else(
            || Tensor::<B, 1>::zeros([1], &grad.device()),
            |max_grad_norm| {
                grad_norms
                    .greater_elem(max_grad_norm)
                    .float()
                    .mean()
                    .reshape([1])
            },
        ),
    }
}

/// Computes clipping-group L2 norms while preserving broadcastable axes.
///
/// The returned tensor has rank `D`. Global geometry has shape `[1; D]`;
/// per-sample geometry has the input batch extent on axis zero and singleton
/// extents on every remaining axis; per-row geometry preserves every leading
/// axis and uses a singleton final axis.
pub fn tensor_l2_norms<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    scope: PcGradientNormScope,
    eps: f32,
) -> Tensor<B, D> {
    let dims = tensor.shape().dims::<D>();
    match scope {
        PcGradientNormScope::Global => tensor.powf_scalar(2.0).sum().reshape([1; D]),
        PcGradientNormScope::PerSample => {
            let batch = dims[0];
            let features = dims[1..].iter().copied().product::<usize>();
            let mut norm_shape = [1; D];
            norm_shape[0] = batch;
            tensor
                .reshape([batch, features])
                .powf_scalar(2.0)
                .sum_dim(1)
                .reshape(norm_shape)
        }
        PcGradientNormScope::PerRow => {
            let rows = dims[..D.saturating_sub(1)]
                .iter()
                .copied()
                .product::<usize>();
            let width = dims[D - 1];
            let mut norm_shape = dims;
            norm_shape[D - 1] = 1;
            tensor
                .reshape([rows, width])
                .powf_scalar(2.0)
                .sum_dim(1)
                .reshape(norm_shape)
        }
    }
    .sqrt()
    .clamp_min(eps)
}

pub fn tensor_l2_norm<B: Backend, const D: usize>(tensor: Tensor<B, D>, eps: f32) -> Tensor<B, 1> {
    tensor.powf_scalar(2.0).sum().sqrt().clamp_min(eps)
}

pub fn tensor_rms<B: Backend, const D: usize>(tensor: Tensor<B, D>, eps: f32) -> Tensor<B, 1> {
    tensor.powf_scalar(2.0).mean().sqrt().clamp_min(eps)
}

/// Copies one scalar tensor to the host for diagnostics.
///
/// This synchronizes the selected backend and must not be used in the hot
/// inference path.
pub fn diagnostic_scalar_f32<B: Backend>(tensor: Tensor<B, 1>) -> f32 {
    let values = tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("scalar tensor");
    values.first().copied().unwrap_or(0.0)
}

pub mod config {
    pub use crate::optimizer::{PcParameterOptimizerConfig, PcParameterOptimizerKind};
    pub use crate::schedule::PcLearningSchedule;
    pub use crate::{PcGradientNormScope, PcInferenceConfig};
}

pub mod inference {
    pub use crate::{
        PcInferenceResult, PcTensorUpdate, pc_sgd_update, pc_sgd_update_with_metrics,
        run_pc_inference,
    };
}

pub mod metrics {
    pub use crate::{
        PcInferenceMetrics, diagnostic_scalar_f32, tensor_l2_norm, tensor_l2_norms, tensor_rms,
    };
}

pub mod prelude {
    pub use crate::graph::{PcFactorId, PcFactorSpec, PcGraphSpec, PcNodeId, PcNodeSpec};
    pub use crate::learning::{
        PcLinearFactorDerivatives, activity_energy_gradient, linear_gaussian_factor,
    };
    pub use crate::optimizer::{
        PcParameterOptimizerConfig, PcParameterOptimizerKind, PcParameterOptimizerState,
        PcParameterUpdate, pc_parameter_update,
    };
    pub use crate::schedule::{PcLearningSchedule, PcSchedulePhase, PcSchedulePlan};
    pub use crate::testkit::{DerivativeCheck, central_difference};
    pub use crate::{
        PcGradientNormScope, PcInferenceConfig, PcInferenceMetrics, PcInferenceResult,
        PcTensorUpdate, run_pc_inference,
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn default_config_validates() {
        PcInferenceConfig::default()
            .validate("pc")
            .expect("default config should validate");
    }

    #[test]
    fn pc_sgd_update_descends_quadratic_energy() {
        let device = Default::default();
        let latent = Tensor::<TestBackend, 1>::from_floats([2.0, -4.0], &device);
        let grad = latent.clone().mul_scalar(2.0);
        let config = PcInferenceConfig {
            step_size: 0.1,
            max_grad_norm: None,
            ..PcInferenceConfig::default()
        };

        let before = diagnostic_scalar_f32(latent.clone().powf_scalar(2.0).mean());
        let updated = pc_sgd_update(latent, grad, &config);
        let after = diagnostic_scalar_f32(updated.powf_scalar(2.0).mean());

        assert!(after < before, "after={after} before={before}");
    }

    #[test]
    fn max_grad_norm_clips_large_update() {
        let device = Default::default();
        let latent = Tensor::<TestBackend, 1>::zeros([4], &device);
        let grad = Tensor::<TestBackend, 1>::from_floats([100.0, 100.0, 100.0, 100.0], &device);
        let config = PcInferenceConfig {
            step_size: 1.0,
            max_grad_norm: Some(1.0),
            gradient_norm_scope: PcGradientNormScope::Global,
            eps: 1.0e-8,
            ..PcInferenceConfig::default()
        };
        let update = pc_sgd_update_with_metrics(latent, grad, &config);
        let rms = diagnostic_scalar_f32(update.delta_rms);
        assert!(
            rms <= 0.51,
            "clipped delta rms should be bounded, got {rms}"
        );
    }

    #[test]
    fn per_sample_clipping_is_invariant_to_batch_replication() {
        let device = Default::default();
        let config = PcInferenceConfig {
            step_size: 1.0,
            max_grad_norm: Some(5.0),
            gradient_norm_scope: PcGradientNormScope::PerSample,
            eps: 1.0e-8,
            ..PcInferenceConfig::default()
        };
        let two_rows = Tensor::<TestBackend, 2>::from_floats([[3.0, 4.0], [6.0, 8.0]], &device);
        let four_rows = Tensor::<TestBackend, 2>::from_floats(
            [[3.0, 4.0], [6.0, 8.0], [3.0, 4.0], [6.0, 8.0]],
            &device,
        );

        let two = pc_sgd_update(
            Tensor::<TestBackend, 2>::zeros([2, 2], &device),
            two_rows,
            &config,
        )
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("two-row update");
        let four = pc_sgd_update(
            Tensor::<TestBackend, 2>::zeros([4, 2], &device),
            four_rows,
            &config,
        )
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("four-row update");

        assert_eq!(&four[..two.len()], two.as_slice());
        assert_eq!(&four[two.len()..], two.as_slice());
        assert_eq!(two, vec![-3.0, -4.0, -3.0, -4.0]);
    }

    #[test]
    fn grouped_metrics_report_mean_max_and_clip_fraction() {
        let device = Default::default();
        let grad = Tensor::<TestBackend, 2>::from_floats([[3.0, 4.0], [6.0, 8.0]], &device);
        let config = PcInferenceConfig {
            step_size: 1.0,
            max_grad_norm: Some(5.0),
            gradient_norm_scope: PcGradientNormScope::PerSample,
            eps: 1.0e-8,
            ..PcInferenceConfig::default()
        };
        let update = pc_sgd_update_with_metrics(
            Tensor::<TestBackend, 2>::zeros([2, 2], &device),
            grad,
            &config,
        );

        let mean = diagnostic_scalar_f32(update.grad_norm);
        let max = diagnostic_scalar_f32(update.grad_norm_max);
        let clipped = diagnostic_scalar_f32(update.clip_fraction);
        assert!((mean - 7.5).abs() <= 1.0e-5, "mean={mean}");
        assert!((max - 10.0).abs() <= 1.0e-5, "max={max}");
        assert!((clipped - 0.5).abs() <= 1.0e-5, "clipped={clipped}");
    }

    #[test]
    fn per_row_clipping_keeps_token_updates_independent() {
        let device = Default::default();
        let grad = Tensor::<TestBackend, 3>::from_floats(
            [[[3.0, 4.0], [6.0, 8.0]], [[0.0, 5.0], [5.0, 12.0]]],
            &device,
        );
        let config = PcInferenceConfig {
            step_size: 1.0,
            max_grad_norm: Some(5.0),
            gradient_norm_scope: PcGradientNormScope::PerRow,
            eps: 1.0e-8,
            ..PcInferenceConfig::default()
        };
        let update = pc_sgd_update(Tensor::zeros([2, 2, 2], &device), grad, &config)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("per-row update");

        let expected = [
            -3.0,
            -4.0,
            -3.0,
            -4.0,
            0.0,
            -5.0,
            -25.0 / 13.0,
            -60.0 / 13.0,
        ];
        for (actual, expected) in update.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1.0e-6);
        }
    }

    #[test]
    fn diagnostic_norms_do_not_hide_small_nonzero_updates() {
        let device = Default::default();
        let values =
            Tensor::<TestBackend, 2>::from_floats([[1.0e-6, -1.0e-6], [1.0e-6, -1.0e-6]], &device);

        let rms = diagnostic_scalar_f32(tensor_rms(values.clone(), 1.0e-8));
        let norms = tensor_l2_norms(values, PcGradientNormScope::PerSample, 1.0e-8)
            .reshape([2])
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("per-sample norms");

        assert!((rms - 1.0e-6).abs() <= 1.0e-9, "rms={rms}");
        for norm in norms {
            assert!((norm - 2.0_f32.sqrt() * 1.0e-6).abs() <= 1.0e-9);
        }
    }

    #[test]
    fn global_clipping_remains_an_explicit_control() {
        let device = Default::default();
        let grad = Tensor::<TestBackend, 2>::from_floats([[3.0, 4.0], [6.0, 8.0]], &device);
        let config = PcInferenceConfig {
            step_size: 1.0,
            max_grad_norm: Some(5.0),
            gradient_norm_scope: PcGradientNormScope::Global,
            eps: 1.0e-8,
            ..PcInferenceConfig::default()
        };
        let update = pc_sgd_update(
            Tensor::<TestBackend, 2>::zeros([2, 2], &device),
            grad,
            &config,
        )
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("global update");

        assert!(update[0].abs() < 3.0, "global clipping must couple rows");
    }

    #[test]
    fn latent_decay_contracts_zero_gradient_state() {
        let device = Default::default();
        let latent = Tensor::<TestBackend, 1>::from_floats([2.0, -4.0], &device);
        let grad = Tensor::<TestBackend, 1>::zeros([2], &device);
        let config = PcInferenceConfig {
            step_size: 0.25,
            latent_decay: 0.5,
            max_grad_norm: None,
            ..PcInferenceConfig::default()
        };

        let updated = pc_sgd_update(latent, grad, &config);
        let values = updated
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("updated latent");
        assert_eq!(values, vec![1.75, -3.5]);
    }

    #[test]
    fn multi_step_inference_descends_quadratic_energy() {
        let device = Default::default();
        let latent = Tensor::<TestBackend, 1>::from_floats([2.0, -4.0], &device);
        let config = PcInferenceConfig {
            steps: 8,
            step_size: 0.1,
            max_grad_norm: None,
            ..PcInferenceConfig::default()
        };
        let before = diagnostic_scalar_f32(latent.clone().powf_scalar(2.0).mean());
        let result = run_pc_inference(latent, &config, |state| state.mul_scalar(2.0));
        let after = diagnostic_scalar_f32(result.latent.powf_scalar(2.0).mean());

        assert_eq!(result.steps_run, 8);
        assert!(after < before * 0.04, "after={after} before={before}");
    }

    #[test]
    fn invalid_inference_config_is_rejected() {
        let error = PcInferenceConfig {
            steps: 0,
            ..PcInferenceConfig::default()
        }
        .validate("pc")
        .expect_err("zero inference steps must fail");
        assert!(error.to_string().contains("pc.steps must be > 0"));

        let error = PcInferenceConfig {
            max_grad_norm: Some(f32::NAN),
            ..PcInferenceConfig::default()
        }
        .validate("pc")
        .expect_err("non-finite clipping threshold must fail");
        assert!(
            error
                .to_string()
                .contains("pc.max_grad_norm must be finite and > 0")
        );
    }
}
