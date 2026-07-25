#![forbid(unsafe_code)]

use anyhow::{Result, anyhow};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct PcInferenceConfig {
    pub steps: usize,
    pub step_size: f32,
    pub latent_decay: f32,
    pub max_grad_norm: Option<f32>,
    pub eps: f32,
}

impl Default for PcInferenceConfig {
    fn default() -> Self {
        Self {
            steps: 1,
            step_size: 0.03,
            latent_decay: 0.0,
            max_grad_norm: Some(1.0),
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
    pub steps_run: usize,
    pub chunks_seen: usize,
    pub chunks_corrected: usize,
    pub skipped_empty_state: usize,
    pub elapsed_ms: f64,
}

#[derive(Debug, Clone)]
pub struct PcTensorUpdate<B: Backend, const D: usize> {
    pub tensor: Tensor<B, D>,
    pub grad_norm: Tensor<B, 1>,
    pub delta_rms: Tensor<B, 1>,
}

pub fn pc_sgd_update<B: Backend, const D: usize>(
    latent: Tensor<B, D>,
    grad: Tensor<B, D>,
    config: &PcInferenceConfig,
) -> Tensor<B, D> {
    let clipped_grad = if let Some(max_grad_norm) = config.max_grad_norm {
        let grad_norm = tensor_l2_norm(grad.clone(), config.eps);
        let scale = grad_norm
            .add_scalar(config.eps)
            .recip()
            .mul_scalar(max_grad_norm)
            .clamp_max(1.0);
        grad * scale.reshape([1; D])
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
    let grad_norm = tensor_l2_norm(grad.clone(), config.eps);
    let clipped_grad = if let Some(max_grad_norm) = config.max_grad_norm {
        let scale = grad_norm
            .clone()
            .add_scalar(config.eps)
            .recip()
            .mul_scalar(max_grad_norm)
            .clamp_max(1.0);
        grad.clone() * scale.reshape([1; D])
    } else {
        grad.clone()
    };
    let decay_scale = (1.0 - config.step_size * config.latent_decay).max(0.0);
    let delta = clipped_grad.mul_scalar(-config.step_size);
    let tensor = latent.mul_scalar(decay_scale) + delta.clone();
    PcTensorUpdate {
        tensor,
        grad_norm,
        delta_rms: tensor_rms(delta, config.eps),
    }
}

pub fn tensor_l2_norm<B: Backend, const D: usize>(tensor: Tensor<B, D>, eps: f32) -> Tensor<B, 1> {
    tensor.powf_scalar(2.0).sum().add_scalar(eps).sqrt()
}

pub fn tensor_rms<B: Backend, const D: usize>(tensor: Tensor<B, D>, eps: f32) -> Tensor<B, 1> {
    tensor.powf_scalar(2.0).mean().add_scalar(eps).sqrt()
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
