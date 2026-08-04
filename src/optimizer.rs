use anyhow::{Result, anyhow};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PcParameterOptimizerKind {
    Sgd,
    Momentum,
    #[default]
    Adamw,
}

/// Optimizer applied to local predictive-coding parameter derivatives.
///
/// This configuration never computes derivatives and has no dependency on an
/// autodiff backend. AdamW here is a tensor update rule, not a backward pass.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct PcParameterOptimizerConfig {
    pub kind: PcParameterOptimizerKind,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
}

impl Default for PcParameterOptimizerConfig {
    fn default() -> Self {
        Self {
            kind: PcParameterOptimizerKind::Adamw,
            learning_rate: 1.0e-3,
            weight_decay: 0.0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1.0e-8,
        }
    }
}

impl PcParameterOptimizerConfig {
    pub fn validate(&self, prefix: &str) -> Result<()> {
        if self.learning_rate <= 0.0 || !self.learning_rate.is_finite() {
            return Err(anyhow!("{prefix}.learning_rate must be finite and > 0"));
        }
        if self.weight_decay < 0.0 || !self.weight_decay.is_finite() {
            return Err(anyhow!("{prefix}.weight_decay must be finite and >= 0"));
        }
        for (name, value) in [("beta1", self.beta1), ("beta2", self.beta2)] {
            if !(0.0..1.0).contains(&value) || !value.is_finite() {
                return Err(anyhow!("{prefix}.{name} must be finite and in [0, 1)"));
            }
        }
        if self.eps <= 0.0 || !self.eps.is_finite() {
            return Err(anyhow!("{prefix}.eps must be finite and > 0"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PcParameterOptimizerState<B: Backend, const D: usize> {
    pub first_moment: Tensor<B, D>,
    pub second_moment: Tensor<B, D>,
    pub step: u64,
}

#[derive(Debug, Clone)]
pub struct PcParameterUpdate<B: Backend, const D: usize> {
    pub parameter: Tensor<B, D>,
    pub state: PcParameterOptimizerState<B, D>,
}

pub fn pc_parameter_update<B: Backend, const D: usize>(
    parameter: Tensor<B, D>,
    derivative: Tensor<B, D>,
    state: Option<PcParameterOptimizerState<B, D>>,
    config: &PcParameterOptimizerConfig,
) -> PcParameterUpdate<B, D> {
    let device = parameter.device();
    let shape = parameter.shape();
    let mut state = state.unwrap_or_else(|| PcParameterOptimizerState {
        first_moment: Tensor::zeros(shape.clone(), &device),
        second_moment: Tensor::zeros(shape, &device),
        step: 0,
    });
    state.step = state.step.saturating_add(1);

    let parameter = match config.kind {
        PcParameterOptimizerKind::Sgd => {
            parameter.mul_scalar(1.0 - config.learning_rate * config.weight_decay)
                - derivative.mul_scalar(config.learning_rate)
        }
        PcParameterOptimizerKind::Momentum => {
            state.first_moment = state.first_moment.mul_scalar(config.beta1)
                + derivative.mul_scalar(1.0 - config.beta1);
            parameter.mul_scalar(1.0 - config.learning_rate * config.weight_decay)
                - state.first_moment.clone().mul_scalar(config.learning_rate)
        }
        PcParameterOptimizerKind::Adamw => {
            state.first_moment = state.first_moment.mul_scalar(config.beta1)
                + derivative.clone().mul_scalar(1.0 - config.beta1);
            state.second_moment = state.second_moment.mul_scalar(config.beta2)
                + derivative.square().mul_scalar(1.0 - config.beta2);
            let first_correction = 1.0 - (config.beta1 as f64).powf(state.step as f64);
            let second_correction = 1.0 - (config.beta2 as f64).powf(state.step as f64);
            let update = state.first_moment.clone().div_scalar(first_correction)
                / state
                    .second_moment
                    .clone()
                    .div_scalar(second_correction)
                    .sqrt()
                    .add_scalar(config.eps);
            parameter.mul_scalar(1.0 - config.learning_rate * config.weight_decay)
                - update.mul_scalar(config.learning_rate)
        }
    };

    PcParameterUpdate {
        parameter: parameter.detach(),
        state,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn scalar(tensor: Tensor<TestBackend, 1>) -> f32 {
        tensor
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32 tensor")[0]
    }

    #[test]
    fn sgd_descends_positive_derivative() {
        let device = Default::default();
        let update = pc_parameter_update(
            Tensor::<TestBackend, 1>::from_floats([1.0], &device),
            Tensor::<TestBackend, 1>::from_floats([2.0], &device),
            None,
            &PcParameterOptimizerConfig {
                kind: PcParameterOptimizerKind::Sgd,
                learning_rate: 0.1,
                ..PcParameterOptimizerConfig::default()
            },
        );
        assert!((scalar(update.parameter) - 0.8).abs() < 1.0e-6);
    }

    #[test]
    fn adamw_first_step_has_bias_correction() {
        let device = Default::default();
        let update = pc_parameter_update(
            Tensor::<TestBackend, 1>::from_floats([1.0], &device),
            Tensor::<TestBackend, 1>::from_floats([2.0], &device),
            None,
            &PcParameterOptimizerConfig {
                kind: PcParameterOptimizerKind::Adamw,
                learning_rate: 0.1,
                ..PcParameterOptimizerConfig::default()
            },
        );
        assert!((scalar(update.parameter) - 0.9).abs() < 2.0e-5);
        assert_eq!(update.state.step, 1);
    }
}
