use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

/// Backend-resident values for a local affine prediction factor.
#[derive(Debug, Clone)]
pub struct PcLinearFactorDerivatives<B: Backend> {
    pub prediction: Tensor<B, 2>,
    pub error: Tensor<B, 2>,
    pub parent_gradient: Tensor<B, 2>,
    pub weight_gradient: Tensor<B, 2>,
    pub bias_gradient: Tensor<B, 1>,
    pub energy: Tensor<B, 1>,
}

/// Computes a Gaussian local-energy factor and all of its analytic derivatives.
///
/// Shapes are `parent=[batch,input]`, `weight=[input,output]`,
/// `activity=[batch,output]`, and `bias=[output]`. Gradients are means over the
/// batch, matching the returned per-sample mean energy. The output dimensions
/// are summed because they belong to one factor observation.
pub fn linear_gaussian_factor<B: Backend>(
    parent: Tensor<B, 2>,
    weight: Tensor<B, 2>,
    bias: Tensor<B, 1>,
    activity: Tensor<B, 2>,
    precision: f32,
) -> PcLinearFactorDerivatives<B> {
    assert!(precision.is_finite() && precision > 0.0);
    let [batch, input] = parent.shape().dims::<2>();
    let [weight_input, output] = weight.shape().dims::<2>();
    assert_eq!(input, weight_input, "linear factor input shape mismatch");
    assert_eq!(activity.shape().dims::<2>(), [batch, output]);
    assert_eq!(bias.shape().dims::<1>(), [output]);

    let prediction = parent.clone().matmul(weight.clone()) + bias.reshape([1, output]);
    let error = (activity - prediction.clone()).mul_scalar(precision);
    let scale = 1.0 / batch.max(1) as f32;
    let parent_gradient = error
        .clone()
        .matmul(weight.clone().transpose())
        .mul_scalar(-scale);
    let weight_gradient = parent.transpose().matmul(error.clone()).mul_scalar(-scale);
    let bias_gradient = error
        .clone()
        .sum_dim(0)
        .reshape([output])
        .mul_scalar(-scale);
    let energy = error
        .clone()
        .mul(error.clone().div_scalar(precision))
        .sum_dim(1)
        .mean()
        .mul_scalar(0.5)
        .reshape([1]);

    PcLinearFactorDerivatives {
        prediction,
        error,
        parent_gradient,
        weight_gradient,
        bias_gradient,
        energy,
    }
}

/// Combines the error of an activity's own prediction with one-hop child VJPs.
pub fn activity_energy_gradient<B: Backend>(
    own_error: Tensor<B, 2>,
    child_parent_gradients: impl IntoIterator<Item = Tensor<B, 2>>,
) -> Tensor<B, 2> {
    child_parent_gradients
        .into_iter()
        .fold(own_error, |gradient, child| gradient + child)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn values<const D: usize>(tensor: Tensor<TestBackend, D>) -> Vec<f32> {
        tensor
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("f32 tensor")
    }

    fn energy(parent: &[f32; 2], weight: &[f32; 4], bias: &[f32; 2]) -> f32 {
        let target = [0.25, -0.75];
        let prediction = [
            parent[0] * weight[0] + parent[1] * weight[2] + bias[0],
            parent[0] * weight[1] + parent[1] * weight[3] + bias[1],
        ];
        0.5 * ((target[0] - prediction[0]).powi(2) + (target[1] - prediction[1]).powi(2))
    }

    #[test]
    fn analytic_linear_derivatives_match_finite_difference() {
        let device = Default::default();
        let parent_values = [0.5, -1.25];
        let weight_values = [0.3, -0.2, 0.7, 0.4];
        let bias_values = [0.1, -0.3];
        let result = linear_gaussian_factor(
            Tensor::<TestBackend, 2>::from_data(
                TensorData::new(parent_values.to_vec(), [1, 2]),
                &device,
            ),
            Tensor::<TestBackend, 2>::from_data(
                TensorData::new(weight_values.to_vec(), [2, 2]),
                &device,
            ),
            Tensor::<TestBackend, 1>::from_floats(bias_values, &device),
            Tensor::<TestBackend, 2>::from_floats([[0.25, -0.75]], &device),
            1.0,
        );

        let epsilon = 1.0e-3;
        let analytic_weight = values(result.weight_gradient);
        for index in 0..weight_values.len() {
            let mut plus = weight_values;
            let mut minus = weight_values;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numerical = (energy(&parent_values, &plus, &bias_values)
                - energy(&parent_values, &minus, &bias_values))
                / (2.0 * epsilon);
            assert!(
                (analytic_weight[index] - numerical).abs() < 2.0e-4,
                "weight {index}: analytic={} numerical={numerical}",
                analytic_weight[index]
            );
        }

        let analytic_parent = values(result.parent_gradient);
        for index in 0..parent_values.len() {
            let mut plus = parent_values;
            let mut minus = parent_values;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let numerical = (energy(&plus, &weight_values, &bias_values)
                - energy(&minus, &weight_values, &bias_values))
                / (2.0 * epsilon);
            assert!(
                (analytic_parent[index] - numerical).abs() < 2.0e-4,
                "parent {index}: analytic={} numerical={numerical}",
                analytic_parent[index]
            );
        }
    }
}
