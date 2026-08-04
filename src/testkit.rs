/// Result of comparing an analytic derivative vector with central differences.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DerivativeCheck {
    pub max_abs_error: f32,
    pub mean_abs_error: f32,
    pub elements: usize,
}

impl DerivativeCheck {
    pub fn compare(analytic: &[f32], numerical: &[f32]) -> Self {
        assert_eq!(
            analytic.len(),
            numerical.len(),
            "derivative vectors must have equal lengths"
        );
        let elements = analytic.len();
        let (max_abs_error, total_abs_error) = analytic
            .iter()
            .zip(numerical)
            .map(|(analytic, numerical)| (analytic - numerical).abs())
            .fold((0.0_f32, 0.0_f32), |(max, total), error| {
                (max.max(error), total + error)
            });
        Self {
            max_abs_error,
            mean_abs_error: if elements == 0 {
                0.0
            } else {
                total_abs_error / elements as f32
            },
            elements,
        }
    }

    pub fn passes(&self, max_abs_tolerance: f32) -> bool {
        self.max_abs_error.is_finite() && self.max_abs_error <= max_abs_tolerance
    }
}

/// Evaluates a scalar function at central finite-difference offsets.
///
/// This host-side helper is intended for small numerical contract tests, never
/// for a training hot path.
pub fn central_difference(
    point: &[f32],
    epsilon: f32,
    mut evaluate: impl FnMut(&[f32]) -> f32,
) -> Vec<f32> {
    assert!(epsilon.is_finite() && epsilon > 0.0);
    let mut probe = point.to_vec();
    (0..point.len())
        .map(|index| {
            probe[index] = point[index] + epsilon;
            let plus = evaluate(&probe);
            probe[index] = point[index] - epsilon;
            let minus = evaluate(&probe);
            probe[index] = point[index];
            (plus - minus) / (2.0 * epsilon)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn central_difference_recovers_quadratic_gradient() {
        let point = [1.5_f32, -2.0];
        let numerical = central_difference(&point, 1.0e-3, |values| {
            values.iter().map(|value| value * value).sum()
        });
        let check = DerivativeCheck::compare(&[3.0, -4.0], &numerical);
        assert!(check.passes(5.0e-4), "{check:?}");
    }
}
