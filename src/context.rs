use serde::{Deserialize, Serialize};

/// Online calibration for one predictive context expert.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq)]
pub struct PredictiveContextCalibration {
    pub observations: u64,
    pub mean_loss: f64,
    pub loss_variance: f64,
}

impl PredictiveContextCalibration {
    pub fn loss_scale(self, minimum_scale: f64) -> f64 {
        self.loss_variance.max(0.0).sqrt().max(minimum_scale)
    }

    fn observe(&mut self, loss: f64, update_rate: f64) {
        if self.observations == 0 {
            self.observations = 1;
            self.mean_loss = loss;
            self.loss_variance = 0.0;
            return;
        }
        let delta = loss - self.mean_loss;
        self.mean_loss += update_rate * delta;
        self.loss_variance =
            (1.0 - update_rate) * (self.loss_variance + update_rate * delta * delta);
        self.observations = self.observations.saturating_add(1);
    }
}

/// Configuration for causal context discovery from per-expert predictive
/// losses. The bank is model-agnostic: downstream code decides how causal
/// observations are encoded and scored.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct PredictiveContextBankConfig {
    pub max_contexts: usize,
    pub minimum_observations: u64,
    pub calibration_update_rate: f64,
    pub novelty_standard_deviations: f64,
    pub novelty_absolute_margin: f64,
    pub minimum_loss_scale: f64,
}

impl Default for PredictiveContextBankConfig {
    fn default() -> Self {
        Self {
            max_contexts: 64,
            minimum_observations: 8,
            calibration_update_rate: 0.1,
            novelty_standard_deviations: 4.0,
            novelty_absolute_margin: 0.25,
            minimum_loss_scale: 0.05,
        }
    }
}

impl PredictiveContextBankConfig {
    pub fn validate(self) -> Result<(), String> {
        if self.max_contexts == 0 {
            return Err("predictive context max_contexts must be > 0".to_string());
        }
        if self.minimum_observations == 0 {
            return Err("predictive context minimum_observations must be > 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.calibration_update_rate)
            || self.calibration_update_rate == 0.0
            || !self.calibration_update_rate.is_finite()
        {
            return Err(
                "predictive context calibration_update_rate must be finite and in (0, 1]"
                    .to_string(),
            );
        }
        for (name, value) in [
            (
                "novelty_standard_deviations",
                self.novelty_standard_deviations,
            ),
            ("novelty_absolute_margin", self.novelty_absolute_margin),
            ("minimum_loss_scale", self.minimum_loss_scale),
        ] {
            if value < 0.0 || !value.is_finite() {
                return Err(format!("predictive context {name} must be finite and >= 0"));
            }
        }
        if self.minimum_loss_scale == 0.0 {
            return Err("predictive context minimum_loss_scale must be > 0".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct PredictiveContextCandidate {
    pub context_index: usize,
    pub loss: f64,
    pub calibrated_surprise: Option<f64>,
    pub rejected_as_novel: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct PredictiveContextSelection {
    pub context_index: usize,
    pub created: bool,
    pub capacity_exhausted: bool,
    /// Every mature expert rejected the observation under its calibrated
    /// predictive-loss envelope. Callers may feed this into a sequential
    /// novelty gate before allocating a context.
    pub novel_evidence: bool,
    pub candidates: Vec<PredictiveContextCandidate>,
}

/// Checkpointable sequential gate for predictive context creation.
///
/// A single transient loss spike must not allocate a permanent expert. The
/// gate confirms a distribution shift only after the configured number of
/// consecutive novel observations. It is separate from
/// [`PredictiveContextBank`] so read-only evaluation can call `select` without
/// mutating discovery state.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct PredictiveContextNoveltyGate {
    required_consecutive_observations: u64,
    consecutive_novel_observations: u64,
}

impl PredictiveContextNoveltyGate {
    pub fn new(required_consecutive_observations: u64) -> Result<Self, String> {
        if required_consecutive_observations == 0 {
            return Err(
                "predictive context novelty confirmations must be greater than zero".to_string(),
            );
        }
        Ok(Self {
            required_consecutive_observations,
            consecutive_novel_observations: 0,
        })
    }

    pub fn required_consecutive_observations(self) -> u64 {
        self.required_consecutive_observations
    }

    pub fn consecutive_novel_observations(self) -> u64 {
        self.consecutive_novel_observations
    }

    /// Observe one mutable training-stream decision. Returns `true` exactly
    /// when a new context should be allocated and resets after confirmation.
    pub fn observe(&mut self, novel_evidence: bool) -> bool {
        if !novel_evidence {
            self.consecutive_novel_observations = 0;
            return false;
        }
        self.consecutive_novel_observations = self
            .consecutive_novel_observations
            .saturating_add(1)
            .min(self.required_consecutive_observations);
        if self.consecutive_novel_observations < self.required_consecutive_observations {
            return false;
        }
        self.consecutive_novel_observations = 0;
        true
    }

    pub fn reset(&mut self) {
        self.consecutive_novel_observations = 0;
    }
}

/// Checkpointable host-side context bank driven only by causal predictive
/// evidence.
///
/// `select` never mutates calibration. Callers explicitly `observe` a selected
/// context after deciding that the evidence belongs to it. This keeps holdout
/// evaluation and read-only routing from silently changing checkpoint state.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct PredictiveContextBank {
    config: PredictiveContextBankConfig,
    calibrations: Vec<PredictiveContextCalibration>,
}

impl PredictiveContextBank {
    pub fn new(config: PredictiveContextBankConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            calibrations: Vec::new(),
        })
    }

    pub fn config(&self) -> PredictiveContextBankConfig {
        self.config
    }

    pub fn known_contexts(&self) -> usize {
        self.calibrations.len()
    }

    pub fn calibrations(&self) -> &[PredictiveContextCalibration] {
        &self.calibrations
    }

    pub fn select(
        &self,
        losses: &[f64],
        allow_create: bool,
    ) -> Result<PredictiveContextSelection, String> {
        if losses.len() != self.calibrations.len() {
            return Err(format!(
                "predictive context loss count {} does not match known context count {}",
                losses.len(),
                self.calibrations.len()
            ));
        }
        if losses.iter().any(|loss| *loss < 0.0 || !loss.is_finite()) {
            return Err("predictive context losses must be finite and >= 0".to_string());
        }
        if self.calibrations.is_empty() {
            if !allow_create {
                return Err(
                    "predictive context bank is empty and context creation is disabled".to_string(),
                );
            }
            return Ok(PredictiveContextSelection {
                context_index: 0,
                created: true,
                capacity_exhausted: false,
                novel_evidence: true,
                candidates: Vec::new(),
            });
        }

        let candidates = losses
            .iter()
            .copied()
            .zip(&self.calibrations)
            .enumerate()
            .map(|(context_index, (loss, calibration))| {
                let mature = calibration.observations >= self.config.minimum_observations;
                let scale = calibration.loss_scale(self.config.minimum_loss_scale);
                let surprise = mature.then(|| (loss - calibration.mean_loss) / scale);
                let novelty_limit = calibration.mean_loss
                    + self
                        .config
                        .novelty_absolute_margin
                        .max(self.config.novelty_standard_deviations * scale);
                PredictiveContextCandidate {
                    context_index,
                    loss,
                    calibrated_surprise: surprise,
                    rejected_as_novel: mature && loss > novelty_limit,
                }
            })
            .collect::<Vec<_>>();
        let all_mature = self
            .calibrations
            .iter()
            .all(|calibration| calibration.observations >= self.config.minimum_observations);
        let all_rejected = candidates
            .iter()
            .all(|candidate| candidate.rejected_as_novel);
        if allow_create
            && all_mature
            && all_rejected
            && self.calibrations.len() < self.config.max_contexts
        {
            return Ok(PredictiveContextSelection {
                context_index: self.calibrations.len(),
                created: true,
                capacity_exhausted: false,
                novel_evidence: true,
                candidates,
            });
        }

        let selected = candidates
            .iter()
            // Expert routing is a likelihood decision, so compare causal
            // predictive losses directly. Per-expert calibration is used only
            // for novelty detection; z-scores are not comparable while an
            // expert is immature and can otherwise select a worse predictor.
            .min_by(|left, right| left.loss.total_cmp(&right.loss))
            .expect("non-empty predictive context candidates");
        Ok(PredictiveContextSelection {
            context_index: selected.context_index,
            created: false,
            capacity_exhausted: allow_create
                && all_mature
                && all_rejected
                && self.calibrations.len() >= self.config.max_contexts,
            novel_evidence: all_mature && all_rejected,
            candidates,
        })
    }

    pub fn create(&mut self) -> Result<usize, String> {
        if self.calibrations.len() >= self.config.max_contexts {
            return Err(format!(
                "predictive context capacity {} is exhausted",
                self.config.max_contexts
            ));
        }
        let context_index = self.calibrations.len();
        self.calibrations
            .push(PredictiveContextCalibration::default());
        Ok(context_index)
    }

    pub fn observe(&mut self, context_index: usize, loss: f64) -> Result<(), String> {
        if loss < 0.0 || !loss.is_finite() {
            return Err("predictive context observed loss must be finite and >= 0".to_string());
        }
        let calibration = self
            .calibrations
            .get_mut(context_index)
            .ok_or_else(|| format!("predictive context {context_index} does not exist"))?;
        calibration.observe(loss, self.config.calibration_update_rate);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mature(bank: &mut PredictiveContextBank, context: usize, loss: f64) {
        for offset in [-0.01, 0.0, 0.01, 0.0] {
            bank.observe(context, loss + offset)
                .expect("calibration observation");
        }
    }

    fn config() -> PredictiveContextBankConfig {
        PredictiveContextBankConfig {
            max_contexts: 2,
            minimum_observations: 4,
            calibration_update_rate: 0.25,
            novelty_standard_deviations: 3.0,
            novelty_absolute_margin: 0.2,
            minimum_loss_scale: 0.02,
        }
    }

    #[test]
    fn predictive_evidence_creates_selects_and_bounds_contexts() {
        let mut bank = PredictiveContextBank::new(config()).expect("valid bank");
        let first = bank.select(&[], true).expect("bootstrap selection");
        assert!(first.created);
        assert_eq!(bank.create().expect("first context"), 0);
        mature(&mut bank, 0, 0.1);

        let familiar = bank.select(&[0.11], true).expect("familiar selection");
        assert_eq!(familiar.context_index, 0);
        assert!(!familiar.created);

        let novel = bank.select(&[0.8], true).expect("novel selection");
        assert_eq!(novel.context_index, 1);
        assert!(novel.created);
        assert_eq!(bank.create().expect("second context"), 1);
        mature(&mut bank, 1, 0.12);

        let second = bank
            .select(&[0.7, 0.13], false)
            .expect("second-context selection");
        assert_eq!(second.context_index, 1);
        let exhausted = bank
            .select(&[0.9, 0.8], true)
            .expect("bounded novel selection");
        assert!(!exhausted.created);
        assert!(exhausted.capacity_exhausted);
    }

    #[test]
    fn selection_is_read_only_and_checkpoint_roundtrips() {
        let mut bank = PredictiveContextBank::new(config()).expect("valid bank");
        bank.create().expect("context");
        mature(&mut bank, 0, 0.2);
        let before = bank.clone();
        let _ = bank.select(&[0.21], false).expect("read-only selection");
        assert_eq!(bank, before);
        let encoded = serde_json::to_string(&bank).expect("serialize context bank");
        let decoded: PredictiveContextBank =
            serde_json::from_str(&encoded).expect("deserialize context bank");
        assert_eq!(decoded, bank);
    }

    #[test]
    fn immature_context_does_not_trigger_duplicate_creation() {
        let mut bank = PredictiveContextBank::new(config()).expect("valid bank");
        bank.create().expect("context");
        bank.observe(0, 0.1).expect("first observation");
        let selection = bank.select(&[1.0], true).expect("immature bank selection");
        assert_eq!(selection.context_index, 0);
        assert!(!selection.created);
    }

    #[test]
    fn routing_uses_absolute_predictive_loss_not_incomparable_surprise() {
        let mut bank = PredictiveContextBank::new(config()).expect("valid bank");
        bank.create().expect("first context");
        mature(&mut bank, 0, 0.1);
        bank.create().expect("second context");
        mature(&mut bank, 1, 2.0);

        let selection = bank
            .select(&[0.3, 1.9], false)
            .expect("predictive selection");
        assert_eq!(selection.context_index, 0);
    }

    #[test]
    fn novelty_gate_requires_consecutive_evidence_and_roundtrips() {
        let mut gate = PredictiveContextNoveltyGate::new(3).expect("valid gate");
        assert!(!gate.observe(true));
        assert!(!gate.observe(false));
        assert_eq!(gate.consecutive_novel_observations(), 0);
        assert!(!gate.observe(true));
        assert!(!gate.observe(true));

        let encoded = serde_json::to_string(&gate).expect("serialize novelty gate");
        let mut decoded: PredictiveContextNoveltyGate =
            serde_json::from_str(&encoded).expect("deserialize novelty gate");
        assert_eq!(decoded, gate);
        assert!(decoded.observe(true));
        assert_eq!(decoded.consecutive_novel_observations(), 0);
    }
}
