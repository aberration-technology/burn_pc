use serde::{Deserialize, Serialize};

/// Policy applied when novel predictive evidence arrives at a full context bank.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PredictiveContextCapacityPolicy {
    /// Keep every existing context and route to the lowest-loss expert.
    #[default]
    Reject,
    /// Reuse the least-recently-observed slot. Slot generations prevent stale
    /// optimizer, mask, checkpoint, or network updates from targeting its new owner.
    ReplaceLeastRecentlyUsed,
}

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
    pub capacity_policy: PredictiveContextCapacityPolicy,
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
            capacity_policy: PredictiveContextCapacityPolicy::Reject,
        }
    }
}

/// Stable identity for one bounded context slot.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PredictiveContextIdentity {
    pub context_index: usize,
    pub generation: u64,
}

/// Checkpointed lifecycle metadata for one context slot.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct PredictiveContextLifecycle {
    pub generation: u64,
    pub created_at_observation: u64,
    pub last_observed_at: u64,
}

impl PredictiveContextLifecycle {
    pub fn identity(self, context_index: usize) -> PredictiveContextIdentity {
        PredictiveContextIdentity {
            context_index,
            generation: self.generation,
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
    /// Existing slot that must be archived/reinitialized before this creation
    /// can be committed. Present only for bounded replacement.
    pub replacement: Option<PredictiveContextIdentity>,
    /// Every mature expert rejected the observation under its calibrated
    /// predictive-loss envelope. Callers may feed this into a sequential
    /// novelty gate before allocating a context.
    pub novel_evidence: bool,
    pub candidates: Vec<PredictiveContextCandidate>,
}

/// Result of committing a context creation decision.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct PredictiveContextAllocation {
    pub identity: PredictiveContextIdentity,
    pub replaced: Option<PredictiveContextIdentity>,
}

/// Symmetric evidence required before two predictive contexts may share one slot.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct PredictiveContextMergeEvidence {
    pub left_reference_loss: f64,
    pub right_reference_loss: f64,
    pub left_model_on_right_loss: f64,
    pub right_model_on_left_loss: f64,
}

/// Model-agnostic acceptance envelope for context merging.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct PredictiveContextMergeConfig {
    pub maximum_absolute_loss_increase: f64,
    pub maximum_relative_loss_increase: f64,
}

impl Default for PredictiveContextMergeConfig {
    fn default() -> Self {
        Self {
            maximum_absolute_loss_increase: 0.01,
            maximum_relative_loss_increase: 0.05,
        }
    }
}

impl PredictiveContextMergeConfig {
    pub fn validate(self) -> Result<(), String> {
        for (name, value) in [
            (
                "maximum_absolute_loss_increase",
                self.maximum_absolute_loss_increase,
            ),
            (
                "maximum_relative_loss_increase",
                self.maximum_relative_loss_increase,
            ),
        ] {
            if value < 0.0 || !value.is_finite() {
                return Err(format!(
                    "predictive context merge {name} must be finite and >= 0"
                ));
            }
        }
        Ok(())
    }

    pub fn accepts(self, evidence: PredictiveContextMergeEvidence) -> Result<bool, String> {
        self.validate()?;
        let losses = [
            evidence.left_reference_loss,
            evidence.right_reference_loss,
            evidence.left_model_on_right_loss,
            evidence.right_model_on_left_loss,
        ];
        if losses.iter().any(|loss| *loss < 0.0 || !loss.is_finite()) {
            return Err("predictive context merge losses must be finite and >= 0".to_string());
        }
        let accepts_direction = |reference: f64, cross: f64| {
            let allowance = self
                .maximum_absolute_loss_increase
                .max(reference * self.maximum_relative_loss_increase);
            cross <= reference + allowance
        };
        Ok(accepts_direction(
            evidence.right_reference_loss,
            evidence.left_model_on_right_loss,
        ) && accepts_direction(
            evidence.left_reference_loss,
            evidence.right_model_on_left_loss,
        ))
    }
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
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PredictiveContextBank {
    config: PredictiveContextBankConfig,
    calibrations: Vec<PredictiveContextCalibration>,
    #[serde(default)]
    lifecycles: Vec<PredictiveContextLifecycle>,
    #[serde(default)]
    observation_clock: u64,
}

#[derive(Deserialize)]
struct PredictiveContextBankRecord {
    config: PredictiveContextBankConfig,
    calibrations: Vec<PredictiveContextCalibration>,
    #[serde(default)]
    lifecycles: Vec<PredictiveContextLifecycle>,
    #[serde(default)]
    observation_clock: u64,
}

impl<'de> Deserialize<'de> for PredictiveContextBank {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let mut record = PredictiveContextBankRecord::deserialize(deserializer)?;
        record.config.validate().map_err(serde::de::Error::custom)?;
        if record.calibrations.len() > record.config.max_contexts {
            return Err(serde::de::Error::custom(
                "predictive context checkpoint exceeds configured capacity",
            ));
        }
        if record.lifecycles.len() > record.calibrations.len() {
            return Err(serde::de::Error::custom(
                "predictive context checkpoint has excess lifecycle records",
            ));
        }
        record
            .lifecycles
            .resize_with(record.calibrations.len(), || PredictiveContextLifecycle {
                generation: 0,
                created_at_observation: 0,
                last_observed_at: 0,
            });
        Ok(Self {
            config: record.config,
            calibrations: record.calibrations,
            lifecycles: record.lifecycles,
            observation_clock: record.observation_clock,
        })
    }
}

impl PredictiveContextBank {
    pub fn new(config: PredictiveContextBankConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self {
            config,
            calibrations: Vec::new(),
            lifecycles: Vec::new(),
            observation_clock: 0,
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

    pub fn lifecycles(&self) -> &[PredictiveContextLifecycle] {
        &self.lifecycles
    }

    pub fn identity(&self, context_index: usize) -> Option<PredictiveContextIdentity> {
        self.lifecycles
            .get(context_index)
            .copied()
            .map(|lifecycle| lifecycle.identity(context_index))
    }

    fn replacement_candidate(&self) -> Option<PredictiveContextIdentity> {
        self.lifecycles
            .iter()
            .copied()
            .enumerate()
            .min_by_key(|(index, lifecycle)| (lifecycle.last_observed_at, *index))
            .map(|(index, lifecycle)| lifecycle.identity(index))
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
                replacement: None,
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
                replacement: None,
                novel_evidence: true,
                candidates,
            });
        }

        let replacement = (allow_create
            && all_mature
            && all_rejected
            && self.calibrations.len() >= self.config.max_contexts
            && matches!(
                self.config.capacity_policy,
                PredictiveContextCapacityPolicy::ReplaceLeastRecentlyUsed
            ))
        .then(|| self.replacement_candidate())
        .flatten();
        if let Some(replacement) = replacement {
            return Ok(PredictiveContextSelection {
                context_index: replacement.context_index,
                created: true,
                capacity_exhausted: false,
                replacement: Some(replacement),
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
            replacement: None,
            novel_evidence: all_mature && all_rejected,
            candidates,
        })
    }

    pub fn create(&mut self) -> Result<usize, String> {
        let selection = if self.calibrations.len() < self.config.max_contexts {
            PredictiveContextSelection {
                context_index: self.calibrations.len(),
                created: true,
                capacity_exhausted: false,
                replacement: None,
                novel_evidence: true,
                candidates: Vec::new(),
            }
        } else if matches!(
            self.config.capacity_policy,
            PredictiveContextCapacityPolicy::ReplaceLeastRecentlyUsed
        ) {
            let replacement = self
                .replacement_candidate()
                .expect("full predictive context bank has a replacement candidate");
            PredictiveContextSelection {
                context_index: replacement.context_index,
                created: true,
                capacity_exhausted: false,
                replacement: Some(replacement),
                novel_evidence: true,
                candidates: Vec::new(),
            }
        } else {
            return Err(format!(
                "predictive context capacity {} is exhausted",
                self.config.max_contexts
            ));
        };
        Ok(self.allocate_selected(&selection)?.identity.context_index)
    }

    /// Commit a read-only selection after downstream code has archived any
    /// context-owned state named by `selection.replacement`.
    pub fn allocate_selected(
        &mut self,
        selection: &PredictiveContextSelection,
    ) -> Result<PredictiveContextAllocation, String> {
        if !selection.created {
            return Err("predictive context selection is not an allocation".to_string());
        }
        self.observation_clock = self.observation_clock.saturating_add(1);
        let context_index = selection.context_index;
        let replaced = selection.replacement;
        if context_index == self.calibrations.len() {
            if replaced.is_some() || self.calibrations.len() >= self.config.max_contexts {
                return Err("predictive context append allocation is inconsistent".to_string());
            }
            let lifecycle = PredictiveContextLifecycle {
                generation: 0,
                created_at_observation: self.observation_clock,
                last_observed_at: self.observation_clock,
            };
            self.calibrations
                .push(PredictiveContextCalibration::default());
            self.lifecycles.push(lifecycle);
            return Ok(PredictiveContextAllocation {
                identity: lifecycle.identity(context_index),
                replaced: None,
            });
        }
        let current = self
            .identity(context_index)
            .ok_or_else(|| format!("predictive context slot {context_index} does not exist"))?;
        if replaced != Some(current) {
            return Err(format!(
                "predictive context replacement is stale: selected {replaced:?}, current {current:?}"
            ));
        }
        let lifecycle = PredictiveContextLifecycle {
            generation: current.generation.saturating_add(1),
            created_at_observation: self.observation_clock,
            last_observed_at: self.observation_clock,
        };
        self.calibrations[context_index] = PredictiveContextCalibration::default();
        self.lifecycles[context_index] = lifecycle;
        Ok(PredictiveContextAllocation {
            identity: lifecycle.identity(context_index),
            replaced,
        })
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
        self.observation_clock = self.observation_clock.saturating_add(1);
        let lifecycle = self.lifecycles.get_mut(context_index).ok_or_else(|| {
            format!("predictive context lifecycle {context_index} does not exist")
        })?;
        lifecycle.last_observed_at = self.observation_clock;
        Ok(())
    }

    /// Record that a specific context generation was used without treating the
    /// current observation as calibration evidence for that context. This is
    /// useful while a sequential novelty gate is confirming a distribution
    /// shift: the fallback expert remains recently used, but its likelihood
    /// envelope is not widened by known out-of-context samples.
    pub fn touch(&mut self, identity: PredictiveContextIdentity) -> Result<(), String> {
        if self.identity(identity.context_index) != Some(identity) {
            return Err(format!(
                "predictive context touch is stale: requested {identity:?}, current {:?}",
                self.identity(identity.context_index)
            ));
        }
        self.observation_clock = self.observation_clock.saturating_add(1);
        self.lifecycles[identity.context_index].last_observed_at = self.observation_clock;
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
            capacity_policy: PredictiveContextCapacityPolicy::Reject,
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
    fn touch_advances_lru_without_changing_calibration_and_rejects_stale_identity() {
        let mut bank = PredictiveContextBank::new(config()).expect("valid bank");
        let first = bank.select(&[], true).expect("bootstrap selection");
        let allocation = bank.allocate_selected(&first).expect("first allocation");
        mature(&mut bank, 0, 0.1);
        let calibration = bank.calibrations()[0];
        let previous_access = bank.lifecycles()[0].last_observed_at;

        bank.touch(allocation.identity)
            .expect("touch current context");
        assert_eq!(bank.calibrations()[0], calibration);
        assert!(bank.lifecycles()[0].last_observed_at > previous_access);

        let stale = PredictiveContextIdentity {
            context_index: 0,
            generation: allocation.identity.generation.saturating_add(1),
        };
        assert!(bank.touch(stale).is_err());
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

    #[test]
    fn lru_replacement_increments_generation_and_rejects_stale_commit() {
        let mut config = config();
        config.capacity_policy = PredictiveContextCapacityPolicy::ReplaceLeastRecentlyUsed;
        let mut bank = PredictiveContextBank::new(config).expect("valid bank");
        assert_eq!(bank.create().expect("first context"), 0);
        mature(&mut bank, 0, 0.1);
        assert_eq!(bank.create().expect("second context"), 1);
        mature(&mut bank, 1, 0.1);
        bank.observe(1, 0.1).expect("refresh second context");

        let selection = bank
            .select(&[0.9, 0.8], true)
            .expect("replacement selection");
        assert!(selection.created);
        assert_eq!(selection.context_index, 0);
        assert_eq!(selection.replacement, bank.identity(0));
        let stale = selection.clone();
        let allocation = bank
            .allocate_selected(&selection)
            .expect("replacement allocation");
        assert_eq!(allocation.identity.context_index, 0);
        assert_eq!(allocation.identity.generation, 1);
        assert!(bank.allocate_selected(&stale).is_err());
    }

    #[test]
    fn merge_gate_requires_bidirectional_parity() {
        let config = PredictiveContextMergeConfig {
            maximum_absolute_loss_increase: 0.02,
            maximum_relative_loss_increase: 0.1,
        };
        assert!(
            config
                .accepts(PredictiveContextMergeEvidence {
                    left_reference_loss: 0.2,
                    right_reference_loss: 0.3,
                    left_model_on_right_loss: 0.32,
                    right_model_on_left_loss: 0.21,
                })
                .expect("valid merge evidence")
        );
        assert!(
            !config
                .accepts(PredictiveContextMergeEvidence {
                    left_reference_loss: 0.2,
                    right_reference_loss: 0.3,
                    left_model_on_right_loss: 0.31,
                    right_model_on_left_loss: 0.5,
                })
                .expect("valid asymmetric evidence")
        );
    }
}
