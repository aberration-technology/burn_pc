use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// Timing of parameter learning relative to activity inference.
#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PcLearningSchedule {
    /// Settle activities, then compute one local parameter update.
    #[default]
    Equilibrium,
    /// Compute local parameter updates after every activity update.
    Incremental,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub enum PcSchedulePhase {
    Initialize,
    Infer { step: usize },
    UpdateParameters { inference_step: usize },
    Complete,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct PcSchedulePlan {
    pub inference_steps: usize,
    pub learning: PcLearningSchedule,
}

impl PcSchedulePlan {
    pub fn validate(&self) -> Result<()> {
        if self.inference_steps == 0 {
            return Err(anyhow!(
                "predictive-coding schedule inference_steps must be > 0"
            ));
        }
        Ok(())
    }

    pub fn phases(&self) -> Vec<PcSchedulePhase> {
        self.validate()
            .expect("invalid predictive-coding schedule plan");
        let mut phases = Vec::with_capacity(match self.learning {
            PcLearningSchedule::Equilibrium => self.inference_steps + 3,
            PcLearningSchedule::Incremental => self.inference_steps * 2 + 2,
        });
        phases.push(PcSchedulePhase::Initialize);
        for step in 0..self.inference_steps {
            phases.push(PcSchedulePhase::Infer { step });
            if matches!(self.learning, PcLearningSchedule::Incremental) {
                phases.push(PcSchedulePhase::UpdateParameters {
                    inference_step: step,
                });
            }
        }
        if matches!(self.learning, PcLearningSchedule::Equilibrium) {
            phases.push(PcSchedulePhase::UpdateParameters {
                inference_step: self.inference_steps - 1,
            });
        }
        phases.push(PcSchedulePhase::Complete);
        phases
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equilibrium_updates_once_after_inference() {
        let phases = PcSchedulePlan {
            inference_steps: 3,
            learning: PcLearningSchedule::Equilibrium,
        }
        .phases();
        assert_eq!(
            phases,
            vec![
                PcSchedulePhase::Initialize,
                PcSchedulePhase::Infer { step: 0 },
                PcSchedulePhase::Infer { step: 1 },
                PcSchedulePhase::Infer { step: 2 },
                PcSchedulePhase::UpdateParameters { inference_step: 2 },
                PcSchedulePhase::Complete,
            ]
        );
    }

    #[test]
    fn incremental_updates_at_each_inference_step() {
        let phases = PcSchedulePlan {
            inference_steps: 2,
            learning: PcLearningSchedule::Incremental,
        }
        .phases();
        assert_eq!(
            phases,
            vec![
                PcSchedulePhase::Initialize,
                PcSchedulePhase::Infer { step: 0 },
                PcSchedulePhase::UpdateParameters { inference_step: 0 },
                PcSchedulePhase::Infer { step: 1 },
                PcSchedulePhase::UpdateParameters { inference_step: 1 },
                PcSchedulePhase::Complete,
            ]
        );
    }

    #[test]
    fn empty_schedule_fails_validation() {
        let plan = PcSchedulePlan {
            inference_steps: 0,
            learning: PcLearningSchedule::Equilibrium,
        };
        assert!(plan.validate().is_err());
    }
}
