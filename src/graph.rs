use std::collections::BTreeSet;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

/// Stable identifier for an activity node in a predictive-coding graph.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PcNodeId(pub u32);

/// Stable identifier for a local prediction factor.
#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PcFactorId(pub u32);

/// Static metadata for one inferred or clamped activity.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct PcNodeSpec {
    pub id: PcNodeId,
    pub name: String,
    pub clamped: bool,
}

/// Static metadata for one directed local prediction.
///
/// The factor predicts `target` from the activities in `parents`. Cyclic graphs
/// are valid; inference scheduling, rather than graph topology, resolves them.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct PcFactorSpec {
    pub id: PcFactorId,
    pub name: String,
    pub parents: Vec<PcNodeId>,
    pub target: PcNodeId,
}

/// Backend-independent predictive-coding graph description.
#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct PcGraphSpec {
    pub version: u32,
    pub nodes: Vec<PcNodeSpec>,
    pub factors: Vec<PcFactorSpec>,
}

impl PcGraphSpec {
    pub const CURRENT_VERSION: u32 = 1;

    pub fn new(nodes: Vec<PcNodeSpec>, factors: Vec<PcFactorSpec>) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            nodes,
            factors,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.version != Self::CURRENT_VERSION {
            return Err(anyhow!(
                "unsupported predictive-coding graph version {}; expected {}",
                self.version,
                Self::CURRENT_VERSION
            ));
        }
        if self.nodes.is_empty() {
            return Err(anyhow!("predictive-coding graph must contain a node"));
        }
        if self.factors.is_empty() {
            return Err(anyhow!("predictive-coding graph must contain a factor"));
        }

        let mut node_ids = BTreeSet::new();
        let mut node_names = BTreeSet::new();
        for node in &self.nodes {
            if !node_ids.insert(node.id) {
                return Err(anyhow!("duplicate predictive-coding node id {}", node.id.0));
            }
            if node.name.trim().is_empty() || !node_names.insert(node.name.as_str()) {
                return Err(anyhow!(
                    "predictive-coding node names must be non-empty and unique: {:?}",
                    node.name
                ));
            }
        }

        let mut factor_ids = BTreeSet::new();
        let mut factor_names = BTreeSet::new();
        for factor in &self.factors {
            if !factor_ids.insert(factor.id) {
                return Err(anyhow!(
                    "duplicate predictive-coding factor id {}",
                    factor.id.0
                ));
            }
            if factor.name.trim().is_empty() || !factor_names.insert(factor.name.as_str()) {
                return Err(anyhow!(
                    "predictive-coding factor names must be non-empty and unique: {:?}",
                    factor.name
                ));
            }
            if factor.parents.is_empty() {
                return Err(anyhow!(
                    "predictive-coding factor {:?} must have a parent",
                    factor.name
                ));
            }
            if !node_ids.contains(&factor.target) {
                return Err(anyhow!(
                    "predictive-coding factor {:?} references missing target {}",
                    factor.name,
                    factor.target.0
                ));
            }
            let mut parents = BTreeSet::new();
            for parent in &factor.parents {
                if !node_ids.contains(parent) {
                    return Err(anyhow!(
                        "predictive-coding factor {:?} references missing parent {}",
                        factor.name,
                        parent.0
                    ));
                }
                if !parents.insert(*parent) {
                    return Err(anyhow!(
                        "predictive-coding factor {:?} repeats parent {}",
                        factor.name,
                        parent.0
                    ));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_graph() -> PcGraphSpec {
        PcGraphSpec::new(
            vec![
                PcNodeSpec {
                    id: PcNodeId(0),
                    name: "input".into(),
                    clamped: true,
                },
                PcNodeSpec {
                    id: PcNodeId(1),
                    name: "hidden".into(),
                    clamped: false,
                },
            ],
            vec![PcFactorSpec {
                id: PcFactorId(0),
                name: "transition".into(),
                parents: vec![PcNodeId(0)],
                target: PcNodeId(1),
            }],
        )
    }

    #[test]
    fn graph_accepts_well_formed_metadata() {
        valid_graph().validate().expect("valid graph");
    }

    #[test]
    fn graph_rejects_unknown_endpoint() {
        let mut graph = valid_graph();
        graph.factors[0].target = PcNodeId(7);
        assert!(graph.validate().is_err());
    }

    #[test]
    fn graph_allows_cycles() {
        let mut graph = valid_graph();
        graph.factors.push(PcFactorSpec {
            id: PcFactorId(1),
            name: "feedback".into(),
            parents: vec![PcNodeId(1)],
            target: PcNodeId(0),
        });
        graph.validate().expect("cyclic graph is valid");
    }
}
