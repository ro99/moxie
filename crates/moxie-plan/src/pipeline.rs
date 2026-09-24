use std::collections::BTreeMap;
use std::ops::Range;

use moxie_graph::{Graph, ValueId, ValueRole};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineLowering {
    stages: Vec<Range<usize>>,
    handoffs: Vec<ValueId>,
}

impl PipelineLowering {
    pub fn stages(&self) -> &[Range<usize>] {
        &self.stages
    }

    pub fn handoffs(&self) -> &[ValueId] {
        &self.handoffs
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineRefused {
    /// Cuts must be nonempty, strictly increasing, and inside the node range.
    Cuts,
    /// A value crossing a stage boundary is not a float activation.
    NonActivation { value: ValueId },
    /// A boundary must carry only the previous stage's final activation.
    Handoffs {
        boundary: usize,
        values: Vec<ValueId>,
    },
}

impl core::fmt::Display for PipelineRefused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Cuts => write!(f, "pipeline cuts must be nonempty, sorted, and interior"),
            Self::NonActivation { value } => {
                write!(f, "pipeline boundary value {value:?} is not an activation")
            }
            Self::Handoffs { boundary, values } => write!(
                f,
                "pipeline boundary {boundary} must carry its final activation only; found {values:?}"
            ),
        }
    }
}

impl std::error::Error for PipelineRefused {}

pub fn lower_pipeline(graph: &Graph, cuts: &[usize]) -> Result<PipelineLowering, PipelineRefused> {
    let node_count = graph.nodes().len();
    if cuts.is_empty()
        || cuts.iter().any(|cut| *cut == 0 || *cut >= node_count)
        || cuts.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err(PipelineRefused::Cuts);
    }

    let mut producers = BTreeMap::new();
    let mut consumers: BTreeMap<ValueId, Vec<usize>> = BTreeMap::new();
    for (index, node) in graph.nodes().iter().enumerate() {
        producers.insert(node.output, index);
        for input in &node.inputs {
            consumers.entry(*input).or_default().push(index);
        }
    }

    let mut stages = Vec::with_capacity(cuts.len() + 1);
    let mut start = 0;
    for &end in cuts {
        stages.push(start..end);
        start = end;
    }
    stages.push(start..node_count);

    let mut handoffs = Vec::with_capacity(cuts.len());
    for (boundary, &cut) in cuts.iter().enumerate() {
        let crossing: Vec<_> = producers
            .iter()
            .filter_map(|(value, producer)| {
                if *producer >= cut
                    || graph.inputs().contains(value)
                    || graph.weights().contains(value)
                {
                    return None;
                }
                let consumed_after = consumers
                    .get(value)
                    .is_some_and(|uses| uses.iter().any(|consumer| *consumer >= cut));
                (consumed_after || *value == graph.output()).then_some(*value)
            })
            .collect();

        for &value in &crossing {
            if !matches!(
                graph.spec(value).map(|spec| spec.role),
                Some(ValueRole::Activation(_))
            ) {
                return Err(PipelineRefused::NonActivation { value });
            }
        }

        let expected = graph.nodes()[cut - 1].output;
        if crossing.len() != 1 || crossing[0] != expected {
            return Err(PipelineRefused::Handoffs {
                boundary,
                values: crossing,
            });
        }
        handoffs.push(expected);
    }

    Ok(PipelineLowering { stages, handoffs })
}

pub fn wavefront(stages: usize, microbatches: usize) -> Vec<(usize, usize)> {
    if stages == 0 || microbatches == 0 {
        return Vec::new();
    }

    let mut schedule = Vec::new();
    for wave in 0..stages + microbatches - 1 {
        for stage in 0..stages {
            if wave >= stage {
                let microbatch = wave - stage;
                if microbatch < microbatches {
                    schedule.push((stage, microbatch));
                }
            }
        }
    }
    schedule
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::wavefront;

    #[test]
    fn wavefront_orders_every_stage_and_microbatch_once() {
        for stages in 1..=4 {
            for microbatches in 1..=4 {
                let schedule = wavefront(stages, microbatches);
                assert_eq!(schedule.len(), stages * microbatches);
                let positions: BTreeMap<_, _> = schedule
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(index, pair)| (pair, index))
                    .collect();
                assert_eq!(positions.len(), stages * microbatches);
                for stage in 0..stages {
                    for microbatch in 0..microbatches {
                        let position = positions[&(stage, microbatch)];
                        if stage > 0 {
                            assert!(position > positions[&(stage - 1, microbatch)]);
                        }
                        if microbatch > 0 {
                            assert!(position > positions[&(stage, microbatch - 1)]);
                        }
                    }
                }
            }
        }
    }
}
