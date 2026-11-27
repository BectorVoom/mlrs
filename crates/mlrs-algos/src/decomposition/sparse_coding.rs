//! Greedy and least-angle sparse coding with a precomputed dictionary Gram.

use crate::linear::ridge_solvers::cholesky_ridge;

/// The stopping rule for a sparse coding path.
#[derive(Clone, Copy)]
pub enum Coding {
    Pursuit,
    LeastAngle,
    Lasso { alpha: f64, positive: bool },
}

/// Encode rows of `cross` with `width` dictionary atoms.
///
/// `gram` is a row-major symmetric matrix with `width * width` entries.
/// `cross` contains an integer number of rows of length `width`.
pub fn encode(gram: &[f64], cross: &[f64], width: usize, steps: usize, method: Coding) -> Vec<f64> {
    cross
        .chunks_exact(width)
        .flat_map(|row| match method {
            Coding::Pursuit => pursuit(gram, row, steps),
            Coding::LeastAngle => least_angle(gram, row, steps, 0.0, false, false),
            Coding::Lasso { alpha, positive } => {
                least_angle(gram, row, steps, alpha, true, positive)
            }
        })
        .collect()
}

fn solve_active(gram: &[f64], width: usize, active: &[usize], rhs: &[f64]) -> Option<Vec<f64>> {
    let subgram: Vec<_> = active
        .iter()
        .flat_map(|&i| active.iter().map(move |&j| gram[i * width + j]))
        .collect();
    // The ridge helper has an absolute pivot floor. Scale the subsystem so
    // that uniformly small but well-conditioned dictionaries remain usable.
    let scale = active
        .iter()
        .map(|&i| gram[i * width + i])
        .fold(0.0, f64::max);
    if scale <= 0.0 {
        return None;
    }
    let normalized: Vec<_> = subgram.into_iter().map(|value| value / scale).collect();
    let target: Vec<_> = rhs.iter().map(|value| value / scale).collect();
    cholesky_ridge(&normalized, &target, active.len(), 0.0)
}

fn correlations(gram: &[f64], target: &[f64], weights: &[f64]) -> Vec<f64> {
    gram.chunks_exact(weights.len())
        .zip(target)
        .map(|(row, value)| value - row.iter().zip(weights).map(|(a, b)| a * b).sum::<f64>())
        .collect()
}

fn best_atom(
    scores: &[f64],
    candidates: impl Iterator<Item = usize>,
    positive: bool,
) -> Option<usize> {
    let score = |i: usize| if positive { scores[i] } else { scores[i].abs() };
    candidates.reduce(|best, candidate| {
        if score(candidate) > score(best) {
            candidate
        } else {
            best
        }
    })
}

fn pursuit(gram: &[f64], target: &[f64], steps: usize) -> Vec<f64> {
    let width = target.len();
    let mut weights = vec![0.0; width];
    let mut active = Vec::with_capacity(steps.min(width));
    for _ in 0..steps.min(width) {
        let residual = correlations(gram, target, &weights);
        let j = best_atom(&residual, 0..width, false).expect("nonempty dictionary");
        if active.contains(&j) || residual[j].abs() <= f64::EPSILON {
            break;
        }
        active.push(j);
        let rhs: Vec<_> = active.iter().map(|&i| target[i]).collect();
        let Some(solution) = solve_active(gram, width, &active, &rhs) else {
            break;
        };
        for (&i, weight) in active.iter().zip(solution) {
            weights[i] = weight;
        }
    }
    weights
}

fn least_angle(
    gram: &[f64],
    target: &[f64],
    steps: usize,
    alpha: f64,
    lasso: bool,
    positive: bool,
) -> Vec<f64> {
    let width = target.len();
    let mut weights = vec![0.0; width];
    let mut active = Vec::new();
    let mut blocked = vec![false; width];
    let mut just_dropped = false;
    let tolerance = 8.0 * f64::EPSILON * target.iter().map(|v| v.abs()).fold(1.0, f64::max);

    for _ in 0..steps {
        let residual = correlations(gram, target, &weights);
        let mut remaining: Vec<_> = (0..width)
            .filter(|&i| !blocked[i] && !active.contains(&i))
            .collect();
        let new_atom = if just_dropped {
            None
        } else {
            best_atom(&residual, remaining.iter().copied(), positive)
        };
        if let Some(j) = new_atom {
            let score = if positive {
                residual[j]
            } else {
                residual[j].abs()
            };
            if score <= alpha + tolerance {
                break;
            }
            active.push(j);
            remaining.retain(|&i| i != j);
        }
        if active.is_empty() {
            break;
        }
        let level = active
            .iter()
            .map(|&i| residual[i].abs())
            .fold(0.0, f64::max);
        if level <= alpha + tolerance {
            break;
        }
        let signs: Vec<_> = active
            .iter()
            .map(|&i| if positive { 1.0 } else { residual[i].signum() })
            .collect();
        let Some(direction) = solve_active(gram, width, &active, &signs) else {
            if let Some(j) = new_atom {
                active.pop();
                blocked[j] = true;
                continue;
            }
            break;
        };
        let mut step = level - alpha;
        let signs_to_check: &[f64] = if positive { &[1.0] } else { &[1.0, -1.0] };
        for j in remaining {
            let slope: f64 = active
                .iter()
                .zip(&direction)
                .map(|(&i, v)| gram[j * width + i] * v)
                .sum();
            for &sign in signs_to_check {
                let denominator = 1.0 - sign * slope;
                if denominator > f64::EPSILON {
                    let crossing = (level - sign * residual[j]) / denominator;
                    if crossing > tolerance {
                        step = step.min(crossing);
                    }
                }
            }
        }
        let mut drop_index = None;
        if lasso {
            for (index, (&i, &velocity)) in active.iter().zip(&direction).enumerate() {
                if velocity.abs() > f64::EPSILON {
                    let crossing = -weights[i] / velocity;
                    if crossing > tolerance && crossing < step {
                        step = crossing;
                        drop_index = Some(index);
                    }
                }
            }
        }
        for (&i, velocity) in active.iter().zip(direction) {
            weights[i] += step * velocity;
        }
        just_dropped = drop_index.is_some();
        if let Some(index) = drop_index {
            weights[active.remove(index)] = 0.0;
        } else if step >= level - alpha {
            break;
        }
    }
    weights
}
