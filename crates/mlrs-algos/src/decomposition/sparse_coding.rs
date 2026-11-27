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
    let (alpha, lasso, positive) = match method {
        Coding::Pursuit => return pursuit(gram, cross, width, steps.min(width)),
        Coding::LeastAngle => (0.0, false, false),
        Coding::Lasso { alpha, positive } => (alpha, true, positive),
    };
    cross
        .chunks_exact(width)
        .flat_map(|row| least_angle(gram, row, steps, alpha, lasso, positive))
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

fn subtract_atom(residual: &mut [f64], gram_row: &[f64], coefficient: f64) {
    for (value, &entry) in residual.iter_mut().zip(gram_row) {
        *value -= coefficient * entry;
    }
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

fn pursuit(gram: &[f64], cross: &[f64], width: usize, steps: usize) -> Vec<f64> {
    let mut codes = vec![0.0; cross.len()];
    // Reuse the factor and work vectors across signals. Each new atom extends
    // the Cholesky factor by one row and adds one forward-substitution term.
    let mut factor = vec![0.0; steps * steps];
    let mut rhs = vec![0.0; steps];
    let mut solution = vec![0.0; steps];
    let mut residual = vec![0.0; width];
    let mut active = Vec::with_capacity(steps);
    for (target, weights) in cross.chunks_exact(width).zip(codes.chunks_exact_mut(width)) {
        residual.copy_from_slice(target);
        active.clear();
        let mut scale = 0.0_f64;
        let mut min_pivot = f64::INFINITY;
        for i in 0..steps {
            let j = best_atom(&residual, 0..width, false).expect("nonempty dictionary");
            if active.contains(&j) || residual[j].abs() <= f64::EPSILON {
                break;
            }
            let gram_row = &gram[j * width..(j + 1) * width];
            let (previous, current) = factor.split_at_mut(i * steps);
            let current = &mut current[..=i];
            let mut pivot = gram_row[j];
            for (k, &atom) in active.iter().enumerate() {
                let lower = &previous[k * steps..k * steps + k];
                let mut value = gram_row[atom];
                for (&a, &b) in current[..k].iter().zip(lower) {
                    value -= a * b;
                }
                value /= previous[k * steps + k];
                current[k] = value;
                pivot -= value * value;
            }
            scale = scale.max(gram_row[j]);
            // Match the relative pivot floor of the original active-set solve.
            if !(pivot > 1e-12 * scale && min_pivot > 1e-12 * scale) {
                break;
            }
            min_pivot = min_pivot.min(pivot);
            let diagonal = pivot.sqrt();
            current[i] = diagonal;
            let mut value = target[j];
            for (&a, &b) in current[..i].iter().zip(&rhs) {
                value -= a * b;
            }
            rhs[i] = value / diagonal;
            active.push(j);

            solution[..=i].copy_from_slice(&rhs[..=i]);
            for k in (0..=i).rev() {
                let lower = &factor[k * steps..k * steps + k + 1];
                let coefficient = solution[k] / lower[k];
                weights[active[k]] = coefficient;
                subtract_atom(&mut solution[..k], &lower[..k], coefficient);
            }
            if i + 1 < steps {
                residual.copy_from_slice(target);
                for &atom in &active {
                    subtract_atom(
                        &mut residual,
                        &gram[atom * width..(atom + 1) * width],
                        weights[atom],
                    );
                }
            }
        }
    }
    codes
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
    let mut residual = vec![0.0; width];
    let mut just_dropped = false;
    let tolerance = 8.0 * f64::EPSILON * target.iter().map(|v| v.abs()).fold(1.0, f64::max);

    for _ in 0..steps {
        residual.copy_from_slice(target);
        for &atom in &active {
            subtract_atom(
                &mut residual,
                &gram[atom * width..(atom + 1) * width],
                weights[atom],
            );
        }
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
