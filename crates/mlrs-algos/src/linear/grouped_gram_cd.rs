//! Coordinate descent with one L2 group per feature, using normalized Gram statistics.

pub struct GramProblem<'a> {
    pub gram: &'a [f64],
    pub cross: &'a [f64],
    pub features: usize,
    pub targets: usize,
    pub target_energy: f64,
}

pub struct Options {
    pub ratio: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub positive: bool,
}

pub struct Path {
    pub coefficients: Vec<f64>,
    pub gaps: Vec<f64>,
    pub iterations: Vec<usize>,
    pub converged: Vec<bool>,
}

/// Fit a descending sequence of alphas, warm-starting each fit from the previous one.
pub fn solve_path(
    problem: GramProblem<'_>,
    mut coef: Vec<f64>,
    alphas: &[f64],
    options: Options,
) -> Path {
    let mut path = Path {
        coefficients: Vec::with_capacity(coef.len() * alphas.len()),
        gaps: Vec::with_capacity(alphas.len()),
        iterations: Vec::with_capacity(alphas.len()),
        converged: Vec::with_capacity(alphas.len()),
    };
    let k = problem.targets;
    let d = problem.features;
    for (j, block) in coef.chunks_exact_mut(k).enumerate() {
        if problem.gram[j * (d + 1)] <= 0.0 {
            block.fill(0.0);
        }
    }
    let mut cross_residual = problem.cross.to_vec();
    for (j, weights) in coef.chunks_exact(k).enumerate() {
        if weights.iter().any(|&coefficient| coefficient != 0.0) {
            shift(
                &mut cross_residual,
                &problem.gram[j * d..(j + 1) * d],
                weights,
            );
        }
    }
    let mut delta = vec![0.0; k];
    let mut inverse = vec![0.0; d];
    for &alpha in alphas {
        let l1 = alpha * options.ratio;
        let l2 = alpha * (1.0 - options.ratio);
        for (j, value) in inverse.iter_mut().enumerate() {
            let diagonal = problem.gram[j * (d + 1)];
            *value = if diagonal > 0.0 {
                1.0 / (diagonal + l2)
            } else {
                0.0
            };
        }
        let mut gap = f64::INFINITY;
        let mut converged = false;
        let mut iteration = 0;
        for sweep in 1..=options.max_iter {
            iteration = sweep;
            let (max_update, max_coef) = if k == 1 {
                sweep_scalar(
                    problem.gram,
                    &mut coef,
                    &mut cross_residual,
                    &inverse,
                    l1,
                    options.positive,
                )
            } else {
                sweep_group(
                    problem.gram,
                    &mut coef,
                    &mut cross_residual,
                    &inverse,
                    l1,
                    options.positive,
                    &mut delta,
                )
            };
            if sweep == options.max_iter || max_update <= options.tol * max_coef {
                (gap, converged) = dual_gap(&problem, &coef, &cross_residual, l1, l2, &options);
                if converged {
                    break;
                }
            }
        }
        path.coefficients.extend_from_slice(&coef);
        path.gaps.push(gap);
        path.iterations.push(iteration);
        path.converged.push(converged);
    }
    path
}

fn sweep_scalar(
    gram: &[f64],
    coef: &mut [f64],
    residual: &mut [f64],
    inverse: &[f64],
    l1: f64,
    positive: bool,
) -> (f64, f64) {
    let d = coef.len();
    let mut max_update: f64 = 0.0;
    let mut max_coef: f64 = 0.0;
    for (j, ((old, &inv), column)) in coef
        .iter_mut()
        .zip(inverse)
        .zip(gram.chunks_exact(d))
        .enumerate()
    {
        if inv == 0.0 {
            continue;
        }
        let target = residual[j] + column[j] * *old;
        let next = if positive {
            (target - l1).max(0.0) * inv
        } else {
            (target.abs() - l1).max(0.0).copysign(target) * inv
        };
        let delta = next - *old;
        if delta != 0.0 {
            *old = next;
            for (r, &entry) in residual.iter_mut().zip(column) {
                *r -= delta * entry;
            }
            max_update = max_update.max(delta.abs());
        }
        max_coef = max_coef.max(next.abs());
    }
    (max_update, max_coef)
}

fn sweep_group(
    gram: &[f64],
    coef: &mut [f64],
    cross_residual: &mut [f64],
    inverse: &[f64],
    l1: f64,
    positive: bool,
    delta: &mut [f64],
) -> (f64, f64) {
    let k = delta.len();
    let d = inverse.len();
    let mut max_update: f64 = 0.0;
    let mut max_coef: f64 = 0.0;
    for (j, (weights, &inv)) in coef.chunks_exact_mut(k).zip(inverse).enumerate() {
        let column = &gram[j * d..(j + 1) * d];
        let diagonal = column[j];
        if inv == 0.0 {
            continue;
        }
        let corr = &cross_residual[j * k..(j + 1) * k];
        let mut norm2 = 0.0;
        for ((tmp, &old), &value) in delta.iter_mut().zip(weights.iter()).zip(corr) {
            let value = value + diagonal * old;
            *tmp = if positive { value.max(0.0) } else { value };
            norm2 += *tmp * *tmp;
        }
        let norm = norm2.sqrt();
        let shrink = if norm > l1 {
            (1.0 - l1 / norm) * inv
        } else {
            0.0
        };
        let mut update: f64 = 0.0;
        for (diff, old) in delta.iter_mut().zip(weights.iter_mut()) {
            let new = *diff * shrink;
            *diff = new - *old;
            *old = new;
            update = update.max(diff.abs());
            max_coef = max_coef.max(new.abs());
        }
        if update > 0.0 {
            shift(cross_residual, column, delta);
            max_update = max_update.max(update);
        }
    }
    (max_update, max_coef)
}

fn shift(residual: &mut [f64], column: &[f64], delta: &[f64]) {
    if let [change] = delta {
        for (r, &g) in residual.iter_mut().zip(column) {
            *r -= g * change;
        }
    } else {
        for (r, &g) in residual.chunks_exact_mut(delta.len()).zip(column) {
            for (value, &change) in r.iter_mut().zip(delta) {
                *value -= g * change;
            }
        }
    }
}

fn dual_gap(
    problem: &GramProblem<'_>,
    coef: &[f64],
    residual: &[f64],
    l1: f64,
    l2: f64,
    options: &Options,
) -> (f64, bool) {
    let mut dual_norm2: f64 = 0.0;
    let mut l21 = 0.0;
    let mut norm2 = 0.0;
    let mut ridge_gap = 0.0;
    for (w, r) in coef
        .chunks_exact(problem.targets)
        .zip(residual.chunks_exact(problem.targets))
    {
        let mut group_norm2 = 0.0;
        let mut dual_group_norm2 = 0.0;
        for (&weight, &corr) in w.iter().zip(r) {
            let gradient = corr - l2 * weight;
            let dual = if options.positive {
                gradient.max(0.0)
            } else {
                gradient
            };
            dual_group_norm2 += dual * dual;
            group_norm2 += weight * weight;
            if l1 == 0.0 && l2 != 0.0 {
                let dual_coef = if options.positive {
                    corr.max(0.0)
                } else {
                    corr
                };
                ridge_gap += (dual_coef - l2 * weight).powi(2) / (2.0 * l2);
                if options.positive {
                    ridge_gap += (-corr).max(0.0) * weight;
                }
            }
        }
        dual_norm2 = dual_norm2.max(dual_group_norm2);
        l21 += group_norm2.sqrt();
        norm2 += group_norm2;
    }
    let scale = problem.target_energy;
    let dual_norm = dual_norm2.sqrt();
    if l1 == 0.0 {
        return if l2 > 0.0 {
            (ridge_gap, ridge_gap <= options.tol * scale)
        } else {
            (0.0, dual_norm <= options.tol * scale.sqrt())
        };
    }
    let factor = if dual_norm > l1 { l1 / dual_norm } else { 1.0 };
    let residual_target = scale - dot(problem.cross, coef);
    let residual_energy = (residual_target - dot(residual, coef)).max(0.0);
    let gap = ((1.0 + factor * factor) * (residual_energy + l2 * norm2) * 0.5 + l1 * l21
        - factor * residual_target)
        .max(0.0);
    (gap, gap <= options.tol * scale)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}
