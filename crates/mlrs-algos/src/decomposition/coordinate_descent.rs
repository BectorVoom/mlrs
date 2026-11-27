//! Row-wise coordinate descent for L1-regularized quadratic objectives.
//! Decomposition updates reuse a single Gram matrix across all rows.

/// Minimize `0.5 * wᵀ G w - dot(c, w) + alpha * ||w||₁` for each row of
/// `cross`. The row-major `weights` buffer supplies the warm start.
#[allow(clippy::too_many_arguments)]
pub fn update(
    gram: &[f64],
    cross: &[f64],
    weights: &mut [f64],
    order: &[usize],
    alpha: f64,
    positive: bool,
    max_iter: usize,
    tol: f64,
) -> f64 {
    let width = order.len();
    let mut first_violation = 0.0;
    for (cross_row, row) in cross
        .chunks_exact(width)
        .zip(weights.chunks_exact_mut(width))
    {
        let mut gradient: Vec<_> = gram
            .chunks_exact(width)
            .zip(cross_row)
            .map(|(g, c)| g.iter().zip(row.iter()).map(|(x, y)| x * y).sum::<f64>() - c)
            .collect();
        let mut initial = 0.0;
        for step in 0..max_iter {
            let mut violation = 0.0;
            for &j in order {
                let diagonal = gram[j * width + j];
                let old = row[j];
                let target = diagonal * old - gradient[j];
                let excess = (target.abs() - alpha).max(0.0);
                let new = if diagonal <= 0.0 || (positive && target <= 0.0) {
                    0.0
                } else {
                    excess.copysign(target) / diagonal
                };
                let projected = if old == 0.0 {
                    if positive {
                        (gradient[j] + alpha).min(0.0)
                    } else {
                        (gradient[j].abs() - alpha).max(0.0)
                    }
                } else {
                    gradient[j] + alpha.copysign(old)
                };
                violation += projected.abs();
                let delta = new - old;
                if delta != 0.0 {
                    row[j] = new;
                    for (g, &entry) in gradient.iter_mut().zip(&gram[j * width..(j + 1) * width]) {
                        *g += delta * entry;
                    }
                }
            }
            if step == 0 {
                initial = violation;
                first_violation += violation;
            }
            if violation <= tol * initial {
                break;
            }
        }
    }
    first_violation
}
