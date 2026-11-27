//! Least-angle Lasso path from a symmetric Gram matrix and target products.

pub struct Path {
    pub alphas: Vec<f64>,
    pub active: Vec<usize>,
    pub coefficients: Vec<f64>,
    pub skipped: bool,
}

pub struct Options {
    pub rows: usize,
    pub minimum: f64,
    pub max_iter: usize,
    pub eps: f64,
    pub positive: bool,
    pub keep_path: bool,
}

struct Cholesky {
    values: Vec<f64>,
    d: usize,
    max_diagonal: f64,
    min_pivot: f64,
}

impl Cholesky {
    fn new(d: usize) -> Self {
        Self {
            values: vec![0.0; d * d],
            d,
            max_diagonal: 0.0,
            min_pivot: f64::INFINITY,
        }
    }

    fn append(&mut self, active: &[usize], gram: &[f64], eps: f64) -> bool {
        let end = active.len() - 1;
        let j = active[end];
        let d = self.d;
        let (previous, rest) = self.values.split_at_mut(end * d);
        let row = &mut rest[..d];
        for (i, &feature) in active[..end].iter().enumerate() {
            let other = &previous[i * d..(i + 1) * d];
            row[i] = (gram[j * d + feature] - dot(&row[..i], &other[..i])) / other[i];
        }
        let diagonal = gram[j * d + j];
        let pivot = (diagonal - dot(&row[..end], &row[..end])).sqrt();
        let max_diagonal = diagonal.max(self.max_diagonal);
        let min_pivot = pivot.min(self.min_pivot);
        let floor = (f64::EPSILON * active.len() as f64 * max_diagonal).sqrt();
        if !pivot.is_finite() || min_pivot <= floor.max(eps) {
            return false;
        }
        row[end] = pivot;
        self.max_diagonal = max_diagonal;
        self.min_pivot = min_pivot;
        true
    }

    fn rebuild(&mut self, active: &[usize], gram: &[f64], eps: f64) -> bool {
        self.max_diagonal = 0.0;
        self.min_pivot = f64::INFINITY;
        (1..=active.len()).all(|end| self.append(&active[..end], gram, eps))
    }

    fn solve(&self, signs: &[f64], direction: &mut [f64]) {
        for (i, &sign) in signs.iter().enumerate() {
            let row = &self.values[i * self.d..];
            direction[i] = (sign - dot(&direction[..i], &row[..i])) / row[i];
        }
        for i in (0..signs.len()).rev() {
            let row = &self.values[i * self.d..];
            direction[i] /= row[i];
            let solved = direction[i];
            for (value, &entry) in direction[..i].iter_mut().zip(row) {
                *value -= solved * entry;
            }
        }
    }
}

pub fn solve(gram: &[f64], cross: &[f64], options: Options) -> Path {
    let d = cross.len();
    let n = options.rows as f64;
    let min_level = options.minimum * n;
    let mut residual = cross.to_vec();
    let mut coef = vec![0.0; d];
    let mut available = vec![true; d];
    let mut signs = Vec::new();
    let mut dropped = vec![false; d];
    let mut direction = vec![0.0; d];
    let mut slopes = vec![0.0; d];
    let mut factor = Cholesky::new(d);
    let mut candidate = best_feature(&residual, &available, options.positive);
    let mut level = candidate.map_or(0.0, |j| score(residual[j], options.positive).max(0.0));
    let tol = 64.0 * f64::EPSILON * level.max(f64::MIN_POSITIVE);
    let mut path = Path {
        alphas: vec![if level <= min_level + tol {
            options.minimum
        } else {
            level / n
        }],
        active: Vec::new(),
        coefficients: if options.keep_path {
            coef.clone()
        } else {
            Vec::new()
        },
        skipped: false,
    };

    while path.alphas.len() <= options.max_iter && level > min_level + tol {
        if let Some(j) = candidate {
            path.active.push(j);
            signs.push(if options.positive {
                1.0
            } else {
                residual[j].signum()
            });
            available[j] = false;
            if !factor.append(&path.active, gram, options.eps) {
                path.active.pop();
                signs.pop();
                path.skipped = true;
                candidate = best_feature(&residual, &available, options.positive)
                    .filter(|&i| score(residual[i], options.positive) >= level - tol);
                continue;
            }
        }
        if path.active.is_empty() {
            break;
        }
        factor.solve(&signs, &mut direction);
        slopes.fill(0.0);
        for (&j, &velocity) in path.active.iter().zip(&direction) {
            for (slope, &g) in slopes.iter_mut().zip(&gram[j * d..(j + 1) * d]) {
                *slope += velocity * g;
            }
        }

        let mut step = level - min_level;
        candidate = None;
        let sign_choices: &[f64] = if options.positive {
            &[1.0]
        } else {
            &[1.0, -1.0]
        };
        for &sign in sign_choices {
            // Find each sign's exact minimum before comparing against the
            // tolerance, keeping the first feature on exact ties.
            let mut crossing = f64::INFINITY;
            let mut entrant = None;
            for j in 0..d {
                if !available[j] {
                    continue;
                }
                let denominator = 1.0 - sign * slopes[j];
                let numerator = level - sign * residual[j];
                if denominator > 16.0 * f64::EPSILON
                    && numerator >= -tol
                    && (!dropped[j] || numerator > tol)
                {
                    let time = numerator.max(0.0) / denominator;
                    if time < crossing {
                        entrant = Some(j);
                        crossing = time;
                    }
                }
            }
            if crossing < step - tol {
                step = crossing;
                candidate = entrant;
            }
        }
        let first_drop = path
            .active
            .iter()
            .zip(&direction)
            .map(|(&j, &v)| -coef[j] / v)
            .filter(|&time| time > tol)
            .fold(f64::INFINITY, f64::min);
        let should_drop = first_drop < step - tol;
        if should_drop {
            step = first_drop;
            candidate = None;
        }

        dropped.fill(false);
        for (index, &j) in path.active.iter().enumerate() {
            let velocity = direction[index];
            if should_drop && (-coef[j] / velocity - first_drop).abs() <= tol {
                dropped[j] = true;
                available[j] = true;
                coef[j] = 0.0;
            } else {
                coef[j] += step * velocity;
            }
        }
        for (corr, &slope) in residual.iter_mut().zip(&slopes) {
            *corr -= step * slope;
        }
        level = (level - step).max(min_level);
        if step > tol {
            path.alphas.push(level / n);
            if options.keep_path {
                path.coefficients.extend_from_slice(&coef);
            }
        }
        if should_drop {
            for i in (0..path.active.len()).rev() {
                if dropped[path.active[i]] {
                    path.active.remove(i);
                    signs.remove(i);
                }
            }
            // The previous factor was positive definite. A subset remains
            // positive definite unless roundoff makes it numerically singular.
            if !factor.rebuild(&path.active, gram, options.eps) {
                break;
            }
        }
    }
    if !options.keep_path {
        path.coefficients = coef;
    }
    path
}

fn score(value: f64, positive: bool) -> f64 {
    if positive {
        value
    } else {
        value.abs()
    }
}

fn best_feature(values: &[f64], available: &[bool], positive: bool) -> Option<usize> {
    let mut best = None;
    let mut best_score = f64::NEG_INFINITY;
    for (j, (&value, &is_available)) in values.iter().zip(available).enumerate() {
        let value = score(value, positive);
        if is_available && value > best_score {
            best = Some(j);
            best_score = value;
        }
    }
    best
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}
