//! Host decomposition bindings for sparse coding, NMF, and topic models.

use mlrs_algos::decomposition::{coordinate_descent, lda, sparse_coding};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn decomp_coordinate_descent(
    py: Python<'_>,
    gram: Vec<f64>,
    cross: Vec<f64>,
    mut weights: Vec<f64>,
    order: Vec<usize>,
    alpha: f64,
    positive: bool,
    max_iter: usize,
    tol: f64,
) -> PyResult<(Vec<f64>, f64)> {
    let width = order.len();
    let mut sorted = order.clone();
    sorted.sort_unstable();
    if width == 0
        || width.checked_mul(width) != Some(gram.len())
        || cross.len() != weights.len()
        || !cross.len().is_multiple_of(width)
        || !sorted.into_iter().eq(0..width)
    {
        return Err(PyValueError::new_err(
            "Invalid coordinate descent geometry.",
        ));
    }
    let violation = py.detach(|| {
        coordinate_descent::update(
            &gram,
            &cross,
            &mut weights,
            &order,
            alpha,
            positive,
            max_iter,
            tol,
        )
    });
    Ok((weights, violation))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn decomp_lda_infer(
    py: Python<'_>,
    offsets: Vec<usize>,
    indices: Vec<usize>,
    counts: Vec<f64>,
    word_topics: Vec<f64>,
    mut distributions: Vec<f64>,
    topics: usize,
    prior: f64,
    max_iter: usize,
    tol: f64,
    epsilon: f64,
    collect_stats: bool,
) -> PyResult<(Vec<f64>, Vec<f64>)> {
    if topics == 0
        || offsets.first() != Some(&0)
        || offsets.last() != Some(&indices.len())
        || offsets.windows(2).any(|span| span[0] > span[1])
        || (offsets.len() - 1).checked_mul(topics) != Some(distributions.len())
        || indices.len() != counts.len()
        || !word_topics.len().is_multiple_of(topics)
        || indices.iter().any(|&i| i >= word_topics.len() / topics)
        || !prior.is_finite()
        || prior <= 0.0
        || distributions.iter().any(|&x| !x.is_finite() || x <= 0.0)
    {
        return Err(PyValueError::new_err("Invalid document inference inputs."));
    }
    let batch = lda::DocumentBatch {
        offsets: &offsets,
        indices: &indices,
        counts: &counts,
        word_topics: &word_topics,
        topics,
    };
    let stats = py.detach(|| {
        lda::infer(
            batch,
            &mut distributions,
            prior,
            max_iter,
            tol,
            epsilon,
            collect_stats,
        )
    });
    Ok((distributions, stats))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn decomp_sparse_code(
    py: Python<'_>,
    gram: Vec<f64>,
    cross: Vec<f64>,
    width: usize,
    steps: usize,
    method: &str,
    alpha: f64,
    positive: bool,
) -> PyResult<Vec<f64>> {
    if width == 0
        || width.checked_mul(width) != Some(gram.len())
        || !cross.len().is_multiple_of(width)
    {
        return Err(PyValueError::new_err("Invalid sparse coding geometry."));
    }
    use sparse_coding::Coding;
    let method = match method {
        "omp" => Coding::Pursuit,
        "lars" => Coding::LeastAngle,
        "lasso_lars" => Coding::Lasso { alpha, positive },
        _ => return Err(PyValueError::new_err("Invalid sparse coding method.")),
    };
    Ok(py.detach(|| sparse_coding::encode(&gram, &cross, width, steps, method)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(decomp_coordinate_descent, m)?)?;
    m.add_function(wrap_pyfunction!(decomp_lda_infer, m)?)?;
    m.add_function(wrap_pyfunction!(decomp_sparse_code, m)?)?;
    Ok(())
}
