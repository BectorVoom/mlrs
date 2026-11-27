//! Device sufficient statistics for the coordinate-descent and LARS paths.

use arrow::array::{Float32Array, Float64Array};
use mlrs_backend::capability::{active_backend_name, f64_device_kernels_available};
use mlrs_backend::device_array::DeviceArray;
use mlrs_backend::prims::gram::{fused_centering_available, gram_xty_multi};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::{egress, ingress, lock_pool};

/// Require the bounded, non-GEMM Gram path. The latter does not support f64
/// on every runtime that supports the repository's own f64 kernels.
#[pyfunction]
fn regularized_gram_backend(features: usize, float64: bool) -> Option<&'static str> {
    let backend = active_backend_name();
    let supported = features != 0
        && features <= 256
        && backend != "cpu"
        && if float64 {
            f64_device_kernels_available() && fused_centering_available::<f64>(features)
        } else {
            fused_centering_available::<f32>(features)
        };
    supported.then_some(backend)
}

macro_rules! bind_gram {
    ($name:ident, $float:ty, $array:ty, $slice:path, $egress:path) => {
        #[pyfunction]
        fn $name<'py>(
            py: Python<'py>,
            design: &Bound<'py, PyAny>,
            targets: &Bound<'py, PyAny>,
            rows: usize,
            features: usize,
            outputs: usize,
        ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
            if regularized_gram_backend(features, size_of::<$float>() == 8).is_none() {
                return Err(PyValueError::new_err(
                    "Unsupported device Gram geometry or dtype.",
                ));
            }
            let design = ingress::capsule_to_array(design)?;
            let targets = ingress::capsule_to_array(targets)?;
            let design = design
                .as_any()
                .downcast_ref::<$array>()
                .ok_or_else(|| crate::errors::unsupported_dtype_err(design.data_type()))?;
            let targets = targets
                .as_any()
                .downcast_ref::<$array>()
                .ok_or_else(|| crate::errors::unsupported_dtype_err(targets.data_type()))?;
            let x = $slice(design)?;
            let y = $slice(targets)?;
            if rows == 0
                || outputs == 0
                || rows.checked_mul(features) != Some(x.len())
                || rows.checked_mul(outputs) != Some(y.len())
                || x.len() > u32::MAX as usize
                || y.len() > u32::MAX as usize
                || features
                    .checked_mul(outputs)
                    .is_none_or(|len| len > u32::MAX as usize)
            {
                return Err(PyValueError::new_err(
                    "Invalid device Gram matrix dimensions.",
                ));
            }
            let (gram, cross) = py.detach(|| {
                let mut pool = lock_pool();
                let x = DeviceArray::from_host(&mut pool, x);
                let y = DeviceArray::from_host(&mut pool, y);
                let result = gram_xty_multi(&mut pool, &x, &y, None, rows, features, outputs);
                x.release_into(&mut pool);
                y.release_into(&mut pool);
                let (gram, cross) =
                    result.map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
                let gram_values = gram.to_host_metered(&mut pool);
                let cross_values = cross.to_host_metered(&mut pool);
                gram.release_into(&mut pool);
                cross.release_into(&mut pool);
                Ok::<_, PyErr>((gram_values, cross_values))
            })?;
            Ok(($egress(py, gram)?, $egress(py, cross)?))
        }
    };
}

bind_gram!(
    regularized_gram_f32,
    f32,
    Float32Array,
    ingress::host_slice_f32,
    egress::f32_vec_to_pyarrow
);
bind_gram!(
    regularized_gram_f64,
    f64,
    Float64Array,
    ingress::host_slice_f64,
    egress::f64_vec_to_pyarrow
);

fn values(array: &arrow::array::ArrayRef) -> PyResult<&[f64]> {
    let floats = array
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| crate::errors::unsupported_dtype_err(array.data_type()))?;
    ingress::host_slice_f64(floats)
}
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn regularized_cd_gram<'py>(
    py: Python<'py>,
    gram: &Bound<'py, PyAny>,
    cross: &Bound<'py, PyAny>,
    initial: &Bound<'py, PyAny>,
    features: usize,
    targets: usize,
    target_energy: f64,
    alphas: Vec<f64>,
    ratio: f64,
    max_iter: usize,
    tol: f64,
    positive: bool,
) -> PyResult<(Bound<'py, PyAny>, Vec<f64>, Vec<usize>, Vec<bool>)> {
    use mlrs_algos::linear::grouped_gram_cd::{solve_path, GramProblem, Options};

    let gram_array = ingress::capsule_to_array(gram)?;
    let cross_array = ingress::capsule_to_array(cross)?;
    let initial_array = ingress::capsule_to_array(initial)?;
    let gram = values(&gram_array)?;
    let cross = values(&cross_array)?;
    let initial = values(&initial_array)?;
    if features == 0
        || targets == 0
        || features.checked_mul(features) != Some(gram.len())
        || features.checked_mul(targets) != Some(cross.len())
        || initial.len() != cross.len()
        || initial.len().checked_mul(alphas.len()).is_none()
        || max_iter == 0
    {
        return Err(PyValueError::new_err(
            "Invalid coordinate-descent Gram geometry.",
        ));
    }
    let problem = GramProblem {
        gram,
        cross,
        features,
        targets,
        target_energy,
    };
    let options = Options {
        ratio,
        max_iter,
        tol,
        positive,
    };
    let result = py.detach(|| solve_path(problem, initial.to_vec(), &alphas, options));
    Ok((
        egress::f64_vec_to_pyarrow(py, result.coefficients)?,
        result.gaps,
        result.iterations,
        result.converged,
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn regularized_lars_gram<'py>(
    py: Python<'py>,
    gram: &Bound<'py, PyAny>,
    cross: &Bound<'py, PyAny>,
    rows: usize,
    minimum: f64,
    max_iter: usize,
    eps: f64,
    positive: bool,
    keep_path: bool,
) -> PyResult<(Vec<f64>, Vec<usize>, Bound<'py, PyAny>, bool)> {
    use mlrs_algos::linear::gram_lasso_lars::{solve, Options};

    let gram = ingress::capsule_to_array(gram)?;
    let cross = ingress::capsule_to_array(cross)?;
    let gram = values(&gram)?;
    let cross = values(&cross)?;
    if rows == 0 || cross.is_empty() || cross.len().checked_mul(cross.len()) != Some(gram.len()) {
        return Err(PyValueError::new_err("Invalid least-angle Gram geometry."));
    }
    let options = Options {
        rows,
        minimum,
        max_iter,
        eps,
        positive,
        keep_path,
    };
    let path = py.detach(|| solve(gram, cross, options));
    Ok((
        path.alphas,
        path.active,
        egress::f64_vec_to_pyarrow(py, path.coefficients)?,
        path.skipped,
    ))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(regularized_gram_backend, module)?)?;
    module.add_function(wrap_pyfunction!(regularized_cd_gram, module)?)?;
    module.add_function(wrap_pyfunction!(regularized_lars_gram, module)?)?;
    module.add_function(wrap_pyfunction!(regularized_gram_f32, module)?)?;
    module.add_function(wrap_pyfunction!(regularized_gram_f64, module)?)?;
    Ok(())
}
