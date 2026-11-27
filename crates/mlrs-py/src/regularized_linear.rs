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

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(regularized_gram_backend, module)?)?;
    module.add_function(wrap_pyfunction!(regularized_gram_f32, module)?)?;
    module.add_function(wrap_pyfunction!(regularized_gram_f64, module)?)?;
    Ok(())
}
