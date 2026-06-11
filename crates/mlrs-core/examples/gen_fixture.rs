//! Throwaway generator for the oracle loader test fixture.
//!
//! Writes `tests/fixtures/oracle_case.npz` with named f32 AND f64 arrays using
//! npyz's own writer — no NumPy required (numpy is not installed in this
//! environment; SPIKE-FINDINGS.md §A4). Run once with:
//!   cargo run -p mlrs-core --example gen_fixture
//! The produced `.npz` is committed; this generator is not part of the test
//! path and can be removed once the fixture exists.

use std::fs::File;

use npyz::npz::NpzWriter;
use npyz::WriterBuilder;

fn main() -> std::io::Result<()> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/oracle_case.npz");
    let file = File::create(path)?;
    let mut npz = NpzWriter::new(file);
    write_arrays(&mut npz)?;
    // Dropping `npz` finalizes the zip central directory.
    drop(npz);
    println!("wrote {path}");
    Ok(())
}

fn write_arrays<W: std::io::Write + std::io::Seek>(
    npz: &mut NpzWriter<W>,
) -> std::io::Result<()> {
    // saxpy-style named case: a (f64 scalar), x/y (f32 inputs),
    // expected (f64 reference). Mixed dtypes exercise both decode paths.
    let x_f32: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let y_f32: Vec<f32> = vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0];
    let a_f64: Vec<f64> = vec![3.0];
    let expected_f64: Vec<f64> = x_f32
        .iter()
        .zip(y_f32.iter())
        .map(|(&x, &y)| a_f64[0] * x as f64 + y as f64)
        .collect();

    npz.array("a", Default::default())?
        .default_dtype()
        .shape(&[a_f64.len() as u64])
        .begin_nd()?
        .extend(a_f64.iter().copied())?;

    npz.array("x", Default::default())?
        .default_dtype()
        .shape(&[x_f32.len() as u64])
        .begin_nd()?
        .extend(x_f32.iter().copied())?;

    npz.array("y", Default::default())?
        .default_dtype()
        .shape(&[y_f32.len() as u64])
        .begin_nd()?
        .extend(y_f32.iter().copied())?;

    npz.array("expected", Default::default())?
        .default_dtype()
        .shape(&[expected_f64.len() as u64])
        .begin_nd()?
        .extend(expected_f64.iter().copied())?;

    Ok(())
}
