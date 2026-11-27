import Lean

open Lean

/- Run from the repository root:
   PYTHONPATH=crates/mlrs-py/python lean --run tests/lean/RegularizedLinear.lean -/

private structure OracleCase where
  name : String
  fields : Json
  deriving FromJson

private structure Observations where
  lasso : Array Float
  elastic : Array Float
  intercept : Array Float
  alphas : Array Float
  pathAtHalf : Array Float
  gaps : Array Float
  iterations : Array Nat
  larsDropIterations : Nat
  observations : Array OracleCase
  deriving FromJson

private def check (condition : Bool) (message : String) : IO Unit :=
  unless condition do
    throw (IO.userError message)

private def vector (label : String) (actual expected : Array Float) (tol : Float) : IO Unit := do
  check (actual.size == expected.size)
    s!"{label}: expected {expected.size} entries, got {actual.size}"
  for i in [:expected.size] do
    let a := actual[i]!
    let b := expected[i]!
    check (decide ((a - b).abs ≤ tol * (1 + b.abs)))
      s!"{label}[{i}]: expected {b}, got {a}"

private def test (obs : Observations) : IO Unit := do
  -- Centered columns are orthogonal with norm²/n = 1. The first feature's
  -- group has norm 5, so Lasso shrinks (3, 4) by 1 - 1.5/5.
  vector "Lasso group shrinkage" obs.lasso #[2.1, 0, 0, 2.8, 0, 0] 1e-11
  vector "Elastic Net group shrinkage" obs.elastic
    #[3 * 0.85 / 1.75, -0.45 / 1.75, 0, 4 * 0.85 / 1.75, 0, 0] 1e-11
  vector "unpenalized intercepts" obs.intercept #[2.8, -8.6] 1e-11
  vector "descending path" obs.alphas #[5, 1.5, 0.5] 1e-11
  vector "last Lasso path coefficient" obs.pathAtHalf #[2.7, -0.7, 0, 3.6, 0, 0] 1e-11
  vector "path dual gaps" obs.gaps #[0, 0, 0] 1e-11
  check (obs.iterations.size == 3 && obs.iterations.all (· > 0)) "iteration counts for each alpha"
  check (obs.larsDropIterations > 9) "the LARS fixture must include feature drops"
  check (obs.observations.size == 10) "all nine estimators and enet_path should be exercised"
  for item in obs.observations do
    let fields ← IO.ofExcept item.fields.getObj?
    for (field, pair) in fields.toList do
      let values : Array (Array Float) ← IO.ofExcept (fromJson? pair)
      check (values.size == 2) s!"{item.name}.{field}: missing oracle comparison"
      vector s!"{item.name}.{field}" values[0]! values[1]! 1e-5

def main (args : List String) : IO Unit := do
  let python ← match args with
    | [] => pure "python3"
    | [python] => pure python
    | _ => throw (IO.userError "Usage: lean --run tests/lean/RegularizedLinear.lean [path-to-python]")
  let result ← IO.Process.output {
    cmd := python
    args := #["tests/lean/regularized_linear_probe.py"]
  }
  check (result.exitCode == 0) s!"Regression probe exited with {result.exitCode}\n{result.stderr}"
  let observations ← IO.ofExcept (Json.parse result.stdout >>= fromJson?)
  test observations
  IO.println "Lean regularized linear regression checks passed."
