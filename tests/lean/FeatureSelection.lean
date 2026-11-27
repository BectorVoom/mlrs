import Lean

open Lean

/- Run from the repository root with the built mlrs extension importable by Python:
   lean --run tests/lean/FeatureSelection.lean [path-to-python] -/

private structure Observations where
  variances : Array Float
  varianceMask : Array Bool
  scores : Array Float
  mask : Array Bool
  selected : Array (Array Int)
  restored : Array (Array Int)
  selectedIsSparse : Bool
  restoredIsSparse : Bool
  emptyShape : Array Nat
  emptyRestoredShape : Array Nat
  emptyRestoredNnz : Nat
  emptyRestoredIsSparseArray : Bool
  emptyWarning : Bool
  customMask : Array Bool
  metaMask : Array Bool
  metaIsSparse : Bool
  denseOutput : Bool
  deriving FromJson

private def check (condition : Bool) (message : String) : IO Unit :=
  unless condition do
    throw (IO.userError message)

private def checkVector (label : String) (actual expected : Array Float) : IO Unit := do
  check (actual.size == expected.size)
    s!"{label}: expected {expected.size} entries, got {actual.size}"
  for i in [:expected.size] do
    check (decide ((actual[i]! - expected[i]!).abs ≤ 1e-8))
      s!"{label}[{i}]: expected {expected[i]!}, got {actual[i]!}"

private def testFeatureSelection (obs : Observations) : IO Unit := do
  checkVector "column population variances" obs.variances #[3.0, 0.0, 0.25, 2.0 / 3.0]
  check (obs.varianceMask == #[true, false, false, true]) "variance support at threshold 0.5"
  -- For the first column, observed class totals are 1 and 11 and expected totals are 6 each.
  checkVector "chi-squared scores" obs.scores #[25.0 / 3.0, 0.0, 1.0 / 3.0, 0.0]
  check (obs.mask == #[true, false, true, false]) "SelectKBest support"
  check (obs.selected == #[#[0, 0], #[0, 1], #[1, 0], #[4, 1], #[4, 0], #[3, 1]])
    "selected columns should retain their original order"
  check (obs.restored == #[#[0, 0, 0, 0], #[0, 0, 1, 0], #[1, 0, 0, 0],
                           #[4, 0, 1, 0], #[4, 0, 0, 0], #[3, 0, 1, 0]])
    "inverse transformation should insert zero columns"
  check (obs.selectedIsSparse && obs.restoredIsSparse) "default transforms should stay sparse"
  check (obs.emptyShape == #[6, 0] && obs.emptyWarning) "empty selection should warn and retain rows"
  check (obs.emptyRestoredShape == #[6, 4] && obs.emptyRestoredNnz == 0 &&
    obs.emptyRestoredIsSparseArray) "empty inverse should restore a sparse array of zeros"
  check (obs.customMask == #[false, true, false, false]) "custom sparse score support"
  check (obs.metaMask == #[true, false, false, false] && obs.metaIsSparse)
    "RFE should select the predictive feature and keep sparse output"
  check obs.denseOutput "explicit numpy output should preserve the selected and restored values"

def main (args : List String) : IO Unit := do
  let python ← match args with
    | [] => pure "python3"
    | [python] => pure python
    | _ => throw (IO.userError "Usage: lean --run tests/lean/FeatureSelection.lean [path-to-python]")
  let result ← IO.Process.output {
    cmd := python
    args := #["tests/lean/feature_selection_probe.py"]
  }
  check (result.exitCode == 0)
    s!"mlrs probe exited with code {result.exitCode}\n{result.stderr}"
  let observed ← IO.ofExcept (Json.parse result.stdout >>= fromJson?)
  testFeatureSelection observed
  IO.println "Lean feature-selection checks passed."
