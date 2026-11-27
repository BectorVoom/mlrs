import Lean

open Lean

/- Run from the repository root with mlrs built from this checkout and importable by Python:
   lean --run tests/lean/Decomposition.lean [path-to-python]
   Tested with Lean 4.24.0. The interpreter defaults to python3. -/

private structure Observations where
  sparseCodes : Array (Array Float)
  nmfWeights : Array (Array Float)
  nmfComponents : Array (Array Float)
  singularValues : Array Float
  topicProbabilities : Array (Array Float)
  deriving FromJson

private def check (condition : Bool) (message : String) : IO Unit :=
  unless condition do
    throw (IO.userError message)

private def checkClose (label : String) (actual expected : Float) : IO Unit :=
  check (decide ((actual - expected).abs ≤ 1e-8))
    s!"{label}: expected {expected}, got {actual}"

private def checkVector (label : String) (actual expected : Array Float) : IO Unit := do
  check (actual.size == expected.size)
    s!"{label}: expected {expected.size} entries, got {actual.size}"
  for i in [:expected.size] do
    checkClose s!"{label}[{i}]" actual[i]! expected[i]!

private def checkMatrix
    (label : String) (actual expected : Array (Array Float)) : IO Unit := do
  check (actual.size == expected.size)
    s!"{label}: expected {expected.size} rows, got {actual.size}"
  for i in [:expected.size] do
    checkVector s!"{label}[{i}]" actual[i]! expected[i]!

private def testDecomposition (observed : Observations) : IO Unit := do
  -- With an orthonormal dictionary, lasso applies soft thresholding.
  checkMatrix "sparse codes" observed.sparseCodes
    #[#[2.0, -1.0, 0.0], #[0.0, 1.0, -3.0]]

  -- The input has rank one. Check reconstruction without fixing the scale of W or H.
  let w := observed.nmfWeights
  let h := observed.nmfComponents
  check (w.size == 2 && w.all (fun row => row.size == 1) &&
    h.size == 1 && h[0]!.size == 3) "NMF: unexpected factor shapes"
  check (w ++ h |>.all (fun row => row.all (fun x => decide (0 ≤ x))))
    "NMF: expected nonnegative factors"
  let reconstructed := w.map fun row => h[0]!.map (row[0]! * ·)
  checkMatrix "NMF reconstruction" reconstructed #[#[2.0, 4.0, 6.0], #[3.0, 6.0, 9.0]]

  checkVector "leading singular values" observed.singularValues #[5.0, 3.0]

  check (observed.topicProbabilities.size == 3) "LDA: expected one row per document"
  for row in observed.topicProbabilities do
    check (row.size == 2 && row.all (fun x => decide (0 ≤ x)))
      "LDA: expected two nonnegative topic probabilities per document"
    checkClose "LDA probability sum" (row.foldl (· + ·) 0.0) 1.0

def main (args : List String) : IO Unit := do
  let python ← match args with
    | [] => pure "python3"
    | [python] => pure python
    | _ => throw (IO.userError "Usage: lean --run tests/lean/Decomposition.lean [path-to-python]")
  let result ← IO.Process.output {
    cmd := python
    args := #["tests/lean/decomposition_probe.py"]
  }
  check (result.exitCode == 0)
    s!"mlrs probe exited with code {result.exitCode}\n{result.stderr}"
  let observed ← IO.ofExcept (Json.parse result.stdout >>= fromJson?)
  testDecomposition observed
  IO.println "Lean decomposition checks passed (sparse coding, NMF, randomized SVD, LDA)."
