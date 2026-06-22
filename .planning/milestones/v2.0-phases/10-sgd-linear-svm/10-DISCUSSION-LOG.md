# Phase 10: SGD / Linear-SVM - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-21
**Phase:** 10-sgd-linear-svm
**Areas discussed:** Construction idiom, Enums vs strings, Shared vs per-estimator, Validation timing

---

## Construction idiom

### Builder scope
| Option | Description | Selected |
|--------|-------------|----------|
| High-arity only | Builder for the four SGD/SVM estimators only; keep new()+with_opts() elsewhere. Two coexisting idioms. | |
| New project standard | Builder is the going-forward standard; retrofit existing later. | ✓ |
| Builder + keep new() | Both a builder and a new(essential) shortcut per estimator. | |

### Defaults
| Option | Description | Selected |
|--------|-------------|----------|
| sklearn-exact defaults | Empty builder reproduces scikit-learn's default estimator. | ✓ |
| Required essentials, no defaults | Force explicit loss/penalty; only fill numeric schedule params. | |

**User's choice:** New project standard + sklearn-exact defaults.
**Notes:** Builder becomes the canonical constructor and the going-forward standard (D-01/D-02), but retrofitting Phases 4–9 estimators is explicitly deferred (out of scope for Phase 10). `builder().build()` with no setters must equal sklearn's default estimator (D-03).

---

## Enums vs strings

### Enums
| Option | Description | Selected |
|--------|-------------|----------|
| Enums everywhere | Loss/Penalty/LearningRate as Rust enums; exhaustive, compile-time validity (KernelKind precedent). | ✓ |
| Strings | Keep sklearn strings like spectral's affinity:String. | |
| Enums, shared in core | Enums everywhere, shared ones in a common module. | |

### Py mapping
| Option | Description | Selected |
|--------|-------------|----------|
| TryFrom<&str>, error on unknown | Enums implement TryFrom<&str> (single source of truth in algos); PyO3 raises ValueError on unknown. | ✓ |
| Map in the Py wrapper | Do the string→enum match inside the PyO3 layer. | |

**User's choice:** Enums everywhere + TryFrom<&str> with ValueError on unknown.
**Notes:** Shared-module placement of the enums is effectively resolved by the "Shared vs per-estimator" area (shared SgdConfig).

---

## Shared vs per-estimator

### Sharing
| Option | Description | Selected |
|--------|-------------|----------|
| Per-estimator builders + shared prim config | Each builder exposes only its valid knobs; all lower into one shared SgdConfig. | ✓ |
| One shared builder for all four | Single SgdBuilder with every knob; exposes invalid combinations. | |
| Fully independent per-estimator | Own builder AND own params lowering, no shared config. | |

### Solver
| Option | Description | Selected |
|--------|-------------|----------|
| Implicit per-estimator default | Solver is internal: MBSGD*→SGD prim; LinearSVC/SVR→CD (dual='auto'). Matches sklearn. | ✓ |
| Explicit solver knob | Expose solver as a builder option. | |

**User's choice:** Per-estimator builders → shared SgdConfig + implicit solver.
**Notes:** Builders surface only valid knobs per estimator type; solver choice (SGD vs CD) stays an internal implementation detail (D-06/D-07).

---

## Validation timing

### Validation
| Option | Description | Selected |
|--------|-------------|----------|
| Split: params at build(), geometry at fit() | build()→Result validates data-independent params; data-dependent checks stay at fit. | ✓ |
| All at fit() (current convention) | Infallible build(); all validation at fit. | |
| Typestate (compile-time where possible) | Invalid combinations don't compile. | |

### Py errors
| Option | Description | Selected |
|--------|-------------|----------|
| ValueError at construction | build() errors + enum TryFrom failures raise Python ValueError at construction. | ✓ |
| Defer to fit() in Python | Surface all errors at fit() in Python. | |

**User's choice:** Split validation (build/fit) + ValueError at construction.
**Notes:** Diverges from the Phases 4–9 fit-time-only convention, justified by the builder (D-08/D-09).

---

## Claude's Discretion

- Exact builder method names, `BuildError` variant set, and `SgdConfig` field layout (within D-01…D-09).
- Whether LinearSVC/SVR builders reuse or wrap the existing CD estimator type.

## Deferred Ideas

- Retrofit existing Phase 4–9 estimators to the builder pattern (future cleanup phase).
- Explicit user-facing SGD-vs-CD solver selection.
- Typestate compile-time validation of invalid knob combinations.
