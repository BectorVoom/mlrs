//! The cross-validation splitters (MODSEL-RS-02).
//!
//! One type per `sklearn.model_selection` splitter, each producing the SAME
//! train/test row indices sklearn produces for the same arguments — see the
//! parity contract in the [module docs](super).
//!
//! ## Index order is part of the contract
//!
//! sklearn's splitters fall into two families, and they disagree about the
//! order of the indices they hand back:
//!
//! * **mask-based** (`LeaveOneOut`, `LeavePOut`, `KFold`, `GroupKFold`,
//!   `StratifiedKFold`, `StratifiedGroupKFold`, `LeaveOneGroupOut`,
//!   `LeavePGroupsOut`, `PredefinedSplit`) run their test set through a boolean
//!   mask in `BaseCrossValidator.split`, so BOTH sides come back **ascending** —
//!   even `KFold(shuffle=True)`, whose shuffle reorders which rows land in a
//!   fold but not the order they are reported in;
//! * **permutation-based** (`ShuffleSplit`, `StratifiedShuffleSplit`) yield
//!   their draw order directly, so the indices are **shuffled**, and a caller
//!   that sorts them is no longer index-for-index compatible.
//!
//! `GroupShuffleSplit` sits between the two: it draws groups in permutation
//! order but recovers rows with `np.flatnonzero`, so its output is ascending.
//! `TimeSeriesSplit` slices `arange`, so it is ascending by construction.
//!
//! These differences are reproduced exactly; each type's docs say which family
//! it belongs to.

use std::collections::HashSet;

use super::rng::{approximate_mode, NumpyRandomState};
use super::{value_err, RandomStateSpec, Result, SizeSpec};

/// One train/test index pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Split {
    pub train: Vec<i64>,
    pub test: Vec<i64>,
}

/// A splitter's full output: the splits plus any `UserWarning` text sklearn
/// would have emitted.
///
/// The warnings are *returned* rather than logged because the Python layer has
/// to re-raise them through `warnings.warn` for `pytest.warns` parity — a
/// swallowed "least populated class" warning is a silent behavior difference
/// that estimator checks do notice.
#[derive(Debug, Clone, Default)]
pub struct Splits {
    pub splits: Vec<Split>,
    pub warnings: Vec<String>,
}

impl Splits {
    fn new(splits: Vec<Split>) -> Self {
        Self {
            splits,
            warnings: Vec::new(),
        }
    }
}

/// Turn a boolean test mask into an ascending `(train, test)` pair — the
/// `BaseCrossValidator.split` body.
fn split_from_mask(mask: &[bool]) -> Split {
    let mut train = Vec::with_capacity(mask.len());
    let mut test = Vec::new();
    for (i, &is_test) in mask.iter().enumerate() {
        if is_test {
            test.push(i as i64);
        } else {
            train.push(i as i64);
        }
    }
    Split { train, test }
}

/// Turn a set of test indices into an ascending `(train, test)` pair.
fn split_from_test_indices(n_samples: usize, test: &[i64]) -> Split {
    let mut mask = vec![false; n_samples];
    for &t in test {
        mask[t as usize] = true;
    }
    split_from_mask(&mask)
}

/// `np.unique(codes, return_counts=True)`-style counts over factorization
/// codes: `counts[k]` is how many elements carry code `k`.
fn code_counts(codes: &[i64]) -> Vec<i64> {
    let n_codes = codes.iter().copied().max().map_or(0, |m| m as usize + 1);
    let mut counts = vec![0i64; n_codes];
    for &c in codes {
        counts[c as usize] += 1;
    }
    counts
}

/// Re-encode sorted-unique class codes into sklearn's **order-of-appearance**
/// encoding (`StratifiedKFold._make_test_folds`'s `y_encoded`).
///
/// sklearn deliberately relabels so that class `0` is the first label
/// *appearing* in `y`, not the lexicographically smallest one. That choice is
/// visible in the output: the per-class fold blocks are laid down in this
/// order, so using sorted codes directly would assign different rows to
/// different folds.
fn encode_by_appearance(codes: &[i64], n_classes: usize) -> Vec<i64> {
    let mut first_seen = vec![usize::MAX; n_classes];
    for (i, &c) in codes.iter().enumerate() {
        let slot = &mut first_seen[c as usize];
        if *slot == usize::MAX {
            *slot = i;
        }
    }
    // rank of each class by first-occurrence position
    let mut order: Vec<usize> = (0..n_classes).collect();
    order.sort_by_key(|&k| first_seen[k]);
    let mut rank = vec![0i64; n_classes];
    for (r, &k) in order.iter().enumerate() {
        rank[k] = r as i64;
    }
    codes.iter().map(|&c| rank[c as usize]).collect()
}

/// Validate the shared `_BaseKFold.__init__` constraints.
fn check_base_kfold(n_splits: usize, shuffle: bool, random_state: RandomStateSpec) -> Result<()> {
    if n_splits <= 1 {
        return Err(value_err!(
            "k-fold cross-validation requires at least one train/test split by \
             setting n_splits=2 or more, got n_splits={n_splits}."
        ));
    }
    if !shuffle && !matches!(random_state, RandomStateSpec::Entropy) {
        return Err(value_err!(
            "Setting a random_state has no effect since shuffle is False. You \
             should leave random_state to its default (None), or set \
             shuffle=True."
        ));
    }
    Ok(())
}

// =========================================================================
// KFold
// =========================================================================

/// `sklearn.model_selection.KFold` — contiguous folds over (optionally
/// shuffled) row indices. **Ascending** output (mask-based).
#[derive(Debug, Clone)]
pub struct KFold {
    pub n_splits: usize,
    pub shuffle: bool,
    pub random_state: RandomStateSpec,
}

impl Default for KFold {
    fn default() -> Self {
        Self {
            n_splits: 5,
            shuffle: false,
            random_state: RandomStateSpec::Entropy,
        }
    }
}

impl KFold {
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            ..Default::default()
        }
    }

    pub fn get_n_splits(&self) -> usize {
        self.n_splits
    }

    pub fn split(&self, n_samples: usize) -> Result<Splits> {
        let mut rng = self.random_state.resolve();
        self.split_with_rng(n_samples, &mut rng)
    }

    pub fn split_with_rng(&self, n_samples: usize, rng: &mut NumpyRandomState) -> Result<Splits> {
        check_base_kfold(self.n_splits, self.shuffle, self.random_state)?;
        if self.n_splits > n_samples {
            return Err(value_err!(
                "Cannot have number of splits n_splits={} greater than the \
                 number of samples: n_samples={n_samples}.",
                self.n_splits
            ));
        }
        Ok(Splits::new(kfold_test_blocks(
            n_samples,
            self.n_splits,
            self.shuffle,
            rng,
        )))
    }
}

/// The shared `KFold._iter_test_indices` body: `arange`, optionally shuffled,
/// cut into `n_splits` near-equal contiguous blocks with the first
/// `n_samples % n_splits` blocks one longer.
fn kfold_test_blocks(
    n_samples: usize,
    n_splits: usize,
    shuffle: bool,
    rng: &mut NumpyRandomState,
) -> Vec<Split> {
    let mut indices: Vec<i64> = (0..n_samples as i64).collect();
    if shuffle {
        rng.shuffle(&mut indices);
    }
    let base = n_samples / n_splits;
    let extra = n_samples % n_splits;
    let mut out = Vec::with_capacity(n_splits);
    let mut current = 0usize;
    for f in 0..n_splits {
        let size = base + usize::from(f < extra);
        let test = &indices[current..current + size];
        out.push(split_from_test_indices(n_samples, test));
        current += size;
    }
    out
}

// =========================================================================
// GroupKFold
// =========================================================================

/// `sklearn.model_selection.GroupKFold` — non-overlapping groups across folds.
/// **Ascending** output (mask-based).
///
/// The two arms are genuinely different algorithms, not one algorithm with a
/// shuffle bolted on: `shuffle=False` runs a deterministic greedy balance
/// (assign the heaviest remaining group to the lightest fold), while
/// `shuffle=True` permutes the unique groups and cuts them into `n_splits`
/// contiguous chunks — which balances *group counts*, not row counts.
#[derive(Debug, Clone)]
pub struct GroupKFold {
    pub n_splits: usize,
    pub shuffle: bool,
    pub random_state: RandomStateSpec,
}

impl Default for GroupKFold {
    fn default() -> Self {
        Self {
            n_splits: 5,
            shuffle: false,
            random_state: RandomStateSpec::Entropy,
        }
    }
}

impl GroupKFold {
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            ..Default::default()
        }
    }

    pub fn get_n_splits(&self) -> usize {
        self.n_splits
    }

    pub fn split(&self, groups: &[i64]) -> Result<Splits> {
        let mut rng = self.random_state.resolve();
        self.split_with_rng(groups, &mut rng)
    }

    pub fn split_with_rng(&self, groups: &[i64], rng: &mut NumpyRandomState) -> Result<Splits> {
        check_base_kfold(self.n_splits, self.shuffle, self.random_state)?;
        let n_samples = groups.len();
        if self.n_splits > n_samples {
            return Err(value_err!(
                "Cannot have number of splits n_splits={} greater than the \
                 number of samples: n_samples={n_samples}.",
                self.n_splits
            ));
        }
        let counts = code_counts(groups);
        let n_groups = counts.len();
        if self.n_splits > n_groups {
            return Err(value_err!(
                "Cannot have number of splits n_splits={} greater than the \
                 number of groups: {n_groups}.",
                self.n_splits
            ));
        }

        // group_to_fold[g] = fold index that group g's rows belong to
        let group_to_fold: Vec<usize> = if self.shuffle {
            let permuted = rng.permutation(n_groups);
            // `np.array_split(unique_groups, n_splits)`: the first
            // `len % n_splits` chunks are one element longer.
            let base = n_groups / self.n_splits;
            let extra = n_groups % self.n_splits;
            let mut mapping = vec![0usize; n_groups];
            let mut cursor = 0usize;
            for f in 0..self.n_splits {
                let size = base + usize::from(f < extra);
                for &g in &permuted[cursor..cursor + size] {
                    mapping[g as usize] = f;
                }
                cursor += size;
            }
            mapping
        } else {
            // Distribute the most frequent group first onto the lightest fold.
            // `np.argsort(..., kind="stable")[::-1]` is a stable ascending sort
            // REVERSED — so among equal counts the LAST group comes first, not
            // the first. Reversing a stable sort is not the same as a stable
            // descending sort, and the difference decides real assignments.
            let mut order: Vec<usize> = (0..n_groups).collect();
            order.sort_by_key(|&g| counts[g]);
            order.reverse();

            let mut fold_weight = vec![0i64; self.n_splits];
            let mut mapping = vec![0usize; n_groups];
            for &g in &order {
                let lightest = fold_weight
                    .iter()
                    .enumerate()
                    .min_by(|(ia, a), (ib, b)| a.cmp(b).then(ia.cmp(ib)))
                    .map(|(i, _)| i)
                    .expect("n_splits >= 2");
                fold_weight[lightest] += counts[g];
                mapping[g] = lightest;
            }
            mapping
        };

        let splits = (0..self.n_splits)
            .map(|f| {
                let mask: Vec<bool> = groups
                    .iter()
                    .map(|&g| group_to_fold[g as usize] == f)
                    .collect();
                split_from_mask(&mask)
            })
            .collect();
        Ok(Splits::new(splits))
    }
}

// =========================================================================
// StratifiedKFold
// =========================================================================

/// `sklearn.model_selection.StratifiedKFold` — folds preserving the class
/// distribution. **Ascending** output (mask-based).
#[derive(Debug, Clone)]
pub struct StratifiedKFold {
    pub n_splits: usize,
    pub shuffle: bool,
    pub random_state: RandomStateSpec,
}

impl Default for StratifiedKFold {
    fn default() -> Self {
        Self {
            n_splits: 5,
            shuffle: false,
            random_state: RandomStateSpec::Entropy,
        }
    }
}

impl StratifiedKFold {
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            ..Default::default()
        }
    }

    pub fn get_n_splits(&self) -> usize {
        self.n_splits
    }

    pub fn split(&self, y: &[i64]) -> Result<Splits> {
        let mut rng = self.random_state.resolve();
        self.split_with_rng(y, &mut rng)
    }

    pub fn split_with_rng(&self, y: &[i64], rng: &mut NumpyRandomState) -> Result<Splits> {
        check_base_kfold(self.n_splits, self.shuffle, self.random_state)?;
        let n_samples = y.len();
        if self.n_splits > n_samples {
            return Err(value_err!(
                "Cannot have number of splits n_splits={} greater than the \
                 number of samples: n_samples={n_samples}.",
                self.n_splits
            ));
        }
        let test_folds = self.make_test_folds(y, rng)?;
        let mut splits = Splits::new(
            (0..self.n_splits)
                .map(|f| {
                    let mask: Vec<bool> = test_folds.folds.iter().map(|&v| v == f as i64).collect();
                    split_from_mask(&mask)
                })
                .collect(),
        );
        splits.warnings = test_folds.warnings;
        Ok(splits)
    }

    fn make_test_folds(&self, y: &[i64], rng: &mut NumpyRandomState) -> Result<TestFolds> {
        let counts_sorted = code_counts(y);
        let n_classes = counts_sorted.len();
        let y_encoded = encode_by_appearance(y, n_classes);
        let y_counts = code_counts(&y_encoded);

        if y_counts.iter().all(|&c| (self.n_splits as i64) > c) {
            return Err(value_err!(
                "n_splits={} cannot be greater than the number of members in \
                 each class.",
                self.n_splits
            ));
        }
        let min_groups = *y_counts.iter().min().expect("at least one class");
        let mut warnings = Vec::new();
        if (self.n_splits as i64) > min_groups {
            warnings.push(format!(
                "The least populated class in y has only {min_groups} members, \
                 which is less than n_splits={}.",
                self.n_splits
            ));
        }

        // sklearn's round-robin allocation: sort the encoded labels, then deal
        // them out `n_splits` at a time. `allocation[f][k]` is how many rows of
        // class `k` fold `f` must hold.
        let mut y_order = y_encoded.clone();
        y_order.sort_unstable();
        let mut allocation = vec![vec![0i64; n_classes]; self.n_splits];
        for (i, &cls) in y_order.iter().enumerate() {
            allocation[i % self.n_splits][cls as usize] += 1;
        }

        let mut folds = vec![0i64; y.len()];
        for k in 0..n_classes {
            // The fold labels for class k, laid down in blocks
            // (`np.arange(n_splits).repeat(allocation[:, k])`).
            let mut folds_for_class: Vec<i64> = Vec::new();
            for (f, alloc) in allocation.iter().enumerate() {
                folds_for_class.extend(std::iter::repeat_n(f as i64, alloc[k] as usize));
            }
            if self.shuffle {
                rng.shuffle(&mut folds_for_class);
            }
            let mut cursor = 0usize;
            for (i, &cls) in y_encoded.iter().enumerate() {
                if cls == k as i64 {
                    folds[i] = folds_for_class[cursor];
                    cursor += 1;
                }
            }
        }
        Ok(TestFolds { folds, warnings })
    }
}

struct TestFolds {
    folds: Vec<i64>,
    warnings: Vec<String>,
}

// =========================================================================
// StratifiedGroupKFold
// =========================================================================

/// `sklearn.model_selection.StratifiedGroupKFold` — non-overlapping groups
/// *and* preserved class balance. **Ascending** output (mask-based).
///
/// Greedy: process groups in descending order of their class-count standard
/// deviation (the most "lopsided" group first) and place each in whichever fold
/// minimizes the resulting per-class imbalance, breaking ties toward the
/// emptier fold.
#[derive(Debug, Clone)]
pub struct StratifiedGroupKFold {
    pub n_splits: usize,
    pub shuffle: bool,
    pub random_state: RandomStateSpec,
}

impl Default for StratifiedGroupKFold {
    fn default() -> Self {
        Self {
            n_splits: 5,
            shuffle: false,
            random_state: RandomStateSpec::Entropy,
        }
    }
}

impl StratifiedGroupKFold {
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            ..Default::default()
        }
    }

    pub fn get_n_splits(&self) -> usize {
        self.n_splits
    }

    pub fn split(&self, y: &[i64], groups: &[i64]) -> Result<Splits> {
        let mut rng = self.random_state.resolve();
        self.split_with_rng(y, groups, &mut rng)
    }

    pub fn split_with_rng(
        &self,
        y: &[i64],
        groups: &[i64],
        rng: &mut NumpyRandomState,
    ) -> Result<Splits> {
        check_base_kfold(self.n_splits, self.shuffle, self.random_state)?;
        let n_samples = y.len();
        if self.n_splits > n_samples {
            return Err(value_err!(
                "Cannot have number of splits n_splits={} greater than the \
                 number of samples: n_samples={n_samples}.",
                self.n_splits
            ));
        }
        let y_cnt = code_counts(y);
        let n_classes = y_cnt.len();
        if y_cnt.iter().all(|&c| (self.n_splits as i64) > c) {
            return Err(value_err!(
                "n_splits={} cannot be greater than the number of members in \
                 each class.",
                self.n_splits
            ));
        }
        let mut warnings = Vec::new();
        let n_smallest_class = *y_cnt.iter().min().expect("at least one class");
        if (self.n_splits as i64) > n_smallest_class {
            warnings.push(format!(
                "The least populated class in y has only {n_smallest_class} \
                 members, which is less than n_splits={}.",
                self.n_splits
            ));
        }

        let groups_cnt = code_counts(groups);
        let n_groups = groups_cnt.len();
        if self.n_splits > n_groups {
            return Err(value_err!(
                "Cannot have number of splits n_splits={} greater than the \
                 number of groups: {n_groups}.",
                self.n_splits
            ));
        }

        let mut y_counts_per_group = vec![vec![0f64; n_classes]; n_groups];
        for (&cls, &grp) in y.iter().zip(groups) {
            y_counts_per_group[grp as usize][cls as usize] += 1.0;
        }

        // The shuffle permutes the GROUP AXIS of the count matrix and remaps
        // `groups_inv` through the inverse permutation, so the greedy order is
        // randomized while every row still points at its own group.
        let mut groups_inv: Vec<i64> = groups.to_vec();
        if self.shuffle {
            let mut perm: Vec<i64> = (0..n_groups as i64).collect();
            rng.shuffle(&mut perm);
            let permuted: Vec<Vec<f64>> = perm
                .iter()
                .map(|&p| y_counts_per_group[p as usize].clone())
                .collect();
            y_counts_per_group = permuted;
            let mut inv_perm = vec![0i64; n_groups];
            for (i, &p) in perm.iter().enumerate() {
                inv_perm[p as usize] = i as i64;
            }
            groups_inv = groups_inv.iter().map(|&g| inv_perm[g as usize]).collect();
        }

        // Stable sort by DESCENDING per-group std over classes. sklearn sorts
        // ascending on the negated std with `kind="stable"`, which keeps the
        // (possibly shuffled) group order among ties — reversing an ascending
        // sort instead would flip those ties.
        let neg_std: Vec<f64> = y_counts_per_group.iter().map(|row| -pop_std(row)).collect();
        let mut sorted_groups_idx: Vec<usize> = (0..n_groups).collect();
        sorted_groups_idx.sort_by(|&a, &b| {
            neg_std[a]
                .partial_cmp(&neg_std[b])
                .expect("counts are finite")
                .then(a.cmp(&b))
        });

        let mut y_counts_per_fold = vec![vec![0f64; n_classes]; self.n_splits];
        let mut group_to_fold = vec![usize::MAX; n_groups];
        for &group_idx in &sorted_groups_idx {
            let group_y_counts = &y_counts_per_group[group_idx];
            let best_fold = self.find_best_fold(&mut y_counts_per_fold, &y_cnt, group_y_counts);
            for (c, v) in group_y_counts.iter().enumerate() {
                y_counts_per_fold[best_fold][c] += v;
            }
            group_to_fold[group_idx] = best_fold;
        }

        let splits = (0..self.n_splits)
            .map(|f| {
                let mask: Vec<bool> = groups_inv
                    .iter()
                    .map(|&g| group_to_fold[g as usize] == f)
                    .collect();
                split_from_mask(&mask)
            })
            .collect();
        Ok(Splits { splits, warnings })
    }

    /// sklearn's `_find_best_fold`: trial-place the group in every fold and
    /// keep the one with the smallest mean per-class std, breaking ties (by
    /// `np.isclose`) toward the fold holding fewer samples.
    fn find_best_fold(
        &self,
        y_counts_per_fold: &mut [Vec<f64>],
        y_cnt: &[i64],
        group_y_counts: &[f64],
    ) -> usize {
        let n_classes = y_cnt.len();
        let mut best_fold = 0usize;
        let mut min_eval = f64::INFINITY;
        let mut min_samples_in_fold = f64::INFINITY;
        for i in 0..self.n_splits {
            for c in 0..n_classes {
                y_counts_per_fold[i][c] += group_y_counts[c];
            }
            // std over the FOLD axis, per class, of the normalized counts
            let mut acc = 0.0;
            for c in 0..n_classes {
                let column: Vec<f64> = y_counts_per_fold
                    .iter()
                    .map(|fold| fold[c] / y_cnt[c] as f64)
                    .collect();
                acc += pop_std(&column);
            }
            let fold_eval = acc / n_classes as f64;
            for c in 0..n_classes {
                y_counts_per_fold[i][c] -= group_y_counts[c];
            }
            let samples_in_fold: f64 = y_counts_per_fold[i].iter().sum();
            let better = fold_eval < min_eval
                || (is_close(fold_eval, min_eval) && samples_in_fold < min_samples_in_fold);
            if better {
                min_eval = fold_eval;
                min_samples_in_fold = samples_in_fold;
                best_fold = i;
            }
        }
        best_fold
    }
}

/// Population standard deviation (`np.std`, i.e. `ddof=0`).
fn pop_std(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    var.sqrt()
}

/// `np.isclose(a, b)` with numpy's default tolerances — note the asymmetry:
/// the relative term is scaled by the SECOND argument, and a non-finite `b`
/// (the `inf` that seeds the tie-break) is never close to a finite `a`.
fn is_close(a: f64, b: f64) -> bool {
    if !b.is_finite() || !a.is_finite() {
        return a == b;
    }
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

// =========================================================================
// TimeSeriesSplit
// =========================================================================

/// `sklearn.model_selection.TimeSeriesSplit` — forward-chaining splits where
/// the test window always follows the train window. **Ascending** output.
#[derive(Debug, Clone)]
pub struct TimeSeriesSplit {
    pub n_splits: usize,
    pub max_train_size: Option<usize>,
    pub test_size: Option<usize>,
    pub gap: usize,
}

impl Default for TimeSeriesSplit {
    fn default() -> Self {
        Self {
            n_splits: 5,
            max_train_size: None,
            test_size: None,
            gap: 0,
        }
    }
}

impl TimeSeriesSplit {
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            ..Default::default()
        }
    }

    pub fn get_n_splits(&self) -> usize {
        self.n_splits
    }

    pub fn split(&self, n_samples: usize) -> Result<Splits> {
        if self.n_splits <= 1 {
            return Err(value_err!(
                "k-fold cross-validation requires at least one train/test split \
                 by setting n_splits=2 or more, got n_splits={}.",
                self.n_splits
            ));
        }
        let n_folds = self.n_splits + 1;
        let test_size = self.test_size.unwrap_or(n_samples / n_folds);
        if test_size == 0 {
            // sklearn reaches `range(start, stop, 0)` here and dies with
            // "range() arg 3 must not be zero"; a `while` loop would instead
            // spin forever, so this is a real guard rather than a nicety.
            return Err(value_err!(
                "test_size=0 is invalid: with n_samples={n_samples} and \
                 n_splits={}, each test fold would be empty.",
                self.n_splits
            ));
        }
        if n_folds > n_samples {
            return Err(value_err!(
                "Cannot have number of folds={n_folds} greater than the number \
                 of samples={n_samples}."
            ));
        }
        // The guard is an i64 subtraction on purpose: with a large `test_size`
        // the product overflows a `usize` subtraction into a huge positive
        // number and the "too many splits" error would never fire.
        if n_samples as i64 - self.gap as i64 - (test_size as i64 * self.n_splits as i64) <= 0 {
            return Err(value_err!(
                "Too many splits={} for number of samples={n_samples} with \
                 test_size={test_size} and gap={}.",
                self.n_splits,
                self.gap
            ));
        }

        let mut splits = Vec::with_capacity(self.n_splits);
        let first_start = n_samples - self.n_splits * test_size;
        let mut test_start = first_start;
        while test_start < n_samples {
            let train_end = test_start - self.gap;
            // `if self.max_train_size and self.max_train_size < train_end` —
            // sklearn's truthiness test means `max_train_size=0` is IGNORED,
            // not "keep zero training rows".
            let train_start = match self.max_train_size {
                Some(m) if m > 0 && m < train_end => train_end - m,
                _ => 0,
            };
            let train: Vec<i64> = (train_start as i64..train_end as i64).collect();
            let test_end = (test_start + test_size).min(n_samples);
            let test: Vec<i64> = (test_start as i64..test_end as i64).collect();
            splits.push(Split { train, test });
            test_start += test_size;
        }
        Ok(Splits::new(splits))
    }
}

// =========================================================================
// LeaveOneOut / LeavePOut  (lazy — see `nth_combination`)
// =========================================================================

/// `sklearn.model_selection.LeaveOneOut`. **Ascending** output.
///
/// Materializing all `n` splits costs `O(n^2)` memory, so the Python layer
/// drives this one split at a time through [`LeaveOneOut::split_at`]; the
/// eager [`LeaveOneOut::split`] is for Rust callers who want the whole thing.
#[derive(Debug, Clone, Copy, Default)]
pub struct LeaveOneOut;

impl LeaveOneOut {
    pub fn get_n_splits(&self, n_samples: usize) -> usize {
        n_samples
    }

    pub fn split_at(&self, n_samples: usize, i: usize) -> Result<Split> {
        if n_samples <= 1 {
            return Err(value_err!(
                "Cannot perform LeaveOneOut with n_samples={n_samples}."
            ));
        }
        Ok(split_from_test_indices(n_samples, &[i as i64]))
    }

    pub fn split(&self, n_samples: usize) -> Result<Splits> {
        let splits = (0..n_samples)
            .map(|i| self.split_at(n_samples, i))
            .collect::<Result<Vec<_>>>()?;
        Ok(Splits::new(splits))
    }
}

/// `sklearn.model_selection.LeavePOut`. **Ascending** output.
///
/// The split count is `C(n, p)`, which is astronomically large for even
/// modest `p` — `C(100, 3)` is 161 700 splits of 97 indices each. The Python
/// layer therefore never materializes them: it asks for the `i`-th split via
/// [`LeavePOut::split_at`], which unranks the `i`-th
/// `itertools.combinations` tuple directly.
#[derive(Debug, Clone, Copy)]
pub struct LeavePOut {
    pub p: usize,
}

impl LeavePOut {
    pub fn new(p: usize) -> Self {
        Self { p }
    }

    pub fn get_n_splits(&self, n_samples: usize) -> Result<u128> {
        if n_samples <= self.p {
            return Err(value_err!(
                "p={} must be strictly less than the number of samples={n_samples}",
                self.p
            ));
        }
        Ok(binomial(n_samples, self.p))
    }

    pub fn split_at(&self, n_samples: usize, i: u128) -> Result<Split> {
        if n_samples <= self.p {
            return Err(value_err!(
                "p={} must be strictly less than the number of samples={n_samples}",
                self.p
            ));
        }
        let test = nth_combination(n_samples, self.p, i)
            .ok_or_else(|| value_err!("split index {i} out of range"))?;
        Ok(split_from_test_indices(n_samples, &test))
    }

    pub fn split(&self, n_samples: usize) -> Result<Splits> {
        let total = self.get_n_splits(n_samples)?;
        let splits = (0..total)
            .map(|i| self.split_at(n_samples, i))
            .collect::<Result<Vec<_>>>()?;
        Ok(Splits::new(splits))
    }
}

/// `C(n, k)` saturating at `u128::MAX` — the multiplicative form, dividing at
/// each step so the intermediate never blows past the result.
pub fn binomial(n: usize, k: usize) -> u128 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 0..k as u128 {
        acc = acc.saturating_mul(n as u128 - i) / (i + 1);
    }
    acc
}

/// The `i`-th `itertools.combinations(range(n), k)` tuple in **lexicographic**
/// order, or `None` if `i >= C(n, k)`.
///
/// Unranking rather than iterating is what lets `LeavePOut` /
/// `LeavePGroupsOut` stream: each call is `O(n)` and holds `O(k)` memory,
/// against the `O(C(n,k) * n)` an eager materialization would need.
pub fn nth_combination(n: usize, k: usize, mut i: u128) -> Option<Vec<i64>> {
    if k > n || i >= binomial(n, k) {
        return None;
    }
    let mut out = Vec::with_capacity(k);
    let mut start = 0usize;
    let mut remaining = k;
    while remaining > 0 {
        for candidate in start..=(n - remaining) {
            let block = binomial(n - candidate - 1, remaining - 1);
            if i < block {
                out.push(candidate as i64);
                start = candidate + 1;
                remaining -= 1;
                break;
            }
            i -= block;
        }
    }
    Some(out)
}

// =========================================================================
// LeaveOneGroupOut / LeavePGroupsOut
// =========================================================================

/// `sklearn.model_selection.LeaveOneGroupOut`. **Ascending** output.
#[derive(Debug, Clone, Copy, Default)]
pub struct LeaveOneGroupOut;

impl LeaveOneGroupOut {
    pub fn get_n_splits(&self, groups: &[i64]) -> Result<usize> {
        let n_groups = code_counts(groups).len();
        if n_groups <= 1 {
            return Err(value_err!(
                "The groups parameter contains fewer than 2 unique groups \
                 ({n_groups}). LeaveOneGroupOut expects at least 2."
            ));
        }
        Ok(n_groups)
    }

    pub fn split(&self, groups: &[i64]) -> Result<Splits> {
        let n_groups = self.get_n_splits(groups)?;
        let splits = (0..n_groups as i64)
            .map(|g| {
                let mask: Vec<bool> = groups.iter().map(|&v| v == g).collect();
                split_from_mask(&mask)
            })
            .collect();
        Ok(Splits::new(splits))
    }
}

/// `sklearn.model_selection.LeavePGroupsOut`. **Ascending** output.
///
/// Streams through [`LeavePGroupsOut::split_at`] for the same combinatorial
/// reason as [`LeavePOut`].
#[derive(Debug, Clone, Copy)]
pub struct LeavePGroupsOut {
    pub n_groups: usize,
}

impl LeavePGroupsOut {
    pub fn new(n_groups: usize) -> Self {
        Self { n_groups }
    }

    fn check(&self, groups: &[i64]) -> Result<usize> {
        let n_unique = code_counts(groups).len();
        if self.n_groups >= n_unique {
            return Err(value_err!(
                "The groups parameter contains fewer than (or equal to) \
                 n_groups ({}) numbers of unique groups ({n_unique}). \
                 LeavePGroupsOut expects that at least n_groups + 1 ({}) \
                 unique groups be present",
                self.n_groups,
                self.n_groups + 1
            ));
        }
        Ok(n_unique)
    }

    pub fn get_n_splits(&self, groups: &[i64]) -> Result<u128> {
        let n_unique = self.check(groups)?;
        Ok(binomial(n_unique, self.n_groups))
    }

    pub fn split_at(&self, groups: &[i64], i: u128) -> Result<Split> {
        let n_unique = self.check(groups)?;
        let combo = nth_combination(n_unique, self.n_groups, i)
            .ok_or_else(|| value_err!("split index {i} out of range"))?;
        let held_out: HashSet<i64> = combo.into_iter().collect();
        let mask: Vec<bool> = groups.iter().map(|g| held_out.contains(g)).collect();
        Ok(split_from_mask(&mask))
    }

    pub fn split(&self, groups: &[i64]) -> Result<Splits> {
        let total = self.get_n_splits(groups)?;
        let splits = (0..total)
            .map(|i| self.split_at(groups, i))
            .collect::<Result<Vec<_>>>()?;
        Ok(Splits::new(splits))
    }
}

// =========================================================================
// PredefinedSplit
// =========================================================================

/// `sklearn.model_selection.PredefinedSplit` — one split per distinct
/// non-negative fold id in `test_fold`; rows tagged `-1` are never tested.
/// **Ascending** output.
#[derive(Debug, Clone)]
pub struct PredefinedSplit {
    pub test_fold: Vec<i64>,
}

impl PredefinedSplit {
    pub fn new(test_fold: Vec<i64>) -> Self {
        Self { test_fold }
    }

    /// The distinct fold ids, ascending, excluding `-1` (`np.unique` then
    /// drop `-1`).
    pub fn unique_folds(&self) -> Vec<i64> {
        let mut folds: Vec<i64> = self.test_fold.iter().copied().filter(|&f| f >= 0).collect();
        folds.sort_unstable();
        folds.dedup();
        folds
    }

    pub fn get_n_splits(&self) -> usize {
        self.unique_folds().len()
    }

    pub fn split(&self) -> Result<Splits> {
        let splits = self
            .unique_folds()
            .into_iter()
            .map(|f| {
                let mask: Vec<bool> = self.test_fold.iter().map(|&v| v == f).collect();
                split_from_mask(&mask)
            })
            .collect();
        Ok(Splits::new(splits))
    }
}

// =========================================================================
// ShuffleSplit family
// =========================================================================

/// `sklearn.model_selection.ShuffleSplit` — independent random train/test
/// draws. **Permutation order** output (NOT ascending).
#[derive(Debug, Clone)]
pub struct ShuffleSplit {
    pub n_splits: usize,
    pub test_size: SizeSpec,
    pub train_size: SizeSpec,
    pub random_state: RandomStateSpec,
}

impl Default for ShuffleSplit {
    fn default() -> Self {
        Self {
            n_splits: 10,
            test_size: SizeSpec::None,
            train_size: SizeSpec::None,
            random_state: RandomStateSpec::Entropy,
        }
    }
}

impl ShuffleSplit {
    pub fn get_n_splits(&self) -> usize {
        self.n_splits
    }

    pub fn split(&self, n_samples: usize) -> Result<Splits> {
        let mut rng = self.random_state.resolve();
        self.split_with_rng(n_samples, &mut rng)
    }

    pub fn split_with_rng(&self, n_samples: usize, rng: &mut NumpyRandomState) -> Result<Splits> {
        let (n_train, n_test) = super::validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            DEFAULT_TEST_SIZE_SHUFFLE,
        )?;
        Ok(Splits::new(shuffle_split_indices(
            n_samples,
            n_train,
            n_test,
            self.n_splits,
            rng,
        )))
    }
}

/// `BaseShuffleSplit._iter_indices`: one fresh permutation per split, test
/// taken off the front and train immediately after it — so with
/// `n_train + n_test < n_samples` the tail of the permutation is simply
/// dropped.
fn shuffle_split_indices(
    n_samples: usize,
    n_train: usize,
    n_test: usize,
    n_splits: usize,
    rng: &mut NumpyRandomState,
) -> Vec<Split> {
    (0..n_splits)
        .map(|_| {
            let permutation = rng.permutation(n_samples);
            Split {
                test: permutation[..n_test].to_vec(),
                train: permutation[n_test..n_test + n_train].to_vec(),
            }
        })
        .collect()
}

/// sklearn's `_default_test_size` for `ShuffleSplit` / `StratifiedShuffleSplit`.
pub const DEFAULT_TEST_SIZE_SHUFFLE: f64 = 0.1;
/// sklearn's `_default_test_size` for `GroupShuffleSplit`.
pub const DEFAULT_TEST_SIZE_GROUP_SHUFFLE: f64 = 0.2;
/// sklearn's `default_test_size` for `train_test_split`.
pub const DEFAULT_TEST_SIZE_TRAIN_TEST: f64 = 0.25;

/// `sklearn.model_selection.GroupShuffleSplit` — a `ShuffleSplit` over the
/// *groups*, expanded back to rows. **Ascending** output (`np.flatnonzero`),
/// unlike its `ShuffleSplit` parent.
#[derive(Debug, Clone)]
pub struct GroupShuffleSplit {
    pub n_splits: usize,
    pub test_size: SizeSpec,
    pub train_size: SizeSpec,
    pub random_state: RandomStateSpec,
}

impl Default for GroupShuffleSplit {
    fn default() -> Self {
        Self {
            n_splits: 5,
            test_size: SizeSpec::None,
            train_size: SizeSpec::None,
            random_state: RandomStateSpec::Entropy,
        }
    }
}

impl GroupShuffleSplit {
    pub fn get_n_splits(&self) -> usize {
        self.n_splits
    }

    pub fn split(&self, groups: &[i64]) -> Result<Splits> {
        let mut rng = self.random_state.resolve();
        self.split_with_rng(groups, &mut rng)
    }

    pub fn split_with_rng(&self, groups: &[i64], rng: &mut NumpyRandomState) -> Result<Splits> {
        let n_groups = code_counts(groups).len();
        let (n_train, n_test) = super::validate_shuffle_split(
            n_groups,
            self.test_size,
            self.train_size,
            DEFAULT_TEST_SIZE_GROUP_SHUFFLE,
        )?;
        let group_splits = shuffle_split_indices(n_groups, n_train, n_test, self.n_splits, rng);
        let splits = group_splits
            .into_iter()
            .map(|gs| {
                let train_groups: HashSet<i64> = gs.train.into_iter().collect();
                let test_groups: HashSet<i64> = gs.test.into_iter().collect();
                let mut train = Vec::new();
                let mut test = Vec::new();
                for (i, g) in groups.iter().enumerate() {
                    if train_groups.contains(g) {
                        train.push(i as i64);
                    }
                    if test_groups.contains(g) {
                        test.push(i as i64);
                    }
                }
                Split { train, test }
            })
            .collect();
        Ok(Splits::new(splits))
    }
}

/// `sklearn.model_selection.StratifiedShuffleSplit` — random draws that
/// preserve the class distribution. **Permutation order** output.
#[derive(Debug, Clone)]
pub struct StratifiedShuffleSplit {
    pub n_splits: usize,
    pub test_size: SizeSpec,
    pub train_size: SizeSpec,
    pub random_state: RandomStateSpec,
}

impl Default for StratifiedShuffleSplit {
    fn default() -> Self {
        Self {
            n_splits: 10,
            test_size: SizeSpec::None,
            train_size: SizeSpec::None,
            random_state: RandomStateSpec::Entropy,
        }
    }
}

impl StratifiedShuffleSplit {
    pub fn get_n_splits(&self) -> usize {
        self.n_splits
    }

    pub fn split(&self, y: &[i64]) -> Result<Splits> {
        let mut rng = self.random_state.resolve();
        self.split_with_rng(y, &mut rng)
    }

    pub fn split_with_rng(&self, y: &[i64], rng: &mut NumpyRandomState) -> Result<Splits> {
        let n_samples = y.len();
        let (n_train, n_test) = super::validate_shuffle_split(
            n_samples,
            self.test_size,
            self.train_size,
            DEFAULT_TEST_SIZE_SHUFFLE,
        )?;
        stratified_shuffle_split_indices(y, n_train, n_test, self.n_splits, rng)
    }
}

/// `StratifiedShuffleSplit._iter_indices` — shared with `train_test_split`'s
/// `stratify=` path, which reaches it with explicit integer sizes.
fn stratified_shuffle_split_indices(
    y: &[i64],
    n_train: usize,
    n_test: usize,
    n_splits: usize,
    rng: &mut NumpyRandomState,
) -> Result<Splits> {
    let class_counts = code_counts(y);
    let n_classes = class_counts.len();
    let min_count = *class_counts.iter().min().unwrap_or(&0);
    if min_count < 2 {
        let too_few: Vec<usize> = class_counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c < 2)
            .map(|(k, _)| k)
            .collect();
        return Err(value_err!(
            "The least populated classes in y have only 1 member, which is too \
             few. The minimum number of groups for any class cannot be less \
             than 2. Classes with too few members are: {too_few:?}"
        ));
    }
    if n_train < n_classes {
        return Err(value_err!(
            "The train_size = {n_train} should be greater or equal to the \
             number of classes = {n_classes}"
        ));
    }
    if n_test < n_classes {
        return Err(value_err!(
            "The test_size = {n_test} should be greater or equal to the number \
             of classes = {n_classes}"
        ));
    }

    // The rows of each class, ascending — `np.argsort(y_indices,
    // kind="stable")` split at the class boundaries.
    let mut class_indices: Vec<Vec<i64>> = vec![Vec::new(); n_classes];
    for (i, &c) in y.iter().enumerate() {
        class_indices[c as usize].push(i as i64);
    }

    let mut splits = Vec::with_capacity(n_splits);
    for _ in 0..n_splits {
        // Ties in the class counts are re-broken every iteration, so these two
        // calls must stay inside the loop — hoisting them would freeze one
        // tie-break for the whole run and drift from sklearn after split 0.
        let n_i = approximate_mode(&class_counts, n_train as i64, rng);
        let remaining: Vec<i64> = class_counts.iter().zip(&n_i).map(|(c, n)| c - n).collect();
        let t_i = approximate_mode(&remaining, n_test as i64, rng);

        let mut train = Vec::with_capacity(n_train);
        let mut test = Vec::with_capacity(n_test);
        for i in 0..n_classes {
            let permutation = rng.permutation(class_counts[i] as usize);
            let picked: Vec<i64> = permutation
                .iter()
                .map(|&p| class_indices[i][p as usize])
                .collect();
            train.extend_from_slice(&picked[..n_i[i] as usize]);
            test.extend_from_slice(&picked[n_i[i] as usize..(n_i[i] + t_i[i]) as usize]);
        }
        // A final permutation of each side, so the classes are interleaved
        // rather than blocked.
        let train = rng.permutation_of(&train);
        let test = rng.permutation_of(&test);
        splits.push(Split { train, test });
    }
    Ok(Splits::new(splits))
}

// =========================================================================
// Repeated splitters
// =========================================================================

/// `sklearn.model_selection.RepeatedKFold` — `n_repeats` shuffled `KFold`s
/// off ONE generator.
///
/// The shared generator is the whole point: sklearn builds a fresh
/// `KFold(random_state=rng, shuffle=True)` per repeat and hands it the *same*
/// live `RandomState`, so repeat `r` continues the stream repeat `r-1` left
/// off. Re-seeding per repeat would make every repeat identical.
#[derive(Debug, Clone)]
pub struct RepeatedKFold {
    pub n_splits: usize,
    pub n_repeats: usize,
    pub random_state: RandomStateSpec,
}

impl Default for RepeatedKFold {
    fn default() -> Self {
        Self {
            n_splits: 5,
            n_repeats: 10,
            random_state: RandomStateSpec::Entropy,
        }
    }
}

impl RepeatedKFold {
    pub fn get_n_splits(&self) -> usize {
        self.n_splits * self.n_repeats
    }

    pub fn split(&self, n_samples: usize) -> Result<Splits> {
        let mut rng = self.random_state.resolve();
        self.split_with_rng(n_samples, &mut rng)
    }

    pub fn split_with_rng(&self, n_samples: usize, rng: &mut NumpyRandomState) -> Result<Splits> {
        if self.n_repeats == 0 {
            return Err(value_err!("Number of repetitions must be greater than 0."));
        }
        let inner = KFold {
            n_splits: self.n_splits,
            shuffle: true,
            random_state: RandomStateSpec::Seed(0), // placeholder; rng is passed in
        };
        let mut all = Vec::with_capacity(self.get_n_splits());
        for _ in 0..self.n_repeats {
            all.extend(inner.split_with_rng(n_samples, rng)?.splits);
        }
        Ok(Splits::new(all))
    }
}

/// `sklearn.model_selection.RepeatedStratifiedKFold` — the stratified sibling
/// of [`RepeatedKFold`], with the same one-shared-generator semantics.
#[derive(Debug, Clone)]
pub struct RepeatedStratifiedKFold {
    pub n_splits: usize,
    pub n_repeats: usize,
    pub random_state: RandomStateSpec,
}

impl Default for RepeatedStratifiedKFold {
    fn default() -> Self {
        Self {
            n_splits: 5,
            n_repeats: 10,
            random_state: RandomStateSpec::Entropy,
        }
    }
}

impl RepeatedStratifiedKFold {
    pub fn get_n_splits(&self) -> usize {
        self.n_splits * self.n_repeats
    }

    pub fn split(&self, y: &[i64]) -> Result<Splits> {
        let mut rng = self.random_state.resolve();
        self.split_with_rng(y, &mut rng)
    }

    pub fn split_with_rng(&self, y: &[i64], rng: &mut NumpyRandomState) -> Result<Splits> {
        if self.n_repeats == 0 {
            return Err(value_err!("Number of repetitions must be greater than 0."));
        }
        let inner = StratifiedKFold {
            n_splits: self.n_splits,
            shuffle: true,
            random_state: RandomStateSpec::Seed(0), // placeholder; rng is passed in
        };
        let mut all = Vec::with_capacity(self.get_n_splits());
        let mut warnings = Vec::new();
        for _ in 0..self.n_repeats {
            let out = inner.split_with_rng(y, rng)?;
            all.extend(out.splits);
            for w in out.warnings {
                if !warnings.contains(&w) {
                    warnings.push(w);
                }
            }
        }
        Ok(Splits {
            splits: all,
            warnings,
        })
    }
}

// =========================================================================
// train_test_split
// =========================================================================

/// `sklearn.model_selection.train_test_split`'s **index** half.
///
/// Returns `(train, test)` for `n_samples` rows. Gathering the actual rows out
/// of each container is [`super::container`]'s job (and the Python shim's, for
/// pandas/pyarrow/scipy inputs Rust never sees).
///
/// `stratify` is the class-code vector, or `None`. Note that the sizes are
/// resolved against `default_test_size=0.25` here and then handed to the
/// underlying shuffle splitter as *absolute* counts — so `train_test_split`
/// does NOT inherit `ShuffleSplit`'s own 0.1 default, which it would if the
/// specs were forwarded unresolved.
pub fn train_test_split_indices(
    n_samples: usize,
    test_size: SizeSpec,
    train_size: SizeSpec,
    shuffle: bool,
    stratify: Option<&[i64]>,
    rng: &mut NumpyRandomState,
) -> Result<Split> {
    let (n_train, n_test) = super::validate_shuffle_split(
        n_samples,
        test_size,
        train_size,
        DEFAULT_TEST_SIZE_TRAIN_TEST,
    )?;

    if !shuffle {
        if stratify.is_some() {
            return Err(value_err!(
                "Stratified train/test split is not implemented for shuffle=False"
            ));
        }
        return Ok(Split {
            train: (0..n_train as i64).collect(),
            test: (n_train as i64..(n_train + n_test) as i64).collect(),
        });
    }

    match stratify {
        Some(y) => {
            let out = stratified_shuffle_split_indices(y, n_train, n_test, 1, rng)?;
            Ok(out.splits.into_iter().next().expect("n_splits == 1"))
        }
        None => Ok(shuffle_split_indices(n_samples, n_train, n_test, 1, rng)
            .into_iter()
            .next()
            .expect("n_splits == 1")),
    }
}
