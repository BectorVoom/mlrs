//! `numpy_rng` — the numpy BIT-COMPATIBILITY helpers `mutual_info_*` needs
//! (FSEL-01): a `numpy.random.RandomState` (MT19937) replica for its
//! tie-breaking noise, and numpy's PAIRWISE summation for the `scale()` divisor
//! that noise is applied on top of.
//!
//! Both exist for the same single reason, developed below: `mutual_info_*` is a
//! neighbour-COUNTING estimator, its counts are decided at the last bit, and an
//! answer that is merely close to sklearn's is not an answer at all on the
//! tied-value data the estimator is most used for.
//!
//! ## Why this crate suddenly needs an MT19937 when every other estimator does not
//! Everywhere else in mlrs, `random_state` seeds a SplitMix64 and the oracle
//! pins the ANSWER rather than the path to it — `TSNE`, `KMeans(init='random')`,
//! `RandomForest`'s bootstrap, `random_projection`'s matrix all document that
//! choice, because in each case the RNG explores a space and any well-behaved
//! stream reaches an equivalent answer.
//!
//! `mutual_info_*` is different in kind. Its `random_state` does not explore
//! anything: it draws `1e-10 · mean|x| · N(0,1)` noise whose ONLY purpose is to
//! break exact ties between repeated values, because the Kraskov and Ross
//! estimators count neighbours inside a radius and a duplicated value makes that
//! count ambiguous. So:
//!
//! * on data with NO duplicates the noise is immaterial and any stream agrees to
//!   ~1e-8 — the SplitMix64 route would have been fine;
//! * on data WITH duplicates (a binned feature, a rounded measurement, a
//!   one-hot column — i.e. most real feature matrices that anyone runs
//!   `mutual_info_classif` on) the noise DECIDES which of the tied points falls
//!   inside the radius, and two different streams give genuinely different
//!   mutual information. There is no "equivalent answer" to converge to; the
//!   value simply differs.
//!
//! That makes the score untestable against sklearn at 1e-5 on exactly the inputs
//! it is most used for, unless the stream matches. So this module matches it —
//! the same decision, for the same reason, that `model_selection`'s splitters
//! made when they chose bit-for-bit MT19937 over a SplitMix64 shuffle.
//!
//! ## What is replicated
//! `numpy.random.RandomState(seed)` for an INTEGER seed, and its
//! `standard_normal` output, which is all `sklearn.utils.check_random_state`
//! plus `_estimate_mi` ever use:
//!
//! * [`NumpyRandomState::new`] is numpy's `mt19937_seed` — the reference
//!   `init_genrand`, which numpy uses for a scalar integer seed (an
//!   array/`SeedSequence` seed goes through `init_by_array` instead and is not
//!   replicated, because `check_random_state` cannot produce one).
//! * [`NumpyRandomState::next_u32`] is the reference `genrand_uint32` (tempered
//!   Mersenne Twister with the standard 624-word twist).
//! * [`NumpyRandomState::next_f64`] is numpy's `random_double`: TWO 32-bit draws
//!   combined as `(a>>5)·2²⁶ + (b>>6)) / 2⁵³`. Note it is not `u64/2⁶⁴` — using
//!   that instead reproduces neither the values nor even the consumption rate.
//! * [`NumpyRandomState::standard_normal`] is numpy's `legacy_gauss`: the
//!   Marsaglia POLAR method with a cached second value, returning `f·x2` FIRST
//!   and caching `f·x1`. The order is observable — swapping them transposes
//!   every consecutive pair of the output.
//!
//! Nothing here is a device concern: it is a scalar host stream feeding a
//! host k-NN estimator.
//!
//! Tests live in `crates/mlrs-algos/tests/` (AGENTS.md §2), pinned against
//! values printed by `numpy.random.RandomState` directly.

/// Mersenne Twister state size, in 32-bit words.
const N: usize = 624;
/// Twist offset.
const M: usize = 397;
/// Twist matrix constant.
const MATRIX_A: u32 = 0x9908_b0df;
/// Most-significant-bit mask.
const UPPER_MASK: u32 = 0x8000_0000;
/// Least-significant 31 bits.
const LOWER_MASK: u32 = 0x7fff_ffff;

/// A bit-exact replica of `numpy.random.RandomState` seeded from an integer.
///
/// Deliberately NOT named `Mt19937`: what callers depend on is agreement with
/// numpy's legacy generator specifically, including its `random_double`
/// bit-packing and its `legacy_gauss` ordering, neither of which is part of the
/// MT19937 specification.
#[derive(Debug, Clone)]
pub struct NumpyRandomState {
    state: [u32; N],
    index: usize,
    /// The cached second value of a `legacy_gauss` polar pair, if one is pending.
    gauss: Option<f64>,
}

impl NumpyRandomState {
    /// Seed exactly as `numpy.random.RandomState(seed)` does for an integer
    /// seed: `mt19937_seed`, i.e. the reference `init_genrand` on
    /// `seed & 0xffffffff`.
    ///
    /// numpy masks the seed to 32 bits (and rejects negatives) before seeding,
    /// so a `u64` seed above `2³²` aliases — reproduced rather than rejected,
    /// because that is what a caller porting `random_state=2**33` observes.
    pub fn new(seed: u64) -> Self {
        let mut state = [0u32; N];
        state[0] = (seed & 0xffff_ffff) as u32;
        for i in 1..N {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Self {
            state,
            index: N,
            gauss: None,
        }
    }

    /// One tempered 32-bit draw — the reference `genrand_uint32`.
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// Regenerate the 624-word block.
    fn twist(&mut self) {
        for i in 0..N {
            let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % N] & LOWER_MASK);
            let mut next = self.state[(i + M) % N] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    /// A `[0, 1)` double, numpy's `random_double`: `((a>>5)·2²⁶ + (b>>6)) / 2⁵³`
    /// from two consecutive 32-bit draws.
    pub fn next_f64(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }

    /// One standard normal, numpy's `legacy_gauss` (Marsaglia polar with a
    /// cached second value).
    ///
    /// Returns `f·x2` and caches `f·x1` — numpy's order, which is observable in
    /// every even-length draw.
    pub fn standard_normal(&mut self) -> f64 {
        if let Some(g) = self.gauss.take() {
            return g;
        }
        loop {
            let x1 = 2.0 * self.next_f64() - 1.0;
            let x2 = 2.0 * self.next_f64() - 1.0;
            let r2 = x1 * x1 + x2 * x2;
            if r2 < 1.0 && r2 != 0.0 {
                let f = (-2.0 * r2.ln() / r2).sqrt();
                self.gauss = Some(f * x1);
                return f * x2;
            }
        }
    }

    /// `rng.standard_normal(size=len)` — a C-order fill, which for a 2-D `size`
    /// means row-major.
    pub fn standard_normal_vec(&mut self, len: usize) -> Vec<f64> {
        (0..len).map(|_| self.standard_normal()).collect()
    }
}

// ===========================================================================
// numpy's PAIRWISE summation, and the mean / var / std built on it
// ===========================================================================
//
// ## Why bit-exact summation is needed and a mathematically-equal sum is not
// `_estimate_mi` divides every continuous column by `np.nanstd(column)` before
// adding the noise. A sequential Rust sum and numpy's pairwise sum agree to
// ~1e-16 RELATIVE, which is normally the definition of "the same answer" — and
// here it is not, for a specific and reproducible reason:
//
// The estimator's radius is `nextafter(kth_neighbour_distance, 0)`, i.e. the
// k-th distance minus ONE ULP, chosen so the k-th neighbour is excluded by the
// narrowest possible margin. On tied-value data (a binned feature, a rounded
// measurement) many points sit within a few ULP of that boundary, so a 1-ULP
// change in the divisor moves points across it and changes `m_all` by whole
// integers. Measured on this crate's own oracle fixture, a sequentially-summed
// `std` shifted `mutual_info_regression` on the tied column by 1.5% — a hundred
// times the 1e-5 contract, from a divisor that differed in its last bit.
//
// So matching numpy's reduction is the same decision, taken for the same reason,
// as matching its MT19937 above: a tie-breaking mechanism can only be verified
// against the reference if it is reproduced exactly.
//
// ## What is replicated
// numpy's `pairwise_sum_DOUBLE` (`numpy/_core/src/umath/loops_utils.h.src`),
// which every `np.add.reduce` over a contiguous double array goes through:
// a plain sequential sum below 8 elements, an 8-way unrolled accumulation with a
// balanced final combine up to `PW_BLOCKSIZE = 128`, and above that a recursive
// split into halves rounded down to a multiple of the unroll factor. Then
// `np.mean` / `np.var` / `np.std` as `_methods._mean` / `_var` / `_std` compose
// them: mean is `sum/n`, variance is `sum((x − mean)²)/n` with BOTH sums
// pairwise, and std is its square root.
//
// NaN-skipping (`nanstd`) is deliberately NOT replicated: `_estimate_mi` calls
// plain `scale()`, which uses `np.nanstd`, but mlrs rejects non-finite input at
// ingress (`base.py`'s `check_array`, and the Rust bridge), so no NaN can reach
// here and a NaN-skipping variant would be untestable dead code.

/// numpy's `PW_BLOCKSIZE` — the size above which `pairwise_sum` recurses.
const PW_BLOCKSIZE: usize = 128;

/// numpy's unroll factor for the block accumulation.
const PW_UNROLL: usize = 8;

/// `np.add.reduce(a)` for a contiguous `f64` slice — numpy's `pairwise_sum_DOUBLE`
/// (see the section comment above for why bit-exactness is load-bearing).
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < PW_UNROLL {
        // numpy's `n < 8` branch: a plain sequential sum starting from 0.0.
        return a.iter().sum();
    }
    if n <= PW_BLOCKSIZE {
        let mut r = [0.0f64; PW_UNROLL];
        r[..PW_UNROLL].copy_from_slice(&a[..PW_UNROLL]);
        let tail = n - (n % PW_UNROLL);
        let mut i = PW_UNROLL;
        while i < tail {
            for (j, acc) in r.iter_mut().enumerate() {
                *acc += a[i + j];
            }
            i += PW_UNROLL;
        }
        // The BALANCED combine, not a left fold over `r` — the association is
        // what makes this differ from a sequential sum, and reproducing it is
        // the whole point.
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        // numpy finishes the non-multiple-of-8 remainder sequentially, AFTER the
        // combine.
        for &v in &a[tail..] {
            res += v;
        }
        return res;
    }
    // Split in half, rounded DOWN to a multiple of the unroll factor so both
    // halves take the same code path numpy's would.
    let mut n2 = n / 2;
    n2 -= n2 % PW_UNROLL;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

/// `np.mean(a)` for a CONTIGUOUS 1-D array — [`pairwise_sum`] divided by the
/// count.
///
/// ## Only for 1-D: an axis-0 reduction of a 2-D array is NOT pairwise
/// numpy applies the pairwise blocking to CONTIGUOUS reductions. Summing a
/// C-contiguous `(n, d)` array along axis 0 walks a stride-`d` axis, and numpy
/// instead vectorises ACROSS the `d` columns and accumulates row by row — a
/// plain SEQUENTIAL sum per column. So `np.nanstd(X, axis=0)` (what
/// `sklearn.preprocessing.scale` calls on a design matrix) is sequential per
/// column, while `np.nanstd(y)` on a 1-D target is pairwise, and reproducing
/// sklearn means using the matching form for each. Verified by comparing
/// `np.mean(np.abs(X), axis=0)` against both forms: the pairwise one disagrees in
/// the last bits on this crate's own oracle design.
///
/// Callers reducing a COLUMN of a 2-D design therefore use
/// [`sequential_sum`]-based helpers, not these.
pub fn numpy_mean(a: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::NAN;
    }
    pairwise_sum(a) / a.len() as f64
}

/// `np.add.reduce` along AXIS 0 of a C-contiguous 2-D array, for one column —
/// the plain sequential accumulation numpy uses there (see [`numpy_mean`]).
pub fn sequential_sum(a: &[f64]) -> f64 {
    let mut acc = 0.0;
    for &v in a {
        acc += v;
    }
    acc
}

/// `np.mean(X, axis=0)` for one column of a C-contiguous 2-D array.
pub fn numpy_mean_axis0(a: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::NAN;
    }
    sequential_sum(a) / a.len() as f64
}

/// `np.std(X, axis=0)` (`ddof = 0`) for one column of a C-contiguous 2-D array.
pub fn numpy_std_axis0(a: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::NAN;
    }
    let mean = numpy_mean_axis0(a);
    let dev: Vec<f64> = a.iter().map(|&v| (v - mean) * (v - mean)).collect();
    (sequential_sum(&dev) / a.len() as f64).sqrt()
}

/// `np.var(a)` (`ddof = 0`) — `numpy._methods._var`: the mean by
/// [`numpy_mean`], then the pairwise sum of the squared deviations, divided by
/// the count. Materialises the deviation vector because numpy does, and the
/// pairwise reduction is over that materialised array.
pub fn numpy_var(a: &[f64]) -> f64 {
    if a.is_empty() {
        return f64::NAN;
    }
    let mean = numpy_mean(a);
    let dev: Vec<f64> = a.iter().map(|&v| (v - mean) * (v - mean)).collect();
    pairwise_sum(&dev) / a.len() as f64
}

/// `np.std(a)` (`ddof = 0`) — the square root of [`numpy_var`].
pub fn numpy_std(a: &[f64]) -> f64 {
    numpy_var(a).sqrt()
}
