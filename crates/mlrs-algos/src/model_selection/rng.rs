//! Bit-for-bit reimplementation of numpy's **legacy** `RandomState` MT19937
//! (MODSEL-RS-01).
//!
//! Every randomized splitter in `sklearn.model_selection` resolves its
//! `random_state` through `sklearn.utils.check_random_state`, which produces a
//! `numpy.random.RandomState` — the *legacy* Mersenne-Twister generator, NOT
//! the modern `Generator`/PCG64 one. mlrs's recorded parity decision for the
//! whole `model_selection` surface is **host-match**: for the same arguments,
//! an mlrs splitter must select the *same rows* as sklearn, index for index,
//! not merely a same-sized/same-balance split. That is only achievable by
//! reproducing the exact draw sequence, so this module reimplements the three
//! numpy entry points sklearn's splitters actually reach:
//!
//! | numpy call                    | this module                         |
//! |-------------------------------|-------------------------------------|
//! | `RandomState(seed)`           | [`NumpyRandomState::from_seed`]      |
//! | `rng.shuffle(a)`              | [`NumpyRandomState::shuffle`]        |
//! | `rng.permutation(n)`          | [`NumpyRandomState::permutation`]    |
//! | `rng.randint(n)` / `.choice`  | [`NumpyRandomState::randint`]        |
//! | `rng.random_sample()`         | [`NumpyRandomState::random_sample`]  |
//!
//! plus sklearn's own `sklearn.utils.random.sample_without_replacement`
//! ([`sample_without_replacement`]), which `ParameterSampler` uses when every
//! search value is a list.
//!
//! ## Why `randint` and `shuffle` share one primitive
//!
//! numpy's `shuffle` draws its Fisher-Yates partner index with
//! `random_interval(i)` (masked rejection sampling: mask the smallest
//! `2^k - 1 >= i` off a fresh 32-bit word, redraw while the value exceeds
//! `i`). `randint(0, n)` reaches `random_bounded_uint64_fill` with
//! `rng = n - 1` and `use_masked = true`, whose 32-bit branch
//! (`bounded_masked_uint32`) is the *identical* loop. So both are
//! [`NumpyRandomState::bounded`] here, and a bug in one cannot silently
//! disagree with the other.
//!
//! ## State round-tripping (why `from_key` / `key` / `pos` are public)
//!
//! sklearn's `check_random_state` passes a **live** `RandomState` object
//! through — `RepeatedKFold` seeds one rng and hands the same object to every
//! repeat, and `ParameterSampler` interleaves its own draws with
//! `scipy.stats` `rvs(random_state=rng)` calls that mlrs cannot host. So the
//! PyO3 layer round-trips the raw MT19937 words: it reads `rs.get_state()`,
//! runs the Rust splitter from those words, and writes the advanced words back
//! with `rs.set_state(...)`. The caller's `RandomState` therefore ends up in
//! exactly the state sklearn would have left it in, and a Python-side scipy
//! draw interleaved between two Rust calls sees the right stream.
//!
//! Tests live in `crates/mlrs-algos/tests/model_selection_rng_test.rs`
//! (no in-source `#[cfg(test)] mod tests`).

/// Number of 32-bit words in the MT19937 state vector.
pub const MT_N: usize = 624;
/// MT19937 recurrence offset.
const MT_M: usize = 397;
/// MT19937 twist matrix constant.
const MATRIX_A: u32 = 0x9908_b0df;
/// Most significant bit.
const UPPER_MASK: u32 = 0x8000_0000;
/// Least significant 31 bits.
const LOWER_MASK: u32 = 0x7fff_ffff;

/// numpy's legacy `numpy.random.RandomState` generator (MT19937).
///
/// Holds exactly the state numpy's `get_state()` exposes for the
/// `"MT19937"` bit generator: the 624-word key and the read position. The
/// `has_gauss` / `cached_gaussian` fields of `get_state()` are *not* modelled —
/// no `model_selection` code path draws a Gaussian, so they are never
/// consumed, and the PyO3 layer preserves the caller's values untouched.
#[derive(Clone)]
pub struct NumpyRandomState {
    key: [u32; MT_N],
    pos: usize,
}

impl core::fmt::Debug for NumpyRandomState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The 624-word key is noise in a failure message; the position and the
        // first/last words are enough to tell two states apart.
        f.debug_struct("NumpyRandomState")
            .field("pos", &self.pos)
            .field("key[0]", &self.key[0])
            .field("key[623]", &self.key[MT_N - 1])
            .finish()
    }
}

impl NumpyRandomState {
    /// Seed exactly as `numpy.random.RandomState(seed)` does for a scalar
    /// integer seed (`mt19937_seed`).
    ///
    /// numpy rejects a seed outside `[0, 2**32 - 1]` with `ValueError` before
    /// reaching this initializer, so the `u32` parameter type *is* the
    /// validation — the PyO3 boundary rejects anything wider with sklearn's
    /// own message.
    ///
    /// The recurrence is randomkit's variant of `init_genrand`: it stores the
    /// current word and then advances, which is algebraically identical to the
    /// reference `mt[i] = 1812433253 * (mt[i-1] ^ (mt[i-1] >> 30)) + i`.
    pub fn from_seed(seed: u32) -> Self {
        let mut key = [0u32; MT_N];
        let mut s = seed;
        for (pos, slot) in key.iter_mut().enumerate() {
            *slot = s;
            s = (1_812_433_253u32.wrapping_mul(s ^ (s >> 30))).wrapping_add(pos as u32 + 1);
        }
        Self { key, pos: MT_N }
    }

    /// Rebuild a generator from raw `get_state()` words.
    ///
    /// `pos` is clamped to `MT_N` rather than rejected: numpy stores the
    /// position as a plain Python int and a `set_state` round trip through a
    /// pickle can legitimately carry `pos == MT_N` (meaning "regenerate on the
    /// next draw"), which is the value [`from_seed`](Self::from_seed) itself
    /// produces.
    pub fn from_key(key: [u32; MT_N], pos: usize) -> Self {
        Self {
            key,
            pos: pos.min(MT_N),
        }
    }

    /// The raw state words, for writing back into a caller's `RandomState`.
    pub fn key(&self) -> &[u32; MT_N] {
        &self.key
    }

    /// The read position, for writing back into a caller's `RandomState`.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Regenerate the whole 624-word block (numpy's `mt19937_gen`).
    fn generate(&mut self) {
        let mt = &mut self.key;
        for i in 0..(MT_N - MT_M) {
            let y = (mt[i] & UPPER_MASK) | (mt[i + 1] & LOWER_MASK);
            mt[i] = mt[i + MT_M] ^ (y >> 1) ^ ((y & 1).wrapping_mul(MATRIX_A));
        }
        for i in (MT_N - MT_M)..(MT_N - 1) {
            let y = (mt[i] & UPPER_MASK) | (mt[i + 1] & LOWER_MASK);
            mt[i] = mt[i + MT_M - MT_N] ^ (y >> 1) ^ ((y & 1).wrapping_mul(MATRIX_A));
        }
        let y = (mt[MT_N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[MT_N - 1] = mt[MT_M - 1] ^ (y >> 1) ^ ((y & 1).wrapping_mul(MATRIX_A));
        self.pos = 0;
    }

    /// One tempered 32-bit word (numpy's `mt19937_next`).
    pub fn next_u32(&mut self) -> u32 {
        if self.pos >= MT_N {
            self.generate();
        }
        let mut y = self.key[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// One 64-bit word (numpy's `mt19937_next64`: high word drawn first).
    pub fn next_u64(&mut self) -> u64 {
        let hi = u64::from(self.next_u32());
        let lo = u64::from(self.next_u32());
        (hi << 32) | lo
    }

    /// numpy's `random_sample()` — a double in `[0, 1)` built from 53 bits.
    pub fn random_sample(&mut self) -> f64 {
        let a = f64::from(self.next_u32() >> 5);
        let b = f64::from(self.next_u32() >> 6);
        (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }

    /// The smallest `2^k - 1` that covers `max` (numpy's `gen_mask`).
    fn gen_mask(max: u64) -> u64 {
        let mut mask = max;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        mask |= mask >> 32;
        mask
    }

    /// A uniform draw from the **inclusive** range `[0, max]` by masked
    /// rejection — numpy's `random_interval`, and equally the body of
    /// `bounded_masked_uint32`/`_uint64` that backs `randint`.
    ///
    /// The 32-bit branch is not an optimization: numpy takes it whenever
    /// `max <= 0xFFFFFFFF`, and it consumes *one* word per attempt where the
    /// 64-bit branch consumes two, so drawing from the wrong branch desyncs
    /// the whole downstream stream.
    pub fn bounded(&mut self, max: u64) -> u64 {
        if max == 0 {
            return 0;
        }
        let mask = Self::gen_mask(max);
        if max <= u64::from(u32::MAX) {
            let mask = mask as u32;
            loop {
                let value = u64::from(self.next_u32() & mask);
                if value <= max {
                    return value;
                }
            }
        } else {
            loop {
                let value = self.next_u64() & mask;
                if value <= max {
                    return value;
                }
            }
        }
    }

    /// numpy's `rng.randint(n)` — a uniform draw from the **half-open** range
    /// `[0, n)`.
    ///
    /// # Panics
    /// Panics if `n == 0`, mirroring numpy's `ValueError: low >= high`. Every
    /// caller in this crate draws from a non-empty population.
    pub fn randint(&mut self, n: u64) -> u64 {
        assert!(n > 0, "randint: empty range");
        self.bounded(n - 1)
    }

    /// numpy's `rng.shuffle(a)` — in-place Fisher-Yates walking *downwards*.
    ///
    /// The direction and the inclusive `bounded(i)` partner draw are both
    /// load-bearing: an upward pass, or an exclusive `randint(i + 1)`, would
    /// produce a valid permutation from a different point in the same stream
    /// and silently break parity everywhere.
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        if xs.len() < 2 {
            return;
        }
        for i in (1..xs.len()).rev() {
            let j = self.bounded(i as u64) as usize;
            xs.swap(i, j);
        }
    }

    /// numpy's `rng.permutation(n)` — `arange(n)` then [`shuffle`](Self::shuffle).
    pub fn permutation(&mut self, n: usize) -> Vec<i64> {
        let mut out: Vec<i64> = (0..n as i64).collect();
        self.shuffle(&mut out);
        out
    }

    /// numpy's `rng.permutation(a)` for an existing array — a shuffled copy.
    pub fn permutation_of<T: Clone>(&mut self, xs: &[T]) -> Vec<T> {
        let mut out = xs.to_vec();
        self.shuffle(&mut out);
        out
    }
}

/// How `sample_without_replacement` picks its algorithm — sklearn's `method`
/// parameter (`sklearn.utils.random.sample_without_replacement`).
///
/// The choice is NOT a performance detail: each algorithm consumes the MT19937
/// stream differently, so two methods return different (equally valid) samples
/// from the same seed. `Auto` reproduces sklearn's dispatch exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleMethod {
    Auto,
    TrackingSelection,
    ReservoirSampling,
    Pool,
}

/// sklearn's `sample_without_replacement(n_population, n_samples, method,
/// random_state)` — `ParameterSampler`'s draw when every search value is a
/// list.
///
/// `Auto` follows sklearn's two-level dispatch: a *middling* sampling ratio
/// (strictly between 0.01 and 0.99) short-circuits to
/// `permutation(n_population)[:n_samples]`; otherwise a ratio below 0.2 uses
/// tracking selection and anything else uses reservoir sampling.
///
/// Returns `None` when the request is impossible (`n_samples > n_population`,
/// or either argument negative — both `ValueError` in sklearn); the caller
/// raises with sklearn's message.
pub fn sample_without_replacement(
    n_population: usize,
    n_samples: usize,
    method: SampleMethod,
    rng: &mut NumpyRandomState,
) -> Option<Vec<i64>> {
    if n_samples > n_population {
        return None;
    }
    if n_samples == 0 {
        return Some(Vec::new());
    }

    let resolved = match method {
        SampleMethod::Auto => {
            let ratio = n_samples as f64 / n_population as f64;
            if ratio > 0.01 && ratio < 0.99 {
                let mut perm = rng.permutation(n_population);
                perm.truncate(n_samples);
                return Some(perm);
            }
            if ratio < 0.2 {
                SampleMethod::TrackingSelection
            } else {
                SampleMethod::ReservoirSampling
            }
        }
        other => other,
    };

    let out = match resolved {
        SampleMethod::TrackingSelection => {
            // sklearn's set-tracking loop, redrawing on a collision. A HashSet
            // would change nothing about the stream, but a sorted Vec of the
            // (few, by construction — this branch only runs at ratio < 0.2)
            // selected values keeps the allocation flat.
            let mut selected: std::collections::HashSet<u64> =
                std::collections::HashSet::with_capacity(n_samples);
            let mut out = Vec::with_capacity(n_samples);
            for _ in 0..n_samples {
                let mut j = rng.randint(n_population as u64);
                while selected.contains(&j) {
                    j = rng.randint(n_population as u64);
                }
                selected.insert(j);
                out.push(j as i64);
            }
            out
        }
        SampleMethod::ReservoirSampling => {
            let mut out: Vec<i64> = (0..n_samples as i64).collect();
            for i in n_samples..n_population {
                let j = rng.randint(i as u64 + 1) as usize;
                if j < n_samples {
                    out[j] = i as i64;
                }
            }
            out
        }
        SampleMethod::Pool => {
            let mut pool: Vec<i64> = (0..n_population as i64).collect();
            let mut out = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let j = rng.randint((n_population - i) as u64) as usize;
                out.push(pool[j]);
                pool[j] = pool[n_population - i - 1];
            }
            out
        }
        SampleMethod::Auto => unreachable!("Auto is resolved above"),
    };
    Some(out)
}

/// sklearn's `_approximate_mode(class_counts, n_draws, rng)` — the mode of a
/// multivariate hypergeometric, used by `StratifiedShuffleSplit` to decide how
/// many rows of each class land in the train (then the test) side.
///
/// The remainder ties are broken by drawing `add_now` of the tied positions
/// *without replacement*, which numpy's legacy `rng.choice(..., replace=False)`
/// implements as `permutation(len(inds))[:add_now]` — so the tie-break burns
/// stream in a way a deterministic "take the first" would not.
pub fn approximate_mode(
    class_counts: &[i64],
    n_draws: i64,
    rng: &mut NumpyRandomState,
) -> Vec<i64> {
    let total: i64 = class_counts.iter().sum();
    let continuous: Vec<f64> = class_counts
        .iter()
        .map(|&c| c as f64 / total as f64 * n_draws as f64)
        .collect();
    let mut floored: Vec<f64> = continuous.iter().map(|v| v.floor()).collect();
    let mut need_to_add = (n_draws as f64 - floored.iter().sum::<f64>()) as i64;

    if need_to_add > 0 {
        let remainder: Vec<f64> = continuous
            .iter()
            .zip(&floored)
            .map(|(c, f)| c - f)
            .collect();
        // `np.sort(np.unique(remainder))[::-1]` — descending distinct values.
        // Bit-equality is the right comparison here: sklearn groups by the
        // exact float, and two remainders that differ in the last ulp are two
        // groups on both sides.
        let mut values: Vec<f64> = remainder.clone();
        values.sort_by(|a, b| a.partial_cmp(b).expect("remainders are finite"));
        values.dedup();
        values.reverse();

        for value in values {
            let inds: Vec<usize> = remainder
                .iter()
                .enumerate()
                .filter(|(_, r)| **r == value)
                .map(|(i, _)| i)
                .collect();
            let add_now = inds.len().min(need_to_add as usize);
            // legacy `rng.choice(inds, size=add_now, replace=False)`
            let perm = rng.permutation(inds.len());
            for &p in perm.iter().take(add_now) {
                floored[inds[p as usize]] += 1.0;
            }
            need_to_add -= add_now as i64;
            if need_to_add == 0 {
                break;
            }
        }
    }

    floored.iter().map(|v| *v as i64).collect()
}
