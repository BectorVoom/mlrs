//! `host_pool` — the shared host worker-pool primitives the cpu arms build on.
//!
//! Every cpu campaign in this crate hits the same wall: `cubecl-cpu` maps ONE
//! OS THREAD PER UNIT, so a device launch is unusable for the small, repeated
//! passes an iterative fit makes, and the host replacement has to spread the
//! work itself. Doing that with `std::thread::scope` per pass is fine when
//! there is ONE pass (`linear_predict_host`) and a disaster when there are
//! dozens: `std::thread` setup is tens of microseconds per worker *when a core
//! is free*, and unbounded when one is not, so a pass that should cost 200 µs
//! costs milliseconds and a wider split makes it worse rather than better.
//!
//! This module owns the two pieces that fix that, both extracted from
//! [`hgb_host`](super::hgb_host) where they were first measured and tuned:
//!
//! - [`Barrier`] — a sense-reversing barrier with a spin → yield → BLOCK
//!   backoff, the synchronization a persistent pool needs.
//! - [`Shared`] — a `&mut [T]` handed across workers whose disjointness is
//!   enforced by the caller's task decomposition.
//!
//! plus one new piece:
//!
//! - [`WorkerPool`] — threads spawned ONCE and reused for many passes, each
//!   pass dispatched as a type-erased `Fn(usize)` the workers call with their
//!   own index. The erasure is per PASS, not per element: the closure the
//!   driver publishes captures the concrete element/loss types, so the inner
//!   loop is still monomorphized and vectorized and the pool costs exactly one
//!   indirect call per worker per pass.
//!
//! Tests live in `crates/mlrs-backend/tests/host_pool_test.rs` (AGENTS.md §2).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

// The pool itself is a cpu-arm construct: on the device backends the passes it
// would drive run as kernels, and nothing constructs one. `Barrier`/`Shared`
// are NOT gated — `hgb_host` builds on every backend.
#[cfg(feature = "cpu")]
use std::cell::UnsafeCell;
#[cfg(feature = "cpu")]
use std::panic::AssertUnwindSafe;
#[cfg(feature = "cpu")]
use std::marker::PhantomData;
#[cfg(feature = "cpu")]
use std::sync::Arc;

/// Sense-reversing barrier with a spin → yield → BLOCK backoff.
///
/// A fit crosses several barriers per pass — thousands per fit — so the fast
/// path has to be cheap: a waiter first spins (the phases it separates are tens
/// of microseconds) and then yields.
///
/// The third stage matters as much as the first two. A pure spin/yield barrier
/// is fine while the pool has a core per worker and CATASTROPHIC when it does
/// not: the waiters burn the very cores the stragglers need, and every barrier
/// costs a scheduler timeslice instead of microseconds. Measured on a
/// 16-core machine already ~70 % busy with unrelated work, spin/yield made a
/// 16-worker fit SLOWER than a 1-worker fit. So after the yield budget a
/// waiter parks on a condvar, handing its core back; an oversubscribed pool
/// then degrades gracefully to "no faster than the free cores allow" instead
/// of collapsing.
pub(crate) struct Barrier {
    workers: usize,
    /// Yields this waiter burns before parking — see [`YIELD_ITERS`] for why
    /// the right budget depends on how long the phases being separated are.
    yield_iters: u32,
    arrived: AtomicUsize,
    epoch: AtomicUsize,
    /// Set when a worker unwinds. Every worker's control flow is identical, so
    /// one that dies never reaches its next barrier and the survivors would
    /// wait forever — a HANG, which is a far worse failure than the panic that
    /// caused it. Waiters therefore also watch this flag and bail out, and the
    /// caller re-raises on the driving thread.
    poisoned: AtomicBool,
    /// Blocking stage. `epoch` is republished under this lock so a waiter that
    /// has taken it either observes the new epoch or is parked on `cv` when
    /// the release happens — no lost wakeup.
    lock: std::sync::Mutex<()>,
    cv: std::sync::Condvar,
}

impl Barrier {
    /// A barrier tuned for SHORT phases (tens of microseconds), where parking
    /// costs more than the phase itself — the `hgb_host` level pipeline.
    pub(crate) fn new(workers: usize) -> Self {
        Self::with_yield_budget(workers, YIELD_ITERS)
    }

    /// A barrier with an explicit yield budget, for phases long enough that
    /// handing the core back is cheap relative to them (see [`YIELD_ITERS`]).
    pub(crate) fn with_yield_budget(workers: usize, yield_iters: u32) -> Self {
        Self {
            workers,
            yield_iters,
            arrived: AtomicUsize::new(0),
            epoch: AtomicUsize::new(0),
            poisoned: AtomicBool::new(false),
            lock: std::sync::Mutex::new(()),
            cv: std::sync::Condvar::new(),
        }
    }

    /// Block until all `workers` have arrived. Returns `false` once the pool is
    /// poisoned, which unwinds the caller's loop instead of waiting for a
    /// worker that will never arrive.
    pub(crate) fn wait(&self) -> bool {
        if self.workers <= 1 {
            return !self.poisoned.load(Ordering::Relaxed);
        }
        let epoch = self.epoch.load(Ordering::Relaxed);
        if self.arrived.fetch_add(1, Ordering::AcqRel) + 1 == self.workers {
            // Last in: reset the counter BEFORE publishing the new epoch, so a
            // worker released here cannot re-enter and observe a stale count.
            self.arrived.store(0, Ordering::Relaxed);
            {
                let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
                self.epoch.store(epoch.wrapping_add(1), Ordering::Release);
            }
            self.cv.notify_all();
            return !self.poisoned.load(Ordering::Relaxed);
        }

        // Stage 1: spin — the common case is a few microseconds away.
        for _ in 0..SPIN_ITERS {
            if self.released(epoch) {
                return !self.poisoned.load(Ordering::Relaxed);
            }
            std::hint::spin_loop();
        }
        // Stage 2: yield — still cheap, and enough when the pool is merely a
        // little wider than the free cores.
        for _ in 0..self.yield_iters {
            if self.released(epoch) {
                return !self.poisoned.load(Ordering::Relaxed);
            }
            std::thread::yield_now();
        }
        // Stage 3: park, giving the core back to whoever is actually running.
        let mut guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        while !self.released(epoch) {
            let (g, _) = self
                .cv
                .wait_timeout(guard, std::time::Duration::from_millis(2))
                .unwrap_or_else(|e| e.into_inner());
            guard = g;
        }
        drop(guard);
        !self.poisoned.load(Ordering::Relaxed)
    }

    #[inline(always)]
    fn released(&self, epoch: usize) -> bool {
        self.epoch.load(Ordering::Acquire) != epoch || self.poisoned.load(Ordering::Relaxed)
    }

    pub(crate) fn poison(&self) {
        self.poisoned.store(true, Ordering::Relaxed);
        let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        drop(_g);
        self.cv.notify_all();
    }

    /// Whether [`poison`](Self::poison) has been called. Read by
    /// [`WorkerPool`]'s `Drop`, which must NOT release a poisoned pool's
    /// workers again — they are already unwinding out of their loop, so the
    /// extra barrier crossing would never complete.
    #[cfg(feature = "cpu")]
    pub(crate) fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Relaxed)
    }
}

/// Spin iterations before a waiter starts yielding (see [`Barrier::wait`]).
const SPIN_ITERS: u32 = 4096;

/// Default yields before a waiter parks (see [`Barrier::wait`]).
///
/// Deliberately LARGE, and correct only for SHORT phases. Measured on the
/// `hgb_host` level pipeline, whose phases are tens of microseconds and whose
/// barriers fire thousands of times per fit: parking early was a big net loss —
/// condvar wake latency is tens of microseconds, so an eager third stage erased
/// the pool's speedup entirely (a 2-worker fit came out slower than a 1-worker
/// one). Yielding is the stage that actually carries an oversubscribed pool;
/// parking exists only so a pathologically starved worker cannot spin a core
/// forever.
///
/// A pool whose phases are MILLISECONDS wants the opposite trade and should
/// pass its own budget to [`Barrier::with_yield_budget`]: there the wake
/// latency is noise, while 100 000 yields against a busy run queue is not.
const YIELD_ITERS: u32 = 100_000;

/// The yield budget for pools whose phases are milliseconds rather than
/// microseconds — long enough that a condvar wake is a rounding error, so a
/// waiter should hand its core back quickly instead of fighting the run queue
/// for it. Used by [`WorkerPool`].
#[cfg(feature = "cpu")]
pub(crate) const YIELD_ITERS_LONG_PHASE: u32 = 256;

/// A `&mut [T]` shared across the worker pool, whose disjointness is enforced
/// by the caller's task decomposition rather than by the borrow checker.
///
/// ## Safety contract (upheld by every use of this type)
/// Within one phase (the span between two [`Barrier::wait`] calls) each worker
/// writes only the elements its own task indices own, and those index sets are
/// pairwise DISJOINT — exactly the property the corresponding device kernels
/// rely on to be race-free without atomics ("each unit writes only memory it
/// exclusively owns", `gbt.rs` module doc). A value written in one phase is
/// read in a LATER phase, and the barrier between them is an
/// `Acquire`/`Release` pair, so the write is visible.
pub(crate) struct Shared<T> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: `Shared` is only ever handed to workers that outlive neither the
// backing allocation nor the barrier discipline documented on the type.
unsafe impl<T: Send> Send for Shared<T> {}
unsafe impl<T: Send> Sync for Shared<T> {}

impl<T> Copy for Shared<T> {}
impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Shared<T> {
    pub(crate) fn new(v: &mut [T]) -> Self {
        Self {
            ptr: v.as_mut_ptr(),
            len: v.len(),
        }
    }

    /// Immutable view of the whole buffer (values written before the last
    /// barrier).
    #[inline(always)]
    pub(crate) fn get(&self) -> &[T] {
        // SAFETY: see the type-level contract.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Mutable view of the whole buffer; the caller writes only its own
    /// task-owned indices.
    #[inline(always)]
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn get_mut(&self) -> &mut [T] {
        // SAFETY: see the type-level contract.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

// ---------------------------------------------------------------------------
// WorkerPool
// ---------------------------------------------------------------------------

/// A type-erased pass: the driver's closure as a data pointer plus the
/// monomorphized stub that calls it.
///
/// Two words, so it cannot live in a single atomic — it is published into an
/// [`UnsafeCell`] BEFORE the release of `start`, and read by every worker only
/// AFTER that same barrier's acquire. The barrier's `Release`/`Acquire` pair is
/// what makes the write visible; there is no data race because no worker is
/// running between the two.
#[cfg(feature = "cpu")]
#[derive(Clone, Copy)]
struct RawTask {
    data: *const (),
    call: unsafe fn(*const (), usize),
}

// SAFETY: `data` points at a closure the driver keeps alive across the whole
// pass (it is borrowed for the duration of `WorkerPool::run`), and the closure
// is required to be `Sync` by `run`'s bound.
#[cfg(feature = "cpu")]
unsafe impl Send for RawTask {}
#[cfg(feature = "cpu")]
unsafe impl Sync for RawTask {}

/// Call `data` — which is a `*const T` — with the worker's index.
///
/// # Safety
/// `data` must point at a live `T`, and `T` must be the type this stub was
/// instantiated for.
#[cfg(feature = "cpu")]
unsafe fn call_stub<T: Fn(usize) + Sync>(data: *const (), unit: usize) {
    unsafe { (*data.cast::<T>())(unit) }
}

/// Shared control block: the two barriers, the published pass, and the
/// shutdown flag.
#[cfg(feature = "cpu")]
struct PoolCtl {
    /// Released when a pass is ready to run (or the pool is shutting down).
    start: Barrier,
    /// Released when every participant has finished the pass.
    done: Barrier,
    /// The pass to run, valid between `start`'s release and `done`'s.
    task: UnsafeCell<Option<RawTask>>,
    /// Set before the final `start` release; tells workers to exit instead of
    /// running a pass.
    shutdown: AtomicBool,
}

// SAFETY: `task` is written only by the driver while every worker is blocked on
// `start`, and read only by workers between `start`'s release and `done`'s —
// the barriers serialize the two, so the `UnsafeCell` is never aliased.
#[cfg(feature = "cpu")]
unsafe impl Sync for PoolCtl {}
#[cfg(feature = "cpu")]
unsafe impl Send for PoolCtl {}

/// Worker threads spawned ONCE and reused for many passes.
///
/// The driving thread participates as unit `0`, so a pool of `units` runs
/// `units - 1` spawned threads. `units <= 1` spawns nothing and
/// [`run`](Self::run) simply calls the closure inline — the small-input path
/// pays no synchronization at all.
///
/// ## Why this and not `std::thread::scope` per pass
/// See the module docs: an iterative fit makes dozens of identical passes, and
/// re-spawning for each of them is the dominant cost at every size where the
/// pass itself is not already several milliseconds. With the pool, a pass costs
/// two barrier crossings (microseconds) regardless of how many times it runs.
#[cfg(feature = "cpu")]
pub(crate) struct WorkerPool {
    units: usize,
    ctl: Arc<PoolCtl>,
    handles: Vec<std::thread::JoinHandle<()>>,
    /// Makes the pool `!Sync`, so `run`'s `&self` cannot be shared across
    /// threads.
    ///
    /// This is a soundness requirement, not a style choice: `run` publishes the
    /// pass into `PoolCtl::task` and then crosses `start`. Two DRIVERS calling
    /// it concurrently would race on that slot and could release the workers
    /// against a task the other thread had already overwritten. There is
    /// exactly one driver by construction — the thread that built the pool —
    /// and this marker is what makes the compiler enforce it.
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

#[cfg(feature = "cpu")]
impl WorkerPool {
    /// Spawn a pool with `units` participants (the driver plus `units - 1`
    /// threads). `units <= 1` is the inline, thread-free pool.
    pub(crate) fn new(units: usize) -> Self {
        let units = units.max(1);
        let ctl = Arc::new(PoolCtl {
            start: Barrier::with_yield_budget(units, YIELD_ITERS_LONG_PHASE),
            done: Barrier::with_yield_budget(units, YIELD_ITERS_LONG_PHASE),
            task: UnsafeCell::new(None),
            shutdown: AtomicBool::new(false),
        });
        let mut handles = Vec::with_capacity(units.saturating_sub(1));
        for unit in 1..units {
            let ctl = Arc::clone(&ctl);
            handles.push(
                std::thread::Builder::new()
                    .name(format!("mlrs-host-pool-{unit}"))
                    .spawn(move || worker_loop(&ctl, unit))
                    .expect("failed to spawn a host worker-pool thread"),
            );
        }
        Self {
            units,
            ctl,
            handles,
            _not_sync: PhantomData,
        }
    }

    /// Participants in this pool, INCLUDING the driving thread.
    pub(crate) fn units(&self) -> usize {
        self.units
    }

    /// Run one pass: every unit `u` in `0..units` calls `f(u)` exactly once,
    /// and `run` returns only after all of them have.
    ///
    /// `f` is borrowed for the whole pass and must be `Sync` because the
    /// workers call it concurrently. It is dispatched through ONE indirect call
    /// per worker per pass, so whatever `f` does internally is still fully
    /// monomorphized and inlined.
    ///
    /// # Panics
    /// Re-raises on the driving thread if any unit's `f` panicked, rather than
    /// leaving the survivors waiting at a barrier that can never complete.
    pub(crate) fn run<T: Fn(usize) + Sync>(&self, f: &T) {
        if self.units <= 1 {
            f(0);
            return;
        }
        // Published while every worker is blocked on `start` (module docs).
        // SAFETY: no worker can observe `task` until `start` releases below.
        unsafe {
            *self.ctl.task.get() = Some(RawTask {
                data: (f as *const T).cast::<()>(),
                call: call_stub::<T>,
            });
        }
        if !self.ctl.start.wait() {
            panic!("mlrs host worker pool: a worker panicked in an earlier pass");
        }
        // The driver runs unit 0's share itself. A panic here would strand the
        // workers at `done`, so it is caught, the barriers poisoned (which
        // releases them), and then re-raised.
        let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| f(0)));
        match outcome {
            Ok(()) => {
                if !self.ctl.done.wait() {
                    panic!("mlrs host worker pool: a worker panicked during the pass");
                }
            }
            Err(payload) => {
                self.ctl.start.poison();
                self.ctl.done.poison();
                std::panic::resume_unwind(payload);
            }
        }
    }
}

#[cfg(feature = "cpu")]
impl Drop for WorkerPool {
    fn drop(&mut self) {
        if self.handles.is_empty() {
            return;
        }
        // A poisoned pool's workers are already unwinding out of their loop —
        // releasing `start` again would block forever, so only join.
        if !self.ctl.start.is_poisoned() && !self.ctl.done.is_poisoned() {
            self.ctl.shutdown.store(true, Ordering::Release);
            self.ctl.start.wait();
        }
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

/// One worker's lifetime: wait for a pass, run its share, report, repeat.
#[cfg(feature = "cpu")]
fn worker_loop(ctl: &PoolCtl, unit: usize) {
    loop {
        if !ctl.start.wait() {
            return;
        }
        if ctl.shutdown.load(Ordering::Acquire) {
            return;
        }
        // SAFETY: published by the driver before `start` released, and not
        // rewritten until after `done` releases (module docs).
        let task = match unsafe { *ctl.task.get() } {
            Some(t) => t,
            None => {
                ctl.start.poison();
                ctl.done.poison();
                return;
            }
        };
        // SAFETY: `task.data` points at the driver's live closure of the type
        // `task.call` was instantiated for.
        let outcome =
            std::panic::catch_unwind(AssertUnwindSafe(|| unsafe { (task.call)(task.data, unit) }));
        if outcome.is_err() {
            // Never wait at `done`: poisoning both barriers is what lets the
            // driver observe the failure instead of hanging.
            ctl.start.poison();
            ctl.done.poison();
            return;
        }
        if !ctl.done.wait() {
            return;
        }
    }
}
