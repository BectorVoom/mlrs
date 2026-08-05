//! `abflag` — process A/B knobs with a THREAD-LOCAL test override.
//!
//! The prims carry a family of `MLRS_*` environment knobs that force a specific
//! kernel variant (`MLRS_TOPK_SERIAL`, `MLRS_DIST_RB2`, `MLRS_KNN_DOT`, …). They
//! exist so a kernel pair can be A/B'd on real target hardware without a
//! rebuild, and so the equivalence tests can drive each variant through the same
//! public entry point.
//!
//! ## Why the environment alone is not enough
//! Those two uses conflict. A benchmark sets the variable once for a whole
//! process; a TEST has to change it mid-process, and `libtest` runs a binary's
//! `#[test]`s on parallel threads. `std::env::set_var` from a test body is
//! therefore two bugs at once:
//!
//! 1. **A data race.** The dispatchers call `std::env::var` on every launch, so
//!    a sibling test is inside `getenv` while this one is inside `setenv` — a
//!    documented race on glibc's `environ` block, which is exactly why
//!    `set_var` is `unsafe` in Rust 2024. It can abort the test binary.
//! 2. **Silently vacuous assertions.** The variable is process-global, so a
//!    forced variant leaks into every other test running at that moment. An
//!    "these two kernels agree bitwise" test whose sibling has just forced one
//!    of them ends up comparing a kernel against ITSELF — it passes, and a real
//!    divergence ships.
//!
//! ## The fix
//! [`var`] reads a THREAD-LOCAL override first and the environment second. A
//! test scopes an override with [`force`]/[`clear`], which returns an RAII
//! [`Guard`] that restores the previous value on drop. The override is visible
//! only on the thread that set it — which is the thread that then calls the
//! prim — so sibling tests are unaffected, nothing touches `environ`, and the
//! benchmark path (no override set anywhere) behaves exactly as before.
//!
//! Prim dispatchers must read knobs through [`var`], never `std::env::var`
//! directly, or they opt back out of both properties.

use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// Per-thread knob overrides: `name -> Some(value)` forces a value,
    /// `name -> None` forces "unset" (so a test can assert the DEFAULT variant
    /// even when the ambient environment forces something else).
    static OVERRIDES: RefCell<HashMap<&'static str, Option<String>>> =
        RefCell::new(HashMap::new());
}

/// The value of A/B knob `name` for the CALLING THREAD: its thread-local
/// override if one is set, otherwise the process environment.
///
/// Drop-in for `std::env::var(name).ok()` in a prim dispatcher.
pub fn var(name: &'static str) -> Option<String> {
    if let Some(hit) = OVERRIDES.with(|o| o.borrow().get(name).cloned()) {
        return hit;
    }
    std::env::var(name).ok()
}

/// The calling thread's OVERRIDE for `name`, if any — the environment is NOT
/// consulted.
///
/// `Some(Some(v))` is a forced value, `Some(None)` a forced "unset", `None` no
/// override at all.
///
/// For dispatchers that CACHE their knob (because reading it per call would show
/// up in a hot loop): cache the environment half once, and consult this on every
/// call so a test's [`force`]/[`clear`] still takes effect. Without it a cached
/// knob makes an in-process A/B silently compare a variant against ITSELF — the
/// same vacuous-assertion failure the module docs describe, arrived at from the
/// other direction. It is cheap enough to sit in a dispatcher: one thread-local
/// borrow and a hash lookup, with no `getenv` and no allocation on the common
/// (no-override) path.
pub fn local_override(name: &'static str) -> Option<Option<String>> {
    OVERRIDES.with(|o| o.borrow().get(name).cloned())
}

/// Whether knob `name` is set to `"1"` on the calling thread — the shape most
/// of these knobs use (`MLRS_TOPK_SERIAL=1`, `MLRS_DIST_RB2=1`, …).
pub fn is_on(name: &'static str) -> bool {
    var(name).map(|v| v == "1").unwrap_or(false)
}

/// Force knob `name` to `value` on THIS THREAD until the returned [`Guard`] is
/// dropped. Test-only; production code sets these from the environment.
#[doc(hidden)]
pub fn force(name: &'static str, value: &str) -> Guard {
    set(name, Some(Some(value.to_string())))
}

/// Force knob `name` to be UNSET on this thread until the [`Guard`] drops, so a
/// test can pin the default variant regardless of the ambient environment.
#[doc(hidden)]
pub fn clear(name: &'static str) -> Guard {
    set(name, Some(None))
}

fn set(name: &'static str, value: Option<Option<String>>) -> Guard {
    let previous = OVERRIDES.with(|o| {
        let mut map = o.borrow_mut();
        let previous = map.remove(name);
        if let Some(v) = value {
            map.insert(name, v);
        }
        previous
    });
    Guard { name, previous }
}

/// Restores the knob override that was in effect before [`force`] / [`clear`].
///
/// Nested and interleaved guards restore correctly as long as they drop in
/// reverse order, which is what `let _g = ...;` in a test body gives.
#[doc(hidden)]
#[must_use = "the override is reverted as soon as the guard is dropped"]
pub struct Guard {
    name: &'static str,
    previous: Option<Option<String>>,
}

impl Drop for Guard {
    fn drop(&mut self) {
        OVERRIDES.with(|o| {
            let mut map = o.borrow_mut();
            match self.previous.take() {
                Some(v) => map.insert(self.name, v),
                None => map.remove(self.name),
            };
        });
    }
}
