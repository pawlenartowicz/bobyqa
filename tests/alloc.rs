//! The M2 zero-alloc test (SPEC §6.4): after `Bobyqa::new`, repeated `minimize` calls —
//! the first included — perform **zero** heap allocations, on the trust-region warm path
//! and on the rarely-taken rescue path alike.
//!
//! The crate itself is `#![forbid(unsafe_code)]`; this dev-only `GlobalAlloc`
//! shim is why `unsafe_code = "forbid"` lives in `lib.rs` rather than the
//! package-wide `[lints]` table (design §8.2). Installed for this test binary
//! only.
//!
//! Everything lives in ONE `#[test]`: the counter is process-global, and libtest runs
//! `#[test]`s on parallel threads whose incidental allocations would pollute a concurrent
//! measurement. The one-shot `bobyqa(...)` wrapper allocates by design (its documented
//! contract) and gets no zero-alloc assertion here.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use bobyqa::{Bobyqa, Config, RestartConfig, Status};

/// Allocations observed since process start.
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

/// Wraps the system allocator and counts every `alloc` call (`realloc` routes through the
/// default `GlobalAlloc::realloc`, which calls `alloc` — growth is counted too).
struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `System` is a sound `GlobalAlloc`; `layout` is forwarded unchanged from this
        // `alloc` call, so the caller's `GlobalAlloc::alloc` guarantee (a valid, non-zero-size
        // layout) is exactly what `System.alloc` requires.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System.alloc` with this same `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

fn alloc_count() -> usize {
    ALLOCATIONS.load(Ordering::Relaxed)
}

/// Booth (the `booth_rescue` golden's objective) — arithmetic identical to
/// `tests/parity_prima.rs`; allocation-free, like every objective below.
fn booth(x: &[f64]) -> f64 {
    let a = x[0] + 2.0 * x[1] - 7.0;
    let b = 2.0 * x[0] + x[1] - 5.0;
    a * a + b * b
}

/// Rosenbrock — a single solve reaches `rho_end` well short of a tight optimum, so a
/// restart-enabled solver restarts on it (`tests/restart.rs` pins that); allocation-free.
fn rosenbrock(x: &[f64]) -> f64 {
    let (a, b) = (1.0 - x[0], x[1] - x[0] * x[0]);
    a * a + 100.0 * b * b
}

#[test]
fn minimize_allocates_zero_after_construction_on_warm_and_rescue_paths() {
    // Scaffold sanity (M0): the counter observes an allocation at all.
    let before = alloc_count();
    let v: Vec<u64> = Vec::with_capacity(32);
    assert!(
        alloc_count() > before,
        "Vec::with_capacity(32) did not bump the allocation counter"
    );
    drop(v);

    // Representative warm path: the sphere, three reuse calls on one solver. The counter must
    // be flat across ALL of them — the first call included (`new()` owns every allocation).
    let mut solver = Bobyqa::new(2, Config::new(2)).expect("valid config");
    let before = alloc_count();
    for call in 0..3 {
        let mut x = [1.0, 2.0]; // stack array — fresh start without heap traffic
        let o = solver.minimize(
            |p: &[f64]| p.iter().map(|v| v * v).sum::<f64>(),
            &mut x,
            &[-5.0, -5.0],
            &[5.0, 5.0],
        );
        assert_eq!(o.status, Status::Converged, "sphere call {call}");
        assert_eq!(
            alloc_count(),
            before,
            "sphere minimize allocated on call {call} (the zero-alloc warm path, SPEC §4)"
        );
    }

    // Rescue path: the `booth_rescue` golden's exact problem (booth, npt 5, rho 0.5 -> 1e-12,
    // x0 = 0, upper[1] = 2.5 pins the optimum to the bound; 40 evals, converged). The capture
    // was built as a rescue stressor, and the solver is deterministic, so this run takes the
    // rescue branch — proving it alloc-free too (M2 §6: zero means zero, not "zero on the
    // happy path"). The n_eval assert ties this run to the golden trajectory.
    let mut config = Config::new(2);
    config.npt = 5;
    config.rho_begin = 0.5;
    config.rho_end = 1e-12;
    config.max_fun = 500;
    let mut solver = Bobyqa::new(2, config).expect("valid config");
    let before = alloc_count();
    for call in 0..3 {
        let mut x = [0.0, 0.0]; // stack array — fresh start without heap traffic
        let o = solver.minimize(booth, &mut x, &[-10.0, -10.0], &[10.0, 2.5]);
        assert_eq!(o.status, Status::Converged, "booth_rescue call {call}");
        assert_eq!(
            o.n_eval, 40,
            "booth_rescue call {call} left the golden trajectory"
        );
        assert_eq!(
            alloc_count(),
            before,
            "rescue-path minimize allocated on call {call} (SPEC §6.4)"
        );
    }

    // Restart path: the same guarantee with restarts enabled. `Bobyqa::new` stays the sole
    // allocation site — the restart bookkeeping is a stack `RestartState`, the rebuild runs
    // `initxf` over the buffers `new` already sized for the opening solve, and the per-cycle
    // boundary store is pre-sized `max_restarts + 1` so filling it never reallocates. The
    // schedule below actually restarts on Rosenbrock, so the measured warm path includes a
    // whole rebuild and not merely the re-widening; the restart-count assert is what keeps
    // that true if the trajectory ever shifts. The stall trigger drives it rather than the
    // shipped default's eval cap, which on this budget may leave the cycle inside its
    // allowance and fire nothing.
    let mut restart = RestartConfig::new();
    restart.cycle_budget_frac = 0.0;
    restart.stall_reductions = 2;
    restart.max_restarts = 8;
    let mut config = Config::new(2);
    config.restart = Some(restart);
    let mut solver = Bobyqa::new(2, config).expect("valid config");
    let before = alloc_count();
    for call in 0..3 {
        let mut x = [-1.2, 1.0]; // stack array — fresh start without heap traffic
        let o = solver.minimize(rosenbrock, &mut x, &[-5.0, -5.0], &[5.0, 5.0]);
        assert_eq!(o.status, Status::Converged, "restart call {call}");
        assert!(
            solver.last_restart_count() >= 1,
            "restart call {call}: no restart fired, so the relocation path went unmeasured"
        );
        assert_eq!(
            solver.last_cycle_boundaries().len(),
            solver.last_restart_count(),
            "restart call {call}: one boundary per restart"
        );
        assert_eq!(
            alloc_count(),
            before,
            "restart-enabled minimize allocated on call {call} (the zero-alloc warm path \
             across the restart hook, point relocation, and the boundary store)"
        );
    }
}
