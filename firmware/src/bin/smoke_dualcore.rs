//! Dual-core smoke test (de-risk for the multicore inference work).
//!
//! Starts the ESP32-S3 APP core (core 1) via `esp_hal`'s `CpuControl`, runs a
//! compute-bound workload first on one core then split across both, and prints
//! correctness + speedup. This validates the toolchain/API and the cross-core
//! atomic handshake BEFORE wiring parallelism into microflow's kernels.
//! See microflow `docs/ESP32_S3_MULTICORE.md`.

#![no_std]
#![no_main]

use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use esp_backtrace as _;
use esp_hal::{
    clock::CpuClock,
    delay::Delay,
    main,
    system::{CpuControl, Stack},
    time::Instant,
};
use esp_println::println;

esp_bootloader_esp_idf::esp_app_desc!();

/// APP-core (core 1) stack. Small for this smoke test; the real inference worker
/// will need much more and must be sized against the 320KB stack wall
/// (see docs/ESP32_S3_MULTICORE.md §6.1).
static mut APP_CORE_STACK: Stack<8192> = Stack::new();

// Cross-core handshake. All in internal SRAM → directly shared by both cores,
// no cache flush needed; atomics provide ordering, not coherence.
static C1_READY: AtomicBool = AtomicBool::new(false);
static GO: AtomicBool = AtomicBool::new(false);
static C1_DONE: AtomicBool = AtomicBool::new(false);
static C1_RESULT: AtomicU32 = AtomicU32::new(0);

const N: u32 = 2_000_000;
const INNER: u32 = 48;

/// Compute-bound workload over `[lo, hi)`: sum (wrapping) of an LCG-iterated
/// value per index. Associative across a split:
/// `work(0, mid) + work(mid, N) == work(0, N)` (wrapping), so the dual-core
/// result must equal the single-core result bit-for-bit.
///
/// `black_box` on each step prevents LLVM from collapsing the affine LCG
/// composition into a single mul-add (which would optimize the workload away
/// and make the timing meaningless).
#[inline(never)]
fn work(lo: u32, hi: u32) -> u32 {
    let mut acc: u32 = 0;
    let mut i = lo;
    while i < hi {
        let mut x = i;
        let mut k = 0;
        while k < INNER {
            x = core::hint::black_box(x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223));
            k += 1;
        }
        acc = acc.wrapping_add(x);
        i += 1;
    }
    acc
}

#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);
    let delay = Delay::new();

    println!("System Init. Clock: Max. Dual-core smoke test.");

    let mid = N / 2;

    // --- single-core baseline ---
    let t0 = Instant::now();
    let single = work(0, N);
    let single_us = t0.elapsed().as_micros();
    println!("SINGLE_CORE: result={} us={}", single, single_us);

    // --- start the APP core into a worker ---
    let mut cpu_control = CpuControl::new(peripherals.CPU_CTRL);
    let worker = || {
        // core 1
        C1_READY.store(true, Ordering::Release);
        while !GO.load(Ordering::Acquire) {
            core::hint::spin_loop();
        }
        let r = work(mid, N);
        C1_RESULT.store(r, Ordering::Relaxed);
        C1_DONE.store(true, Ordering::Release);
        // One-shot for this smoke test; idle (the guard parks it on drop).
        loop {
            core::hint::spin_loop();
        }
    };
    let _guard = cpu_control
        .start_app_core(
            unsafe { &mut *core::ptr::addr_of_mut!(APP_CORE_STACK) },
            worker,
        )
        .unwrap();

    // Wait until core 1 has booted and is poised on GO, so its startup latency
    // is excluded from the timed region.
    while !C1_READY.load(Ordering::Acquire) {
        core::hint::spin_loop();
    }

    // --- dual-core run ---
    let t1 = Instant::now();
    GO.store(true, Ordering::Release); // release core 1
    let half0 = work(0, mid); // core 0 does its half in parallel
    while !C1_DONE.load(Ordering::Acquire) {
        core::hint::spin_loop();
    }
    let dual = half0.wrapping_add(C1_RESULT.load(Ordering::Relaxed));
    let dual_us = t1.elapsed().as_micros();
    println!("DUAL_CORE:   result={} us={}", dual, dual_us);

    println!(
        "MATCH:{} SPEEDUP_x1000:{}",
        single == dual,
        if dual_us > 0 { (single_us * 1000) / dual_us } else { 0 }
    );
    println!("SMOKE_DONE");

    loop {
        delay.delay_millis(1000);
    }
}
