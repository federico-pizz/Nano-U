//! Isolation test for `microflow::multicore::parallel_for_rows`.
//!
//! The nano_u dual-core inference overflowed core 0's stack (the multicore
//! depthwise path), and that overflow corrupted the dispatch statics, surfacing
//! as an app-core `PC=0` (`InstrProhibited`). This bin exercises the *exact*
//! trampoline/closure dispatch path with a tiny stack and no model, to confirm
//! the dispatch LOGIC itself is sound independent of the stack wall.
//!
//! Pass criterion: both halves of the output array are filled by the two cores
//! across many dispatches, with `f(r) == r * 7 + 1` everywhere (i.e. core 1's
//! band was computed correctly via the transmuted fn-pointer trampoline).

#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::{
    clock::CpuClock,
    delay::Delay,
    main,
    system::{CpuControl, Stack},
};
use esp_println::println;

use microflow::multicore::{parallel_for_rows, worker_loop, worker_ready, SharedRows};

esp_bootloader_esp_idf::esp_app_desc!();

static mut APP_CORE_STACK: Stack<8192> = Stack::new();

const N: usize = 240; // like an output-row count

#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);
    let delay = Delay::new();

    println!("System Init. mc_dispatch_test.");

    // Start core 1 into the real microflow worker loop.
    let mut cpu_control = CpuControl::new(peripherals.CPU_CTRL);
    let _guard = cpu_control
        .start_app_core(
            unsafe { &mut *core::ptr::addr_of_mut!(APP_CORE_STACK) },
            || worker_loop(),
        )
        .unwrap();
    while !worker_ready() {
        core::hint::spin_loop();
    }
    println!("WORKER_READY:1");

    // Run many dispatches to exercise the generation counter and the trampoline.
    let mut all_ok = true;
    let mut total_bad = 0u32;
    for round in 0..1000u32 {
        let mut out = [0i32; N];
        let base = SharedRows(out.as_mut_ptr());
        parallel_for_rows(N, |r0, r1| {
            let base = base;
            for i in r0..r1 {
                // SAFETY: disjoint bands across cores; barrier orders the writes.
                unsafe { base.0.add(i).write((i as i32) * 7 + 1 + round as i32) };
            }
        });
        // Verify the whole array (both cores' bands).
        for i in 0..N {
            if out[i] != (i as i32) * 7 + 1 + round as i32 {
                all_ok = false;
                total_bad += 1;
            }
        }
    }

    println!("DISPATCH_OK:{} BAD:{}", all_ok, total_bad);
    println!("MC_DISPATCH_DONE");

    loop {
        delay.delay_millis(1000);
    }
}
