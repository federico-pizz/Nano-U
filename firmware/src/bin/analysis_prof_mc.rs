//! Per-layer latency profile of nano_u (dual core).
//!
//! Like `analysis_prof`, but starts the APP core into `worker_loop` first, so the
//! per-layer durations are the dual-core wall times (compute + barrier + bus
//! contention). Build with `["buffer-reuse", "multicore", "profiling"]`. Compare
//! against `analysis_prof` (single core) to see, per layer, how much the row
//! split actually saved vs. what the barrier cost.
//!
//! Op tags: 1=conv 2=depthwise 3=maxpool 4=resize.

#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::{
    clock::CpuClock,
    delay::Delay,
    main,
    rtc_cntl::Rtc,
    system::{CpuControl, Stack},
    timer::timg::TimerGroup,
};
use esp_println::println;

esp_bootloader_esp_idf::esp_app_desc!();

use microflow::buffer::Buffer4D;
use microflow::model;
use microflow::multicore::{worker_loop, worker_ready};

use nano_u_esp::{IMG_H, IMG_W};

#[model("models/nano_u.tflite")]
struct UNet;

static mut APP_CORE_STACK: Stack<32768> = Stack::new();

fn now_us() -> u64 {
    esp_hal::time::Instant::now()
        .duration_since_epoch()
        .as_micros()
}

#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);
    let delay = Delay::new();

    let mut rtc = Rtc::new(peripherals.LPWR);
    rtc.swd.disable();
    rtc.rwdt.disable();
    let mut timg0 = TimerGroup::new(peripherals.TIMG0);
    timg0.wdt.disable();
    let mut timg1 = TimerGroup::new(peripherals.TIMG1);
    timg1.wdt.disable();

    println!("System Init. Per-layer profile (dual core).");

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

    microflow::profile::set_now(now_us);

    let input: Buffer4D<i8, 1, IMG_H, IMG_W, 3> =
        [microflow::buffer::Buffer2D::from_element([0, 0, 0])];

    let _ = UNet::predict_quantized(input);

    for _round in 0..3 {
        microflow::profile::reset();
        let start = esp_hal::time::Instant::now();
        let _ = UNet::predict_quantized(input);
        let total = start.elapsed().as_micros();

        let n = microflow::profile::count();
        println!("PROFILE_BEGIN n={} total_us={}", n, total);
        let mut sum = [0u64; 5];
        let mut cnt = [0u32; 5];
        for i in 0..n {
            let (op, us) = microflow::profile::entry(i);
            println!("L{:02} op={} us={}", i, op, us);
            let o = (op as usize).min(4);
            sum[o] += us as u64;
            cnt[o] += 1;
        }
        let names = ["?", "conv", "depthwise", "maxpool", "resize"];
        for o in 1..5 {
            println!("SUM op={} ({}) count={} us={}", o, names[o], cnt[o], sum[o]);
        }
        println!("PROFILE_END");
        delay.delay_millis(500);
    }

    loop {
        delay.delay_millis(1000);
    }
}
