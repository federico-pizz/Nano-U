//! Per-layer latency profile of nano_u (dual core).
//!
//! Builds with `microflow` features `["buffer-reuse", "multicore", "profiling"]`.
//! Starts core 1 into the worker via [`start_dual_core!`], installs a microsecond
//! timestamp hook into `microflow::profile`, runs one inference, and prints each
//! layer's op-type and duration. Timing is data-independent, so a zero input is
//! fine. Use the output to see which layers dominate the dual-core run, and to
//! compare against the single-core profile on the `main` branch.
//!
//! Op tags: 1=conv 2=depthwise 3=maxpool 4=resize.

#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::{clock::CpuClock, delay::Delay, main, rtc_cntl::Rtc, timer::timg::TimerGroup};
use esp_println::println;

esp_bootloader_esp_idf::esp_app_desc!();

use microflow::buffer::Buffer4D;
use microflow::model;

use nano_u_esp::{IMG_H, IMG_W};

#[model("models/nano_u.tflite")]
struct UNet;

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

    nano_u_esp::start_dual_core!(peripherals.CPU_CTRL);

    println!("System Init. Per-layer profile (dual core).");
    microflow::profile::set_now(now_us);

    let input: Buffer4D<i8, 1, IMG_H, IMG_W, 3> =
        [microflow::buffer::Buffer2D::from_element([0, 0, 0])];

    // Warm run (discard), then a measured run.
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
