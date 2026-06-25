//! Dual-core nano_u inference analysis.
//!
//! Identical workload to `analysis.rs` (timed nano_u inference + stack
//! high-water via paint/measure), but it first starts the APP core (core 1)
//! into `microflow::multicore::worker_loop`, so the heavy conv/depthwise layers
//! split their output rows across both cores.
//!
//! Build with `microflow` features `["buffer-reuse", "multicore"]`: buffer-reuse
//! frees the stack so the second core's output band no longer overflows core 0
//! (the regression that previously surfaced as an app-core `PC=0`), and
//! multicore turns on the row split. Compare `Inference done in N ms` against the
//! single-core `analysis` bin built with just `["buffer-reuse"]`.

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

use nano_u_esp::{
    build_quant_luts, measure_stack, paint_stack, preprocess_rgb, stack_total, IMG_H, IMG_SIZE,
    IMG_W, NUM_IMAGES,
};

#[model("models/nano_u.tflite")]
struct UNet;

static mut INPUT_BUFFER: Option<Buffer4D<i8, 1, IMG_H, IMG_W, 3>> = None;

// APP-core stack (core 1). It only ever runs one `pixel()` band-fill at a time,
// whose frame is small, but keep generous headroom.
static mut APP_CORE_STACK: Stack<32768> = Stack::new();

#[main]
#[allow(static_mut_refs)]
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

    println!("System Init. Clock: Max. WDT Disabled. Dual-core inference.");

    // Start core 1 into the microflow worker loop and wait until it is polling.
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

    unsafe {
        if INPUT_BUFFER.is_none() {
            let input_image =
                microflow::buffer::Buffer2D::<[i8; 3], IMG_H, IMG_W>::from_element([0, 0, 0]);
            INPUT_BUFFER = Some([input_image]);
        }
    }

    unsafe {
        paint_stack();
    }

    const RAW_IMAGES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/input_images.bin"));
    let luts = build_quant_luts();

    for i in 0..50 {
        let img_idx = if NUM_IMAGES > 0 { i % NUM_IMAGES } else { 0 };
        let start_idx = img_idx * IMG_SIZE;
        unsafe {
            if let Some(ref mut batch) = INPUT_BUFFER
                && start_idx + IMG_SIZE <= RAW_IMAGES.len()
            {
                preprocess_rgb(&RAW_IMAGES[start_idx..start_idx + IMG_SIZE], &luts, &mut batch[0]);
            }
        }

        let start = esp_hal::time::Instant::now();
        let _output_batch =
            unsafe { UNet::predict_quantized(INPUT_BUFFER.expect("Buffer uninitialized")) };
        let duration = start.elapsed();

        println!("Inference done in {} ms", duration.as_millis());
        unsafe {
            println!("STACK_PEAK:{}", measure_stack());
            println!("STACK_TOTAL:{}", stack_total());
        }

        delay.delay_millis(500);
    }

    println!("ANALYSIS_DONE");
    loop {
        delay.delay_millis(1000);
    }
}
