#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::{clock::CpuClock, delay::Delay, main, rtc_cntl::Rtc, timer::timg::TimerGroup};
use esp_println::println;

esp_bootloader_esp_idf::esp_app_desc!();

use microflow::buffer::Buffer4D;
use microflow::model;

use nano_u_esp::{build_quant_luts, measure_stack, paint_stack, preprocess_rgb, stack_total, IMG_H, IMG_SIZE, IMG_W, NUM_IMAGES};

// Model is copied from ../models/nano_u.tflite by build.rs (Python pipeline output)
// Input: 60x80x3, Output: 60x80x1
#[model("models/nano_u.tflite")]
struct UNet;

static mut INPUT_BUFFER: Option<Buffer4D<i8, 1, IMG_H, IMG_W, 3>> = None;

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

    // Dual-core: start core 1 so the timed inferences and the power loop both run
    // split across both cores (compare against the single-core analysis on main).
    nano_u_esp::start_dual_core!(peripherals.CPU_CTRL);

    println!("System Init. Clock: Max. WDT Disabled. Dual-core inference.");
    println!("Allocating Input in STATIC memory (.bss)...");

    unsafe {
        if INPUT_BUFFER.is_none() {
            let input_image =
                microflow::buffer::Buffer2D::<[i8; 3], IMG_H, IMG_W>::from_element([0, 0, 0]);
            INPUT_BUFFER = Some([input_image]);
        }
    }

    unsafe { paint_stack(); }

    const RAW_IMAGES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/input_images.bin"));

    let luts = build_quant_luts();

    for i in 0..50 {
        let img_idx = if NUM_IMAGES > 0 { i % NUM_IMAGES } else { 0 };
        println!("Running Inference Iteration {}...", i + 1);

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
    println!("POWER_MEASUREMENT_START");

    println!("Preparing static image for continuous inference...");
    let mut static_input_image =
        microflow::buffer::Buffer2D::<[i8; 3], IMG_H, IMG_W>::from_element([0, 0, 0]);
    if IMG_SIZE <= RAW_IMAGES.len() {
        preprocess_rgb(&RAW_IMAGES[0..IMG_SIZE], &luts, &mut static_input_image);
    }

    let static_batch: Buffer4D<i8, 1, IMG_H, IMG_W, 3> = [static_input_image];
    println!("Starting continuous inference loop for power measurement.");

    let mut tick: u32 = 0;
    loop {
        let _output_batch = UNet::predict_quantized(static_batch);
        tick = tick.wrapping_add(1);
        if tick.is_multiple_of(100) {
            println!("POWER_TICK:{}", tick);
        }
    }
}
