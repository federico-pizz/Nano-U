#![no_std]
#![no_main]
#![deny(unsafe_code)]

use esp_backtrace as _;
use esp_hal::{clock::CpuClock, delay::Delay, main, rtc_cntl::Rtc, timer::timg::TimerGroup};
use esp_println::println;

esp_bootloader_esp_idf::esp_app_desc!();

use microflow::buffer::{Buffer2D, Buffer4D};
use microflow::model;

// Input: 60x80x3, Output: 60x80x1
#[model("models/nano_u.tflite")]
struct UNet;

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

    println!("System Init. Clock: Max. WDT Disabled. Starting Inference...");

    loop {
        // Allocate 36KB on the stack for the input batch (int8 quantization).
        let input_image = Buffer2D::<[i8; 3], 60, 80>::from_element([0, 0, 0]);
        let input_batch: Buffer4D<i8, 1, 60, 80, 3> = [input_image];

        println!("Running Inference Iteration...");

        let start = esp_hal::time::Instant::now();

        // Run Prediction with quantized inputs (i8) to save memory
        let _output_batch = UNet::predict_quantized(input_batch);

        let duration = start.elapsed();
        println!("Inference done in {} ms", duration.as_millis());

        delay.delay_millis(1000);
    }
}
