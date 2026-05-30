#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::{clock::CpuClock, delay::Delay, main, rtc_cntl::Rtc, timer::timg::TimerGroup};
use esp_println::println;

esp_bootloader_esp_idf::esp_app_desc!();

use microflow::buffer::{Buffer2D, Buffer4D};
use microflow::model;

use nano_u_esp::{measure_stack, paint_stack, stack_total};

// Model: person_detect.tflite
// Input: 96x96x1 (grayscale), Output: 1x1x2 (person / no person logits)
#[model("models/person_detect.tflite")]
struct PersonDetect;

static mut INPUT_BUFFER: Option<Buffer4D<i8, 1, 96, 96, 1>> = None;

/// Decodes a 96×96 BMP into an INT8 buffer.
// BMP stores pixels bottom-up natively, but for this benchmark we only need
// valid data flowing through the model, so row ordering is intentionally ignored.
fn decode_bmp_into(raw: &[u8], out: &mut Buffer2D<[i8; 1], 96, 96>) {
    const HEADER: usize = 54;
    if raw.len() >= HEADER + 96 * 96 * 3 {
        for h in 0..96 {
            for w in 0..96 {
                out[(h, w)] = [(raw[HEADER + (h * 96 + w) * 3] as i16 - 128) as i8];
            }
        }
    } else if raw.len() >= HEADER + 96 * 96 {
        for h in 0..96 {
            for w in 0..96 {
                out[(h, w)] = [(raw[HEADER + h * 96 + w] as i16 - 128) as i8];
            }
        }
    }
}

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

    println!("System Init. Clock: Max. WDT Disabled. Starting Person Detection Inference...");

    unsafe {
        if INPUT_BUFFER.is_none() {
            let input_image = Buffer2D::<[i8; 1], 96, 96>::from_element([0]);
            INPUT_BUFFER = Some([input_image]);
        }
    }

    unsafe { paint_stack(); }

    const RAW_IMAGE: &[u8] = include_bytes!("../../models/test_img.bmp");

    for i in 0..50 {
        println!("Running Inference Iteration {}...", i + 1);

        unsafe {
            if let Some(ref mut batch) = INPUT_BUFFER {
                decode_bmp_into(RAW_IMAGE, &mut batch[0]);
            }
        }

        let start = esp_hal::time::Instant::now();
        let _output_batch = unsafe { PersonDetect::predict_quantized(INPUT_BUFFER.expect("Buffer uninitialized")) };
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

    let mut static_input_image = Buffer2D::<[i8; 1], 96, 96>::from_element([0]);
    decode_bmp_into(RAW_IMAGE, &mut static_input_image);

    let static_batch: Buffer4D<i8, 1, 96, 96, 1> = [static_input_image];
    println!("Starting continuous inference loop for power measurement.");

    let mut tick: u32 = 0;
    loop {
        let _output_batch = PersonDetect::predict_quantized(static_batch);
        tick = tick.wrapping_add(1);
        if tick.is_multiple_of(100) {
            println!("POWER_TICK:{}", tick);
        }
    }
}
