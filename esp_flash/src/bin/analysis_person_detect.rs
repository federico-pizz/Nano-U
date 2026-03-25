#![no_std]
#![no_main]
#![feature(asm_experimental_arch)]

use esp_backtrace as _;
use esp_hal::{clock::CpuClock, delay::Delay, main, rtc_cntl::Rtc, timer::timg::TimerGroup};
use esp_println::println;

esp_bootloader_esp_idf::esp_app_desc!();

use microflow::buffer::Buffer4D;
use microflow::model;

// Model: person_detect.tflite
// Input: 96x96x1 (grayscale), Output: 1x1x2 (person / no person logits)
#[model("models/person_detect.tflite")]
struct PersonDetect;

static mut INPUT_BUFFER: Option<Buffer4D<i8, 1, 96, 96, 1>> = None;

unsafe extern "C" {
    static mut _stack_start: u32;
    static mut _stack_end: u32;
}

const STACK_PATTERN: u8 = 0xAA;
const STACK_BOTTOM_MARGIN: usize = 512;

unsafe fn paint_stack() {
    let sp: usize;
    unsafe {
        core::arch::asm!("mov {0}, a1", out(reg) sp);
    }

    let stack_end = { core::ptr::addr_of!(_stack_end) as usize };
    let paint_end = sp - 256;
    let paint_start = stack_end + STACK_BOTTOM_MARGIN;

    if paint_end > paint_start {
        let len = paint_end - paint_start;
        let ptr = paint_start as *mut u8;
        unsafe {
            core::ptr::write_bytes(ptr, STACK_PATTERN, len);
        }
        println!(
            "Stack painted from 0x{:x} to 0x{:x} ({} bytes)",
            paint_start, paint_end, len
        );
    } else {
        println!("Error: Stack overflow imminent or invalid SP!");
    }
}

unsafe fn measure_stack() -> usize {
    let stack_end = { core::ptr::addr_of!(_stack_end) as usize };
    let stack_start = { core::ptr::addr_of!(_stack_start) as usize };

    let scan_start = stack_end + STACK_BOTTOM_MARGIN;
    let ptr = scan_start as *const u8;
    let scan_len = stack_start - scan_start;

    let mut used_bytes = 0;

    for i in 0..scan_len {
        if unsafe { *ptr.add(i) } != STACK_PATTERN {
            let current_addr = scan_start + i;
            used_bytes = stack_start - current_addr;
            break;
        }
    }
    used_bytes
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
            let input_image = microflow::buffer::Buffer2D::<[i8; 1], 96, 96>::from_element([0]);
            INPUT_BUFFER = Some([input_image]);
        }
    }

    unsafe {
        paint_stack();
    }

    // Include the original MicroFlow sample image (as BMP)
    const RAW_IMAGE: &[u8] = include_bytes!("../../models/test_img.bmp");

    for i in 0..50 {
        println!("Running Inference Iteration {}...", i + 1);

        unsafe {
            if let Some(ref mut batch) = INPUT_BUFFER {
                // A standard 96x96 24-bit BMP has a 54 byte header + 96*96*3 bytes of data
                let header_offset = 54;
                if RAW_IMAGE.len() >= header_offset + (96 * 96 * 3) {
                    for h in 0..96 {
                        for w in 0..96 {
                            // BMP stores pixels bottom-up natively, but for performance/benchmark 
                            // we just need valid data going into the model.
                            let idx = header_offset + (h * 96 + w) * 3;
                            // Grab the first channel (B) from BGR and cast it.
                            // Assuming model expects standard TFLite micro int8 centered around 0.
                            let val = (RAW_IMAGE[idx] as i16 - 128) as i8;
                            batch[0][(h, w)] = [val];
                        }
                    }
                } else if RAW_IMAGE.len() >= header_offset + (96 * 96) {
                    // Fallback for 8-bit grayscale BMP
                    for h in 0..96 {
                        for w in 0..96 {
                            let idx = header_offset + (h * 96 + w);
                            let val = (RAW_IMAGE[idx] as i16 - 128) as i8;
                            batch[0][(h, w)] = [val];
                        }
                    }
                }
            }
        }

        let start = esp_hal::time::Instant::now();
        let _output_batch = unsafe { PersonDetect::predict_quantized(INPUT_BUFFER.expect("Buffer uninitialized")) };
        let duration = start.elapsed();

        println!("Inference done in {} ms", duration.as_millis());

        unsafe {
            let peak_stack = measure_stack();
            println!("STACK_PEAK:{}", peak_stack);

            let stack_start = core::ptr::addr_of!(_stack_start) as usize;
            let stack_end = core::ptr::addr_of!(_stack_end) as usize;
            println!("STACK_TOTAL:{}", stack_start - stack_end);
        }

        delay.delay_millis(500);
    }

    println!("ANALYSIS_DONE");
    println!("POWER_MEASUREMENT_START");

    let mut static_input_image = microflow::buffer::Buffer2D::<[i8; 1], 96, 96>::from_element([0]);
    let header_offset = 54;
    if RAW_IMAGE.len() >= header_offset + (96 * 96 * 3) {
        for h in 0..96 {
            for w in 0..96 {
                let idx = header_offset + (h * 96 + w) * 3;
                static_input_image[(h, w)] = [(RAW_IMAGE[idx] as i16 - 128) as i8];
            }
        }
    } else if RAW_IMAGE.len() >= header_offset + (96 * 96) {
        for h in 0..96 {
            for w in 0..96 {
                let idx = header_offset + (h * 96 + w);
                static_input_image[(h, w)] = [(RAW_IMAGE[idx] as i16 - 128) as i8];
            }
        }
    }

    let static_batch: Buffer4D<i8, 1, 96, 96, 1> = [static_input_image];
    println!("Starting continuous inference loop for power measurement.");

    loop {
        let _output_batch = PersonDetect::predict_quantized(static_batch);
    }
}
