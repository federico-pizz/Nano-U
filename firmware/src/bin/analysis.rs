#![no_std]
#![no_main]
#![feature(asm_experimental_arch)]

use esp_backtrace as _;
use esp_hal::{clock::CpuClock, delay::Delay, main, rtc_cntl::Rtc, timer::timg::TimerGroup};
use esp_println::println;

esp_bootloader_esp_idf::esp_app_desc!();

use microflow::buffer::Buffer4D;
use microflow::model;

// Model is copied from ../models/nano_u.tflite by build.rs (Python pipeline output)
// Input: 60x80x3, Output: 60x80x1
#[model("models/nano_u.tflite")]
struct UNet;

const fn parse_u32_const(s: &str) -> u32 {
    let bytes = s.as_bytes();
    let mut val: u32 = 0;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] >= b'0' && bytes[i] <= b'9' {
            val = val * 10 + (bytes[i] - b'0') as u32;
        }
        i += 1;
    }
    val
}

const fn str_to_i32_const(s: &str) -> i32 {
    let bytes = s.as_bytes();
    let mut val: i64 = 0;
    let mut neg = false;
    let mut i = 0;

    while i < bytes.len() && bytes[i] == b' ' { i += 1; }

    if i < bytes.len() && bytes[i] == b'-' {
        neg = true;
        i += 1;
    } else if i < bytes.len() && bytes[i] == b'+' {
        i += 1;
    }

    while i < bytes.len() {
        let b = bytes[i];
        if b >= b'0' && b <= b'9' {
            val = val * 10 + (b - b'0') as i64;
        } else if b == b'.' {
            break;
        }
        i += 1;
    }
    let res = if neg { -val } else { val };
    res as i32
}

const IMG_H: usize = parse_u32_const(env!("NANO_U_IMG_H")) as usize;
const IMG_W: usize = parse_u32_const(env!("NANO_U_IMG_W")) as usize;
const IMG_SIZE: usize = IMG_H * IMG_W * 3;

// ── Quantization parameters ───────────────────────────────────────────────────
const INPUT_SCALE: f32       = f32::from_bits(parse_u32_const(env!("NANO_U_INPUT_SCALE_BITS")));
const INPUT_ZERO_POINT: i32  = str_to_i32_const(env!("NANO_U_INPUT_ZERO_POINT"));
const NUM_IMAGES: usize      = parse_u32_const(env!("NANO_U_NUM_IMAGES")) as usize;
const MEAN_R: f32 = f32::from_bits(parse_u32_const(env!("NANO_U_MEAN_R_BITS")));
const MEAN_G: f32 = f32::from_bits(parse_u32_const(env!("NANO_U_MEAN_G_BITS")));
const MEAN_B: f32 = f32::from_bits(parse_u32_const(env!("NANO_U_MEAN_B_BITS")));
const STD_R: f32  = f32::from_bits(parse_u32_const(env!("NANO_U_STD_R_BITS")));
const STD_G: f32  = f32::from_bits(parse_u32_const(env!("NANO_U_STD_G_BITS")));
const STD_B: f32  = f32::from_bits(parse_u32_const(env!("NANO_U_STD_B_BITS")));

static mut INPUT_BUFFER: Option<Buffer4D<i8, 1, IMG_H, IMG_W, 3>> = None;

unsafe extern "C" {
    static mut _stack_start: u32;
    static mut _stack_end: u32;
}

const STACK_PATTERN: u8 = 0xAA;
const STACK_BOTTOM_MARGIN: usize = 512; // Bottom-of-stack safety margin to avoid touching the guard region

/// Fills the unused stack memory with a pattern to detect usage.
unsafe fn paint_stack() {
    let sp: usize;
    unsafe {
        core::arch::asm!("mov {0}, a1", out(reg) sp);
    }

    let stack_end = { core::ptr::addr_of!(_stack_end) as usize };

    // Leave a safety margin of 256 bytes below the current stack pointer as an extra guard
    let paint_end = sp - 256;
    let paint_start = stack_end + STACK_BOTTOM_MARGIN;

    if paint_end > paint_start {
        let len = paint_end - paint_start;
        let ptr = paint_start as *mut u8;
        // Fill with pattern
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

/// Scans the stack from the bottom up to find the high water mark.
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

    println!("System Init. Clock: Max. WDT Disabled. Starting Inference...");
    println!("Allocating Input in STATIC memory (.bss)...");

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

    // Build quantization LUTs once (same formula as inference.rs)
    let mut quant_lut_r = [0i8; 256];
    let mut quant_lut_g = [0i8; 256];
    let mut quant_lut_b = [0i8; 256];
    for j in 0..256 {
        let x = j as f32 / 255.0;
        let q_r = libm::roundf((x - MEAN_R) / STD_R / INPUT_SCALE + INPUT_ZERO_POINT as f32) as i32;
        quant_lut_r[j] = q_r.clamp(-128, 127) as i8;
        let q_g = libm::roundf((x - MEAN_G) / STD_G / INPUT_SCALE + INPUT_ZERO_POINT as f32) as i32;
        quant_lut_g[j] = q_g.clamp(-128, 127) as i8;
        let q_b = libm::roundf((x - MEAN_B) / STD_B / INPUT_SCALE + INPUT_ZERO_POINT as f32) as i32;
        quant_lut_b[j] = q_b.clamp(-128, 127) as i8;
    }

    // Run a finite number of iterations for stack analysis
    for i in 0..50 {
        let img_idx = if NUM_IMAGES > 0 { i % NUM_IMAGES } else { 0 };
        println!("Running Inference Iteration {}...", i + 1);

        let start_idx = img_idx * IMG_SIZE;
        unsafe {
            if let Some(ref mut batch) = INPUT_BUFFER {
                if start_idx + IMG_SIZE <= RAW_IMAGES.len() {
                    let img_data = &RAW_IMAGES[start_idx..start_idx + IMG_SIZE];

                    for h in 0..IMG_H {
                        for w in 0..IMG_W {
                            let base_idx = (h * IMG_W + w) * 3;
                            let r = quant_lut_r[img_data[base_idx]     as usize];
                            let g = quant_lut_g[img_data[base_idx + 1] as usize];
                            let b = quant_lut_b[img_data[base_idx + 2] as usize];
                            batch[0][(h, w)] = [r, g, b];
                        }
                    }
                }
            }
        }

        let start = esp_hal::time::Instant::now();
        let _output_batch =
            unsafe { UNet::predict_quantized(INPUT_BUFFER.expect("Buffer uninitialized")) };
        let duration = start.elapsed();

        println!("Inference done in {} ms", duration.as_millis());

        unsafe {
            let peak_stack = measure_stack();
            println!("STACK_PEAK:{}", peak_stack);

            let stack_start = core::ptr::addr_of!(_stack_start) as usize;
            let stack_end = core::ptr::addr_of!(_stack_end) as usize;
            println!("STACK_TOTAL:{}", stack_start - stack_end);
        }

        // Re-initialize for next loop safety
        unsafe {
            let dummy_image =
                microflow::buffer::Buffer2D::<[i8; 3], IMG_H, IMG_W>::from_element([0, 0, 0]);
            INPUT_BUFFER = Some([dummy_image]);
        }

        delay.delay_millis(500);
    }

    println!("ANALYSIS_DONE");
    println!("POWER_MEASUREMENT_START");

    // --- CONTINUOUS REAL-IMAGE LOAD FOR MULTIMETER ---
    // Pre-process a single image once into a buffer to avoid timing/energy jitter
    println!("Preparing static image for continuous inference...");
    let mut static_input_image =
        microflow::buffer::Buffer2D::<[i8; 3], IMG_H, IMG_W>::from_element([0, 0, 0]);
    if IMG_SIZE <= RAW_IMAGES.len() {
        let img_data = &RAW_IMAGES[0..IMG_SIZE];
        for h in 0..IMG_H {
            for w in 0..IMG_W {
                let base_idx = (h * IMG_W + w) * 3;
                let r = quant_lut_r[img_data[base_idx]     as usize];
                let g = quant_lut_g[img_data[base_idx + 1] as usize];
                let b = quant_lut_b[img_data[base_idx + 2] as usize];
                static_input_image[(h, w)] = [r, g, b];
            }
        }
    }

    let static_batch: Buffer4D<i8, 1, IMG_H, IMG_W, 3> = [static_input_image];
    println!("Starting continuous inference loop for power measurement.");

    loop {
        let _output_batch = UNet::predict_quantized(static_batch);
    }
}
