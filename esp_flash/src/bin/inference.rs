#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::{clock::CpuClock, delay::Delay, main, rtc_cntl::Rtc, timer::timg::TimerGroup};
use esp_println::println;

esp_bootloader_esp_idf::esp_app_desc!();

use microflow::buffer::{Buffer2D, Buffer4D};
use microflow::model;

// Input: 60x80x3, Output: 60x80x1
#[model("models/nano_u.tflite")]
struct UNet;

// ── Quantization parameters ───────────────────────────────────────────────────
const INPUT_SCALE: f32      = f32::from_bits(parse_u32_const(env!("NANO_U_INPUT_SCALE_BITS")));
const INPUT_ZERO_POINT: i32 = str_to_i32_const(env!("NANO_U_INPUT_ZERO_POINT"));
const OUTPUT_SCALE: f32     = f32::from_bits(parse_u32_const(env!("NANO_U_OUTPUT_SCALE_BITS")));
const OUTPUT_ZERO_POINT: i32= str_to_i32_const(env!("NANO_U_OUTPUT_ZERO_POINT"));

const IMG_H: usize = parse_u32_const(env!("NANO_U_IMG_H")) as usize;
const IMG_W: usize = parse_u32_const(env!("NANO_U_IMG_W")) as usize;

const MEAN_R: f32 = f32::from_bits(parse_u32_const(env!("NANO_U_MEAN_R_BITS")));
const MEAN_G: f32 = f32::from_bits(parse_u32_const(env!("NANO_U_MEAN_G_BITS")));
const MEAN_B: f32 = f32::from_bits(parse_u32_const(env!("NANO_U_MEAN_B_BITS")));
const STD_R: f32  = f32::from_bits(parse_u32_const(env!("NANO_U_STD_R_BITS")));
const STD_G: f32  = f32::from_bits(parse_u32_const(env!("NANO_U_STD_G_BITS")));
const STD_B: f32  = f32::from_bits(parse_u32_const(env!("NANO_U_STD_B_BITS")));

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
    println!(
        "Quant params: input(scale={}, zp={}, {}x{}) output(scale={}, zp={})",
        INPUT_SCALE, INPUT_ZERO_POINT, IMG_W, IMG_H, OUTPUT_SCALE, OUTPUT_ZERO_POINT
    );

    const RAW_IMAGES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/input_images.bin"));
    const IMG_SIZE: usize = IMG_H * IMG_W * 3;
    let num_images = RAW_IMAGES.len() / IMG_SIZE;

    let mut input_image = Buffer2D::<[i8; 3], IMG_H, IMG_W>::from_element([0, 0, 0]);

    println!("BENCHMARK_START:{}", num_images);

    for i in 0..num_images {
        let start_idx = i * IMG_SIZE;
        let img_data = &RAW_IMAGES[start_idx..start_idx + IMG_SIZE];

        let mut quant_lut_r = [0i8; 256];
        let mut quant_lut_g = [0i8; 256];
        let mut quant_lut_b = [0i8; 256];

        for j in 0..256 {
            let x = j as f32 / 255.0;
            
            let q_float_r = (x - MEAN_R) / STD_R / INPUT_SCALE + INPUT_ZERO_POINT as f32;
            let q_int_r = libm::roundf(q_float_r) as i32;
            quant_lut_r[j] = q_int_r.clamp(-128, 127) as i8;

            let q_float_g = (x - MEAN_G) / STD_G / INPUT_SCALE + INPUT_ZERO_POINT as f32;
            let q_int_g = libm::roundf(q_float_g) as i32;
            quant_lut_g[j] = q_int_g.clamp(-128, 127) as i8;

            let q_float_b = (x - MEAN_B) / STD_B / INPUT_SCALE + INPUT_ZERO_POINT as f32;
            let q_int_b = libm::roundf(q_float_b) as i32;
            quant_lut_b[j] = q_int_b.clamp(-128, 127) as i8;
        }

        for h in 0..IMG_H {
            for w in 0..IMG_W {
                let base_idx = (h * IMG_W + w) * 3;
                let r = quant_lut_r[img_data[base_idx]     as usize];
                let g = quant_lut_g[img_data[base_idx + 1] as usize];
                let b = quant_lut_b[img_data[base_idx + 2] as usize];
                input_image[(h, w)] = [r, g, b];
            }
        }

        let input_batch: Buffer4D<i8, 1, IMG_H, IMG_W, 3> = [input_image];

        let start = esp_hal::time::Instant::now();
        let output_batch = UNet::predict_quantized(input_batch);
        let duration = start.elapsed();

        println!("IMG_DURATION:{}:{}ms", i, duration.as_millis());

        // Diagnostic: Print output range
        let mut min_val: f32 = 1.0e10;
        let mut max_val: f32 = -1.0e10;
        for row in 0..IMG_H {
            for col in 0..IMG_W {
                let val: f32 = output_batch[0][(row as usize, col as usize)][0];
                if val < min_val { min_val = val; }
                if val > max_val { max_val = val; }
            }
        }
        println!("IMG_STATS: min={}, max={}", min_val, max_val);

        println!("IMG_OUTPUT_START:{}", i);

        // ── Output serialization ─────────────────────────────────────────────
        // We will send bytes of the dequantized values or map them back to i8
        // For now, let's map them to i8 for the Python script compatibility
        let mut hex_buf = [0u8; 320];
        let row_len = IMG_W * 2;

        for h in 0..IMG_H {
            for w in 0..IMG_W {
                let f_val = output_batch[0][(h, w)][0];  
                
                // Map back to i8 for Python benchmark_on_esp.py
                // Formula: q = round(f / scale) + zp
                let q_val = libm::roundf((f_val / OUTPUT_SCALE) + OUTPUT_ZERO_POINT as f32) as i32;
                let val = q_val.clamp(-128, 127) as i8 as u8;

                let hi = val >> 4;
                let lo = val & 0x0F;
                hex_buf[w * 2]     = if hi < 10 { b'0' + hi } else { b'a' + (hi - 10) };
                hex_buf[w * 2 + 1] = if lo < 10 { b'0' + lo } else { b'a' + (lo - 10) };
            }
            let s = unsafe { core::str::from_utf8_unchecked(&hex_buf[..row_len]) };
            println!("{}", s);
        }
        println!("IMG_OUTPUT_END:{}", i);
        delay.delay_millis(100);
    }

    println!("BENCHMARK_DONE");

    loop {
        delay.delay_millis(1000);
    }
}