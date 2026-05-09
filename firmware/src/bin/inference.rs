#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::{clock::CpuClock, delay::Delay, main, rtc_cntl::Rtc, timer::timg::TimerGroup};
use esp_println::println;

esp_bootloader_esp_idf::esp_app_desc!();

use microflow::buffer::{Buffer2D, Buffer4D};
use microflow::model;

use nano_u_esp::{parse_u32_const, str_to_i32_const};

// Input: 60x80x3, Output: 60x80x1
#[model("models/nano_u.tflite")]
struct UNet;

// ── Quantization parameters ───────────────────────────────────────────────────
const INPUT_SCALE: f32      = f32::from_bits(parse_u32_const(env!("NANO_U_INPUT_SCALE_BITS")));
const INPUT_ZERO_POINT: i32 = str_to_i32_const(env!("NANO_U_INPUT_ZERO_POINT"));

const IMG_H: usize = parse_u32_const(env!("NANO_U_IMG_H")) as usize;
const IMG_W: usize = parse_u32_const(env!("NANO_U_IMG_W")) as usize;

const MEAN_R: f32 = f32::from_bits(parse_u32_const(env!("NANO_U_MEAN_R_BITS")));
const MEAN_G: f32 = f32::from_bits(parse_u32_const(env!("NANO_U_MEAN_G_BITS")));
const MEAN_B: f32 = f32::from_bits(parse_u32_const(env!("NANO_U_MEAN_B_BITS")));
const STD_R: f32  = f32::from_bits(parse_u32_const(env!("NANO_U_STD_R_BITS")));
const STD_G: f32  = f32::from_bits(parse_u32_const(env!("NANO_U_STD_G_BITS")));
const STD_B: f32  = f32::from_bits(parse_u32_const(env!("NANO_U_STD_B_BITS")));

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
        "Quant params: input(scale={}, zp={}) img={}x{}",
        INPUT_SCALE, INPUT_ZERO_POINT, IMG_W, IMG_H
    );

    const RAW_IMAGES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/input_images.bin"));
    const IMG_SIZE: usize = IMG_H * IMG_W * 3;
    let num_images = RAW_IMAGES.len() / IMG_SIZE;

    let mut input_image = Buffer2D::<[i8; 3], IMG_H, IMG_W>::from_element([0, 0, 0]);

    println!("BENCHMARK_START:{}", num_images);

    let mut quant_lut_r = [0i8; 256];
    let mut quant_lut_g = [0i8; 256];
    let mut quant_lut_b = [0i8; 256];
    for j in 0..256 {
        let x = j as f32 / 255.0;
        quant_lut_r[j] = (libm::roundf((x - MEAN_R) / STD_R / INPUT_SCALE + INPUT_ZERO_POINT as f32) as i32).clamp(-128, 127) as i8;
        quant_lut_g[j] = (libm::roundf((x - MEAN_G) / STD_G / INPUT_SCALE + INPUT_ZERO_POINT as f32) as i32).clamp(-128, 127) as i8;
        quant_lut_b[j] = (libm::roundf((x - MEAN_B) / STD_B / INPUT_SCALE + INPUT_ZERO_POINT as f32) as i32).clamp(-128, 127) as i8;
    }

    for i in 0..num_images {
        let start_idx = i * IMG_SIZE;
        let img_data = &RAW_IMAGES[start_idx..start_idx + IMG_SIZE];

        let preprocess_start = esp_hal::time::Instant::now();
        for h in 0..IMG_H {
            for w in 0..IMG_W {
                let base_idx = (h * IMG_W + w) * 3;
                let r = quant_lut_r[img_data[base_idx]     as usize];
                let g = quant_lut_g[img_data[base_idx + 1] as usize];
                let b = quant_lut_b[img_data[base_idx + 2] as usize];
                input_image[(h, w)] = [r, g, b];
            }
        }
        let preprocess_us = preprocess_start.elapsed().as_micros();

        let input_batch: Buffer4D<i8, 1, IMG_H, IMG_W, 3> = [input_image];

        let infer_start = esp_hal::time::Instant::now();
        let output_batch = UNet::predict_quantized(input_batch);
        let infer_us = infer_start.elapsed().as_micros();

        println!("IMG_PREPROCESS:{}:{}us", i, preprocess_us);
        println!("IMG_INFERENCE:{}:{}us", i, infer_us);

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
        println!("IMG_STATS:{}:{},{}", i, min_val, max_val);

        println!("IMG_OUTPUT_START:{}", i);

        let mut hex_buf = [0u8; IMG_W * 8];
        let row_len = IMG_W * 8;

        for h in 0..IMG_H {
            for w in 0..IMG_W {
                let f_val = output_batch[0][(h, w)][0];
                let bits = f_val.to_bits();
                for byte_idx in 0..4usize {
                    let byte = ((bits >> (byte_idx * 8)) & 0xFF) as u8;
                    let hi = byte >> 4;
                    let lo = byte & 0x0F;
                    hex_buf[w * 8 + byte_idx * 2]     = if hi < 10 { b'0' + hi } else { b'a' + (hi - 10) };
                    hex_buf[w * 8 + byte_idx * 2 + 1] = if lo < 10 { b'0' + lo } else { b'a' + (lo - 10) };
                }
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