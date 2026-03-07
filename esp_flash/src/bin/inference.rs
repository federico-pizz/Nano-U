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

    // Load raw test image sequences
    const RAW_IMAGES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/input_images.bin"));
    const IMG_SIZE: usize = 60 * 80 * 3;
    let mut num_images = RAW_IMAGES.len() / IMG_SIZE;
    
    // Allocate 14.4KB on the stack for the input batch (int8 quantization).
    let mut input_image = Buffer2D::<[i8; 3], 60, 80>::from_element([0, 0, 0]);

    println!("BENCHMARK_START:{}", num_images);
    
    for i in 0..num_images {
        let start_idx = i * IMG_SIZE;
        let img_data = &RAW_IMAGES[start_idx..start_idx + IMG_SIZE];
        
        // Setup the quantization mapping LUT (0-255 -> int8)
        let mut quant_lut = [0i8; 256];
        for (x, q) in quant_lut.iter_mut().enumerate() {
            let x_norm = (x as f32 / 255.0 - 0.5) / 0.5;
            let q_float = (x_norm / 0.0071664745) - 13.0;
            // Round ties to nearest, then clamp to i8
            let mut q_int = libm::roundf(q_float) as i32;
            if q_int < -128 { q_int = -128; }
            if q_int > 127 { q_int = 127; }
            *q = q_int as i8;
        }

        // Map 0-255 inputs to int8 [-128, 127] perfectly synchronized with the host model
        for h in 0..60 {
            for w in 0..80 {
                let base_idx = (h * 80 + w) * 3;
                let r = quant_lut[img_data[base_idx] as usize];
                let g = quant_lut[img_data[base_idx + 1] as usize];
                let b = quant_lut[img_data[base_idx + 2] as usize];
                input_image[(h, w)] = [r, g, b];
            }
        }
        
        let input_batch: Buffer4D<i8, 1, 60, 80, 3> = [input_image];
        
        if i == 0 {
            println!("Rust input layout Top-Left First 3 Pixels:");
            println!("{:?}", input_batch[0][(0, 0)]);
            println!("{:?}", input_batch[0][(0, 1)]);
            println!("{:?}", input_batch[0][(0, 2)]);
        }
        
        let start = esp_hal::time::Instant::now();
        let output_batch = UNet::predict_quantized(input_batch);
        let duration = start.elapsed();
        
        println!("IMG_DURATION:{}:{}ms", i, duration.as_millis());
        println!("IMG_OUTPUT_START:{}", i);
        
        // Stream the hex-encoded i8 tensor safely
        let mut hex_buf = [0u8; 80 * 2];
        for h in 0..60 {
            for w in 0..80 {
                let f_val = output_batch[0][(h, w)][0];
                let mut int_val = f_val / 0.06899828463792801 + 94.0;
                int_val = if int_val > 127.0 { 127.0 } else if int_val < -128.0 { -128.0 } else { int_val };
                let val = libm::roundf(int_val) as i8 as u8;
                
                let hi = val >> 4;
                let lo = val & 0x0F;
                hex_buf[w * 2] = if hi < 10 { b'0' + hi } else { b'a' + (hi - 10) };
                hex_buf[w * 2 + 1] = if lo < 10 { b'0' + lo } else { b'a' + (lo - 10) };
            }
            let s = unsafe { core::str::from_utf8_unchecked(&hex_buf) };
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
