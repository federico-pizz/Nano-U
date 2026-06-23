#![no_std]
#![no_main]

use esp_backtrace as _;
use esp_hal::{
    clock::CpuClock,
    delay::Delay,
    main,
    rtc_cntl::Rtc,
    system::{software_reset, CpuControl, Stack},
    timer::timg::TimerGroup,
};
use esp_println::println;

esp_bootloader_esp_idf::esp_app_desc!();

/// Stack for the APP core (core 1) running microflow's dual-core worker. Sized
/// for one op's row-fill call depth (not the whole layer chain), so it stays far
/// smaller than core 0's predict() stack. Competes with the S3 stack budget —
/// see microflow docs/ESP32_S3_MULTICORE.md §6.1.
static mut APP_CORE_STACK: Stack<32768> = Stack::new();

use microflow::buffer::{Buffer2D, Buffer4D};
use microflow::model;

use nano_u_esp::{build_quant_luts, parse_u32_const, preprocess_rgb, IMG_H, IMG_SIZE, IMG_W};

// Input: 60x80x3, Output: 60x80x1
#[model("models/nano_u.tflite")]
struct UNet;

const TARGET_IDX: usize = parse_u32_const(env!("NANO_U_TARGET_IMG_IDX")) as usize;

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

    // Start the APP core (core 1) into microflow's dual-core worker, then wait
    // until it is polling so the very first layer runs in parallel. The guard
    // must stay alive for the whole run (dropping it parks core 1).
    let mut cpu_control = CpuControl::new(peripherals.CPU_CTRL);
    let _app_guard = cpu_control
        .start_app_core(
            unsafe { &mut *core::ptr::addr_of_mut!(APP_CORE_STACK) },
            || microflow::multicore::worker_loop(),
        )
        .unwrap();
    while !microflow::multicore::worker_ready() {
        core::hint::spin_loop();
    }

    println!("System Init. Clock: Max. WDT Disabled. Dual-core worker up. Starting Single Inference...");
    println!("Target Image Index: {}", TARGET_IDX);

    const RAW_IMAGES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/input_images.bin"));
    let num_images = RAW_IMAGES.len() / IMG_SIZE;

    if TARGET_IDX >= num_images {
        println!("Error: Target index {} is out of bounds (0..{})", TARGET_IDX, num_images - 1);
        println!("BENCHMARK_DONE");
        delay.delay_millis(200);
        software_reset();
    }

    let mut input_image = Buffer2D::<[i8; 3], IMG_H, IMG_W>::from_element([0, 0, 0]);

    println!("BENCHMARK_START:1");

    let start_idx = TARGET_IDX * IMG_SIZE;
    let img_data = &RAW_IMAGES[start_idx..start_idx + IMG_SIZE];

    let luts = build_quant_luts();

    preprocess_rgb(img_data, &luts, &mut input_image);

    let input_batch: Buffer4D<i8, 1, IMG_H, IMG_W, 3> = [input_image];

    let start = esp_hal::time::Instant::now();
    let output_batch = UNet::predict_quantized(input_batch);
    let duration = start.elapsed();

    println!("IMG_DURATION:{}:{}ms", TARGET_IDX, duration.as_millis());

    println!("IMG_OUTPUT_START:{}", TARGET_IDX);

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
        // hex_buf holds only ASCII hex digits, so this never fails.
        let s = core::str::from_utf8(&hex_buf[..row_len]).unwrap_or("");
        println!("{}", s);
    }
    println!("IMG_OUTPUT_END:{}", TARGET_IDX);

    println!("BENCHMARK_DONE");

    // Flush UART then reset so the host receives BENCHMARK_DONE cleanly.
    delay.delay_millis(200);
    software_reset();
}
