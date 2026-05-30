#![no_std]
#![no_main]
#![deny(unsafe_code)]

//! Continuous ("regular") inference binary — the default `cargo run` target.
//!
//! Its only job is to *run*: it executes the segmentation model in an endless
//! loop, one inference every [`INFERENCE_INTERVAL_MS`] milliseconds, and prints
//! a short per-frame summary. It contains **no `unsafe` code** (enforced by
//! `#![deny(unsafe_code)]`) and no profiling — stack/power measurement lives in
//! the `analysis` binaries.
//!
//! The single input buffer is allocated once before the loop and overwritten in
//! place each iteration, so the loop does no per-iteration allocation.

use esp_backtrace as _;
use esp_hal::{clock::CpuClock, delay::Delay, main, rtc_cntl::Rtc, timer::timg::TimerGroup};
use esp_println::println;

esp_bootloader_esp_idf::esp_app_desc!();

use microflow::buffer::{Buffer2D, Buffer4D};
use microflow::model;

use nano_u_esp::{build_quant_luts, preprocess_rgb, IMG_H, IMG_SIZE, IMG_W, NUM_IMAGES};

// Model is copied from ../models/<MODEL_NAME>.tflite by build.rs (Python pipeline output).
// Input: 60x80x3, Output: 60x80x1
#[model("models/nano_u.tflite")]
struct UNet;

/// Delay between successive inferences. Change this to set the run cadence.
const INFERENCE_INTERVAL_MS: u32 = 500;

/// Input frames, baked in at build time from the test image set.
///
/// To change the input, swap the PNGs the build script reads (see `build.rs`,
/// env var `TEST_IMG_DIR`) and rebuild — no code change is needed. Frames are
/// stored back-to-back as raw RGB8, `IMG_SIZE` bytes each.
const RAW_IMAGES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/input_images.bin"));

#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);
    let delay = Delay::new();

    // Disable the watchdogs so a long-running inference loop is never reset.
    let mut rtc = Rtc::new(peripherals.LPWR);
    rtc.swd.disable();
    rtc.rwdt.disable();
    let mut timg0 = TimerGroup::new(peripherals.TIMG0);
    timg0.wdt.disable();
    let mut timg1 = TimerGroup::new(peripherals.TIMG1);
    timg1.wdt.disable();

    println!("System Init. Clock: Max. WDT Disabled. Starting continuous inference...");

    let luts = build_quant_luts();

    // Allocate the input frame once and reuse it every iteration.
    let mut input_image = Buffer2D::<[i8; 3], IMG_H, IMG_W>::from_element([0, 0, 0]);

    println!("RUN_START num_images={}", NUM_IMAGES);

    let mut frame: u32 = 0;
    loop {
        let img_idx = if NUM_IMAGES > 0 {
            (frame as usize) % NUM_IMAGES
        } else {
            0
        };

        // Load and quantize the next frame in place.
        let start_idx = img_idx * IMG_SIZE;
        if start_idx + IMG_SIZE <= RAW_IMAGES.len() {
            preprocess_rgb(&RAW_IMAGES[start_idx..start_idx + IMG_SIZE], &luts, &mut input_image);
        }

        // Run one inference.
        let input_batch: Buffer4D<i8, 1, IMG_H, IMG_W, 3> = [input_image];
        let start = esp_hal::time::Instant::now();
        let output_batch = UNet::predict_quantized(input_batch);
        let infer_ms = start.elapsed().as_millis();

        // Cheap summary of the segmentation mask.
        let mut traversable: u32 = 0;
        for h in 0..IMG_H {
            for w in 0..IMG_W {
                if output_batch[0][(h, w)][0] > 0.5 {
                    traversable += 1;
                }
            }
        }

        println!(
            "FRAME:{} img={} infer={}ms traversable={}/{}",
            frame,
            img_idx,
            infer_ms,
            traversable,
            IMG_H * IMG_W
        );

        frame = frame.wrapping_add(1);
        delay.delay_millis(INFERENCE_INTERVAL_MS);
    }
}
