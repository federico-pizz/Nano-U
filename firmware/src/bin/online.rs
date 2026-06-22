#![no_std]
#![no_main]

//! Online (live-camera) inference — the real-time control loop.
//!
//! Unlike `run`/`inference`/`single_inference`, which replay PNGs baked into the
//! binary at build time, this binary captures **live frames from the OV2640** on
//! the Goouuu ESP32-S3-CAM, runs the segmentation model, and prints a navigation
//! decision every frame:
//!
//! ```text
//! FRAME:<n> infer=<ms>ms cover L=<f> C=<f> R=<f> -> GO|STOP steer=LEFT|CENTER|RIGHT
//! ```
//!
//! Pipeline per frame (all buffers allocated once, reused in place):
//!   capture (DMA, PSRAM) → 2×2 downscale + INT8 quantize (SRAM) →
//!   `predict_quantized` (SRAM activations) → region coverage → `decide`.
//!
//! The loop is inference-bound (~800 ms/frame); capture + downscale add ~a few ms
//! and run against PSRAM, leaving model latency/throughput unchanged. See
//! `src/camera.rs` for the hardware-only capture notes.

use esp_backtrace as _;
use esp_hal::{clock::CpuClock, delay::Delay, main, rtc_cntl::Rtc, timer::timg::TimerGroup};
use esp_println::println;

esp_bootloader_esp_idf::esp_app_desc!();

use microflow::buffer::{Buffer2D, Buffer4D};
use microflow::model;

use nano_u_esp::{
    build_quant_luts,
    camera::{OnboardCamera, FRAME_BYTES},
    control::{decide, Policy, Steer},
    IMG_H, IMG_W,
};

// Model copied from ../models/<MODEL_NAME>.tflite by build.rs.
// Input: 60x80x3, Output: 60x80x1.
#[model("models/nano_u.tflite")]
struct UNet;

/// Drivable probability threshold (matches the firmware/host convention).
const ROAD_THRESHOLD: f32 = 0.5;

#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);
    let delay = Delay::new();

    // Camera framebuffer in PSRAM (auto-mapped by `esp_hal::init`). Internal SRAM
    // has no room — the model activations consume nearly all of it, and a 38 KB
    // SRAM frame overflows the stack guard. No heap: we take the first FRAME_BYTES
    // of the mapped PSRAM region directly (PSRAM base is page-aligned, satisfying
    // the 64-byte DMA alignment). Octal mode via ESP_HAL_CONFIG_PSRAM_MODE (N16R8).
    let (psram_ptr, psram_len) = esp_hal::psram::psram_raw_parts(&peripherals.PSRAM);
    assert!(psram_len >= FRAME_BYTES, "PSRAM too small for framebuffer");
    let framebuffer: &'static mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(psram_ptr, FRAME_BYTES) };

    // Disable watchdogs so the long inference never trips a reset.
    let mut rtc = Rtc::new(peripherals.LPWR);
    rtc.swd.disable();
    rtc.rwdt.disable();
    let mut timg0 = TimerGroup::new(peripherals.TIMG0);
    timg0.wdt.disable();
    let mut timg1 = TimerGroup::new(peripherals.TIMG1);
    timg1.wdt.disable();

    println!("System Init. Clock: Max. WDT Disabled. Starting ONLINE inference...");

    let luts = build_quant_luts();

    // Bring up the OV2640 (Goouuu ESP32-S3-CAM = ESP32-S3-EYE pin map). The
    // driver steals the LCD_CAM / I2C0 / DMA / GPIO singletons it needs (camera
    // pins only), so we don't thread them through here.
    let mut camera = match OnboardCamera::new(framebuffer) {
        Ok(c) => c,
        Err(e) => {
            println!("CAMERA_INIT_FAILED: {:?}", e);
            loop {
                delay.delay_millis(1000);
            }
        }
    };

    println!("Camera up. Entering control loop.");

    // Input frame allocated once, overwritten in place each iteration.
    let mut input_image = Buffer2D::<[i8; 3], IMG_H, IMG_W>::from_element([0, 0, 0]);
    let policy = Policy::default();

    let mut frame: u32 = 0;
    loop {
        // 1. Live capture → downscale → quantize, in place.
        if let Err(e) = camera.capture_into(&luts, &mut input_image) {
            println!("FRAME:{} CAPTURE_ERROR: {:?}", frame, e);
            delay.delay_millis(100);
            continue;
        }

        // 2. Inference.
        let input_batch: Buffer4D<i8, 1, IMG_H, IMG_W, 3> = [input_image];
        let start = esp_hal::time::Instant::now();
        let output = UNet::predict_quantized(input_batch);
        let infer_ms = start.elapsed().as_millis();

        // 3. Region coverage over the lower band (near ground), split L/C/R.
        let (left, center, right) = region_coverage(&output);

        // 4. Decide.
        let d = decide(left, center, right, policy);
        let steer = match d.steer {
            Steer::Left => "LEFT",
            Steer::Center => "CENTER",
            Steer::Right => "RIGHT",
        };
        println!(
            "FRAME:{} infer={}ms cover L={} C={} R={} -> {} steer={}",
            frame,
            infer_ms,
            left,
            center,
            right,
            if d.go { "GO" } else { "STOP" },
            steer,
        );

        frame = frame.wrapping_add(1);
    }
}

/// Drivable-pixel fraction in the left / centre / right thirds of the lower
/// half of the mask (where the near ground lies). Returns `(left, center, right)`
/// each in `0.0..=1.0`.
fn region_coverage(output: &Buffer4D<f32, 1, IMG_H, IMG_W, 1>) -> (f32, f32, f32) {
    let row_start = IMG_H / 2; // lower band only
    let third = IMG_W / 3;

    let mut counts = [0u32; 3];
    let mut totals = [0u32; 3];

    for h in row_start..IMG_H {
        for w in 0..IMG_W {
            let region = if w < third {
                0
            } else if w < 2 * third {
                1
            } else {
                2
            };
            totals[region] += 1;
            if output[0][(h, w)][0] > ROAD_THRESHOLD {
                counts[region] += 1;
            }
        }
    }

    let frac = |i: usize| {
        if totals[i] == 0 {
            0.0
        } else {
            counts[i] as f32 / totals[i] as f32
        }
    };
    (frac(0), frac(1), frac(2))
}
