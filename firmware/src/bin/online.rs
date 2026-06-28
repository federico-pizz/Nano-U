#![no_std]
#![no_main]

//! Online (live-camera) inference — the real-time control loop on the Goouuu
//! ESP32-S3-CAM. Captures live OV2640 frames, runs the segmentation model, and
//! prints one navigation decision per frame (see `firmware/README.md`).
//!
//! Per frame, every buffer allocated once and reused in place:
//!   capture (DMA, PSRAM) → 2×2 downscale + INT8 quantize → predict → region
//!   coverage → decide. The loop is inference-bound (~800 ms/frame); capture and
//!   downscale add a few ms against PSRAM, leaving model latency unchanged.

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

    // Camera framebuffer in PSRAM (auto-mapped by `esp_hal::init`): the model
    // activations fill nearly all internal SRAM, and a 38 KB SRAM frame would
    // overflow the stack guard. No heap — take the first FRAME_BYTES of the
    // page-aligned PSRAM region directly (satisfies the 64-byte DMA alignment).
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

    // Start core 1 before the camera so the very first frame's inference is split
    // across both cores (capture/DMA stay on core 0).
    nano_u_esp::start_dual_core!(peripherals.CPU_CTRL);

    println!("System Init. Clock: Max. WDT Disabled. Dual-core worker up. Starting ONLINE inference...");

    let luts = build_quant_luts();

    // The driver steals the LCD_CAM / I2C0 / DMA / GPIO singletons it needs, so we
    // don't thread them through here.
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
        // Live capture → downscale → quantize, in place.
        if let Err(e) = camera.capture_into(&luts, &mut input_image) {
            println!("FRAME:{} CAPTURE_ERROR: {:?}", frame, e);
            delay.delay_millis(100);
            continue;
        }

        let input_batch: Buffer4D<i8, 1, IMG_H, IMG_W, 3> = [input_image];
        let start = esp_hal::time::Instant::now();
        let output = UNet::predict_quantized(input_batch);
        let infer_ms = start.elapsed().as_millis();

        // Drivable coverage over the lower band (near ground), split L/C/R.
        let (left, center, right) = region_coverage(&output);

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
