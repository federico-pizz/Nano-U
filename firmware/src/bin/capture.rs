#![no_std]
#![no_main]

//! Frame-capture / pipeline-validation binary for the Goouuu ESP32-S3-CAM.
//!
//! Runs the same bring-up as `online` but stops before inference, streaming the
//! intermediate images over serial so each stage can be checked on the host (see
//! `firmware/README.md`): `RAW` (160×120 RGB565 straight from DMA) then `DOWN`
//! (60×80 RGB888 after the 2×2 box downscale — the pixels the quantizer feeds the
//! model). `single_inference` streams the output mask, so the three together
//! cover the whole pipeline.
//!
//! Each frame is emitted as base64 framed by markers consumed by
//! `../scripts/capture_view.py`:
//!
//! ```text
//! <TAG>_BEGIN idx=<n> w=<w> h=<h> bytes=<len> fmt=<rgb565|rgb888>
//! <base64 line>...
//! <TAG>_END idx=<n>
//! ```

use esp_backtrace as _;
use esp_hal::{clock::CpuClock, delay::Delay, main, rtc_cntl::Rtc, timer::timg::TimerGroup};
use esp_println::println;

esp_bootloader_esp_idf::esp_app_desc!();

use nano_u_esp::{
    camera::{downscale_rgb888, OnboardCamera, CAM_H, CAM_W, FRAME_BYTES},
    IMG_H, IMG_W,
};

/// Standard base64 alphabet.
const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Delay between frames, so the host has time to drain the serial buffer and the
/// stream stays human-scannable. Capture itself is ~tens of ms.
const FRAME_INTERVAL_MS: u32 = 1000;

/// Stream `data` as base64 between `<tag>_BEGIN`/`<tag>_END` markers, 48 input
/// bytes (→ 64 chars) per line using only a stack buffer — no heap.
fn dump_b64(tag: &str, fmt: &str, idx: u32, w: usize, h: usize, data: &[u8]) {
    println!(
        "{}_BEGIN idx={} w={} h={} bytes={} fmt={}",
        tag,
        idx,
        w,
        h,
        data.len(),
        fmt
    );
    let mut line = [0u8; 64];
    for chunk in data.chunks(48) {
        let mut o = 0;
        for tri in chunk.chunks(3) {
            let b0 = tri[0] as u32;
            let b1 = *tri.get(1).unwrap_or(&0) as u32;
            let b2 = *tri.get(2).unwrap_or(&0) as u32;
            let n = (b0 << 16) | (b1 << 8) | b2;
            line[o] = B64[((n >> 18) & 63) as usize];
            line[o + 1] = B64[((n >> 12) & 63) as usize];
            line[o + 2] = if tri.len() > 1 { B64[((n >> 6) & 63) as usize] } else { b'=' };
            line[o + 3] = if tri.len() > 2 { B64[(n & 63) as usize] } else { b'=' };
            o += 4;
        }
        // SAFETY: every byte written above is an ASCII base64 char or '='.
        println!("{}", unsafe { core::str::from_utf8_unchecked(&line[..o]) });
    }
    println!("{}_END idx={}", tag, idx);
}

#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);
    let delay = Delay::new();

    // Camera framebuffer in PSRAM — identical setup to `online` (see that binary
    // for why the frame can't live in internal SRAM).
    let (psram_ptr, psram_len) = esp_hal::psram::psram_raw_parts(&peripherals.PSRAM);
    assert!(psram_len >= FRAME_BYTES, "PSRAM too small for framebuffer");
    let framebuffer: &'static mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(psram_ptr, FRAME_BYTES) };

    // Disable watchdogs (no long inference here, but keep parity with `online`).
    let mut rtc = Rtc::new(peripherals.LPWR);
    rtc.swd.disable();
    rtc.rwdt.disable();
    let mut timg0 = TimerGroup::new(peripherals.TIMG0);
    timg0.wdt.disable();
    let mut timg1 = TimerGroup::new(peripherals.TIMG1);
    timg1.wdt.disable();

    println!("System Init. Clock: Max. WDT Disabled. Starting CAPTURE stream...");

    let mut camera = match OnboardCamera::new(framebuffer) {
        Ok(c) => c,
        Err(e) => {
            println!("CAMERA_INIT_FAILED: {:?}", e);
            loop {
                delay.delay_millis(1000);
            }
        }
    };

    println!("Camera up. Streaming frames (RAW + DOWN). Ctrl-C to stop.");

    // Post-downscale RGB888 preview buffer, reused in place. No inference here, so
    // SRAM has room for it.
    let mut down = [0u8; IMG_H * IMG_W * 3];

    let mut idx: u32 = 0;
    loop {
        match camera.capture_raw() {
            Ok(frame) => {
                dump_b64("RAW", "rgb565", idx, CAM_W, CAM_H, frame);
                downscale_rgb888(frame, &mut down);
                dump_b64("DOWN", "rgb888", idx, IMG_W, IMG_H, &down);
                println!("FRAME_DONE idx={}", idx);
            }
            Err(e) => println!("CAPTURE_ERROR idx={} {:?}", idx, e),
        }
        idx = idx.wrapping_add(1);
        delay.delay_millis(FRAME_INTERVAL_MS);
    }
}
