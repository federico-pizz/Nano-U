#![no_std]
#![feature(asm_experimental_arch)]

use esp_println::println;
use microflow::buffer::Buffer2D;

// Online-inference support modules.
//
// `control` is pure (core-only) navigation logic and is host-testable in
// isolation. `camera` is the OV2640 live-capture driver and is hardware-only
// (it pulls in `esp-hal`'s `lcd_cam`, `i2c`, `dma` and PSRAM); it is only used
// by the `online` binary and does not affect the baked-image inference path.
pub mod camera;
pub mod control;

pub const fn parse_u32_const(s: &str) -> u32 {
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

pub const fn str_to_i32_const(s: &str) -> i32 {
    let bytes = s.as_bytes();
    let mut val: i64 = 0;
    let mut neg = false;
    let mut i = 0;

    while i < bytes.len() && bytes[i] == b' ' {
        i += 1;
    }

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

// Quantization parameters (emitted by build.rs from _quant_params.json)
pub const INPUT_SCALE: f32      = f32::from_bits(parse_u32_const(env!("NANO_U_INPUT_SCALE_BITS")));
pub const INPUT_ZERO_POINT: i32 = str_to_i32_const(env!("NANO_U_INPUT_ZERO_POINT"));
pub const IMG_H: usize          = parse_u32_const(env!("NANO_U_IMG_H")) as usize;
pub const IMG_W: usize          = parse_u32_const(env!("NANO_U_IMG_W")) as usize;
pub const IMG_SIZE: usize       = IMG_H * IMG_W * 3;
pub const NUM_IMAGES: usize     = parse_u32_const(env!("NANO_U_NUM_IMAGES")) as usize;
pub const MEAN_R: f32 = f32::from_bits(parse_u32_const(env!("NANO_U_MEAN_R_BITS")));
pub const MEAN_G: f32 = f32::from_bits(parse_u32_const(env!("NANO_U_MEAN_G_BITS")));
pub const MEAN_B: f32 = f32::from_bits(parse_u32_const(env!("NANO_U_MEAN_B_BITS")));
pub const STD_R: f32  = f32::from_bits(parse_u32_const(env!("NANO_U_STD_R_BITS")));
pub const STD_G: f32  = f32::from_bits(parse_u32_const(env!("NANO_U_STD_G_BITS")));
pub const STD_B: f32  = f32::from_bits(parse_u32_const(env!("NANO_U_STD_B_BITS")));

/// Builds per-channel [0..255] → i8 lookup tables for INT8 image preprocessing.
pub fn build_quant_luts() -> ([i8; 256], [i8; 256], [i8; 256]) {
    let mut lut_r = [0i8; 256];
    let mut lut_g = [0i8; 256];
    let mut lut_b = [0i8; 256];
    for j in 0..256 {
        let x = j as f32 / 255.0;
        lut_r[j] = (libm::roundf((x - MEAN_R) / STD_R / INPUT_SCALE + INPUT_ZERO_POINT as f32) as i32).clamp(-128, 127) as i8;
        lut_g[j] = (libm::roundf((x - MEAN_G) / STD_G / INPUT_SCALE + INPUT_ZERO_POINT as f32) as i32).clamp(-128, 127) as i8;
        lut_b[j] = (libm::roundf((x - MEAN_B) / STD_B / INPUT_SCALE + INPUT_ZERO_POINT as f32) as i32).clamp(-128, 127) as i8;
    }
    (lut_r, lut_g, lut_b)
}

/// Quantizes one raw RGB8 frame (`IMG_SIZE` bytes, row-major) into `out` using
/// the per-channel LUTs from [`build_quant_luts`]. `out` is reused in place, so
/// the caller does no per-frame allocation.
pub fn preprocess_rgb(
    img: &[u8],
    luts: &([i8; 256], [i8; 256], [i8; 256]),
    out: &mut Buffer2D<[i8; 3], IMG_H, IMG_W>,
) {
    let (lut_r, lut_g, lut_b) = luts;
    for h in 0..IMG_H {
        for w in 0..IMG_W {
            let base = (h * IMG_W + w) * 3;
            out[(h, w)] = [
                lut_r[img[base] as usize],
                lut_g[img[base + 1] as usize],
                lut_b[img[base + 2] as usize],
            ];
        }
    }
}

// Stack painting (shared by analysis binaries)

unsafe extern "C" {
    static mut _stack_start: u32;
    static mut _stack_end: u32;
}

const STACK_PATTERN: u8 = 0xAA;
const STACK_BOTTOM_MARGIN: usize = 512;

/// Fills unused stack memory with a sentinel pattern for high-water-mark measurement.
///
/// # Safety
/// Must be called before any significant stack usage. Writes to the region between
/// `_stack_end + STACK_BOTTOM_MARGIN` and `SP - 256`; caller must ensure no live
/// data occupies that range.
pub unsafe fn paint_stack() {
    let sp: usize;
    unsafe { core::arch::asm!("mov {0}, a1", out(reg) sp); }
    let stack_end = core::ptr::addr_of!(_stack_end) as usize;
    let paint_end = sp - 256;
    let paint_start = stack_end + STACK_BOTTOM_MARGIN;
    if paint_end > paint_start {
        let len = paint_end - paint_start;
        unsafe { core::ptr::write_bytes(paint_start as *mut u8, STACK_PATTERN, len); }
        println!("Stack painted from 0x{:x} to 0x{:x} ({} bytes)", paint_start, paint_end, len);
    } else {
        println!("Error: Stack overflow imminent or invalid SP!");
    }
}

/// Scans from the bottom of the stack upward to find the high-water mark.
///
/// # Safety
/// Reads raw memory between `_stack_end` and `_stack_start`. Must be called after
/// `paint_stack()` and only from a context where the linker symbols are valid.
pub unsafe fn measure_stack() -> usize {
    let stack_end = core::ptr::addr_of!(_stack_end) as usize;
    let stack_start = core::ptr::addr_of!(_stack_start) as usize;
    let scan_start = stack_end + STACK_BOTTOM_MARGIN;
    let ptr = scan_start as *const u8;
    let scan_len = stack_start - scan_start;
    let mut used_bytes = 0;
    for i in 0..scan_len {
        if unsafe { *ptr.add(i) } != STACK_PATTERN {
            used_bytes = stack_start - (scan_start + i);
            break;
        }
    }
    used_bytes
}

/// Returns total configured stack size in bytes.
///
/// # Safety
/// Reads the `_stack_start` and `_stack_end` linker symbols; valid only after
/// the firmware's memory map has been initialised by the runtime.
pub unsafe fn stack_total() -> usize {
    let stack_end = core::ptr::addr_of!(_stack_end) as usize;
    let stack_start = core::ptr::addr_of!(_stack_start) as usize;
    stack_start - stack_end
}

/// Starts the ESP32-S3 APP core (core 1) into microflow's dual-core
/// `worker_loop` and blocks until it is polling, so the **very first** inference
/// layer already runs split across both cores. Emits the `WORKER_READY:1` serial
/// marker once core 1 is up.
///
/// This is the one piece of multicore plumbing every dual-core binary needs;
/// it lives here so the bins don't each copy the `CpuControl` + static-stack
/// boilerplate. The microflow side (`worker_loop` / `parallel_for_rows`) only
/// exists when microflow's `multicore` feature is on — which the firmware's
/// `Cargo.toml` enables on the multicore branch.
///
/// Expands to statements in the **caller's** scope: the `CpuControl` and the
/// returned `AppCoreGuard` are bound with hygienic names and kept alive until
/// the end of the enclosing block. Dropping the guard parks core 1, so it must
/// outlive every `predict_quantized` call — invoke this once near the top of a
/// `-> !` `main`, after `esp_hal::init`.
///
/// # Example
/// ```ignore
/// let peripherals = esp_hal::init(config);
/// // ... watchdogs disabled ...
/// nano_u_esp::start_dual_core!(peripherals.CPU_CTRL);
/// // core 1 now serves microflow row-split jobs for the rest of main.
/// ```
#[macro_export]
macro_rules! start_dual_core {
    ($cpu_ctrl:expr) => {
        // APP-core stack (core 1). Sized for one op's row-fill call depth, not the
        // whole layer chain, so it stays far smaller than core 0's predict() stack.
        // See microflow docs/ESP32_S3_MULTICORE.md §6.1.
        static mut APP_CORE_STACK: ::esp_hal::system::Stack<32768> =
            ::esp_hal::system::Stack::new();
        let mut __mc_cpu_control = ::esp_hal::system::CpuControl::new($cpu_ctrl);
        // Guard must stay alive for the whole run; dropping it parks core 1.
        let __mc_app_guard = __mc_cpu_control
            .start_app_core(
                unsafe { &mut *::core::ptr::addr_of_mut!(APP_CORE_STACK) },
                || ::microflow::multicore::worker_loop(),
            )
            .unwrap();
        while !::microflow::multicore::worker_ready() {
            ::core::hint::spin_loop();
        }
        ::esp_println::println!("WORKER_READY:1");
    };
}
