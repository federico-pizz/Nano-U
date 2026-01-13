#![no_std]
#![no_main]
#![feature(asm_experimental_arch)]

// --- Imports ---
use esp_backtrace as _;
use esp_hal::{
    clock::CpuClock,
    main,
    delay::Delay,
};
use esp_println::println;

// --- Bootloader ---
esp_bootloader_esp_idf::esp_app_desc!();

// Microflow & Math
use microflow::buffer::{Buffer4D, Buffer2D}; // Use Buffer4D for images (Batch, H, W, Channel)
use microflow::model;

// --- Model Definition ---
// Ensure "models/dummy1.tflite" exists in your project root!
// The macro assumes Input: 48x48x3 and Output: 48x48x1
#[model("models/dummy5.tflite")]
struct UNet;

unsafe extern "C" {
    // These symbols are defined in the linker script (stack.x)
    static mut _stack_start: u32;
    static mut _stack_end: u32;
}

const STACK_PATTERN: u8 = 0xAA;
const STACK_BOTTOM_MARGIN: usize = 512; // Safety margin to avoid stack guard

/// Fills the unused stack memory with a pattern to detect usage.
/// WARNING: This is unsafe and must be used with caution.
unsafe fn paint_stack() {
    let sp: usize;
    unsafe {
        core::arch::asm!("mov {0}, a1", out(reg) sp);
    }
    
    let stack_end = { core::ptr::addr_of!(_stack_end) as usize };
    
    // Leave a safety margin of 256 bytes below current SP to avoid corrupting active frames
    let paint_end = sp - 256;
    let paint_start = stack_end + STACK_BOTTOM_MARGIN;
    
    if paint_end > paint_start {
        let len = paint_end - paint_start;
        let ptr = paint_start as *mut u8;
        // Fill with pattern
        unsafe {
            core::ptr::write_bytes(ptr, STACK_PATTERN, len);
        }
        println!("Stack painted from 0x{:x} to 0x{:x} ({} bytes)", paint_start, paint_end, len);
    } else {
        println!("Error: Stack overflow imminent or invalid SP!");
    }
}

/// Scans the stack from the bottom up to find the high water mark.
unsafe fn measure_stack() -> usize {
    let stack_end = { core::ptr::addr_of!(_stack_end) as usize };
    let stack_start = { core::ptr::addr_of!(_stack_start) as usize };
    
    // Start scanning from the painted region
    let scan_start = stack_end + STACK_BOTTOM_MARGIN;
    let ptr = scan_start as *const u8;
    
    // The size of the area we are scanning (from bottom margin up to top)
    // Note: We are scanning the whole stack space above the margin, 
    // not just what we painted. If we reach unpainted territory (used stack), we stop.
    let scan_len = stack_start - scan_start;

    let mut used_bytes = 0;
    
    // Scan from bottom (scan_start) upwards
    for i in 0..scan_len {
        if unsafe { *ptr.add(i) } != STACK_PATTERN {
            // Found the first byte that was touched (or wasn't painted)
            // The used stack size is roughly (Total - i_relative_to_margin) ??
            // No.
            // Address of this byte is: scan_start + i
            // Used stack is from Top (stack_start) down to this address.
            // used = stack_start - (scan_start + i)
            let current_addr = scan_start + i;
            used_bytes = stack_start - current_addr;
            break;
        }
    }
    
    used_bytes
}

#[main]
fn main() -> ! {
    // 1. Init System at Max Speed (240MHz) for Inference
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let _peripherals = esp_hal::init(config);
    let delay = Delay::new();

    println!("System Init. Clock: Max. Starting Inference...");

    // 2. Prepare Input Data (48x48x3)
    // We create it here. NOTE: This lives on the STACK.
    // Ensure your stack size is > 50KB in .cargo/config.toml
    println!("Allocating Input...");
    
    // Create a 48x48 matrix where each element is [f32; 3] (RGB)
    // We initialize it with dummy values (e.g., 0.5)
    let input_image = Buffer2D::<[f32; 3], 48, 48>::from_element([0.5, 0.5, 0.5]);
    
    // Microflow expects a 4D Batch: [Batch, Row, Col, Channel] -> [1, 48, 48, 3]
    let input_batch: Buffer4D<f32, 1, 48, 48, 3> = [input_image];

    unsafe { paint_stack(); }

    // Run a finite number of iterations
    for i in 0..3 {
        println!("Running Inference Iteration {}...", i + 1);
        let start = esp_hal::time::Instant::now();

        // 3. Run Prediction
        // The macro generates `predict` which takes the input by value.
        let _output_batch = UNet::predict(input_batch);

        let duration = start.elapsed();
        println!("Inference done in {} ms", duration.as_millis());
        
        unsafe {
            let peak_stack = measure_stack();
            println!("STACK_PEAK:{}", peak_stack);
            
            let stack_start = core::ptr::addr_of!(_stack_start) as usize;
            let stack_end = core::ptr::addr_of!(_stack_end) as usize;
            println!("STACK_TOTAL:{}", stack_start - stack_end);
        }
        
        delay.delay_millis(500);
    }
    
    println!("ANALYSIS_DONE");
    
    loop {
        delay.delay_millis(1000);
    }
}