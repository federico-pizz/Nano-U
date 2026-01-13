#![no_std]
#![no_main]

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
#[model("models/dummy1.tflite")]
struct UNet;

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
    println!("Allocating Input (27KB)...");
    
    // Create a 48x48 matrix where each element is [f32; 3] (RGB)
    // We initialize it with dummy values (e.g., 0.5)
    let input_image = Buffer2D::<[f32; 3], 48, 48>::from_element([0.5, 0.5, 0.5]);
    
    // Microflow expects a 4D Batch: [Batch, Row, Col, Channel] -> [1, 48, 48, 3]
    let input_batch: Buffer4D<f32, 1, 48, 48, 3> = [input_image];

    loop {
        println!("Running Inference...");
        let start = esp_hal::time::Instant::now();

        // 3. Run Prediction
        // The macro generates `predict` which takes the input by value.
        let output_batch = UNet::predict(input_batch);

        let duration = start.elapsed();
        println!("Inference done in {} ms", duration.as_millis());

        // 4. Process Output (48x48x1)
        // output_batch is [Buffer2D<[f32; 1], 48, 48>; 1]
        let output_image = output_batch[0]; // Get the first image in batch
        
        // Example: Print the center pixel value
        let center_pixel = output_image[(24, 24)]; // (Row, Col)
        println!("Center Pixel Prediction: {:.4}", center_pixel[0]);

        delay.delay_millis(1000);
    }
}