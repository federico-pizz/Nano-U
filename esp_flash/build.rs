use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    // make sure linkall.x is the last linker script (otherwise might cause problems with flip-link)
    println!("cargo:rustc-link-arg=-Tlinkall.x");

    // ── Paths from Python pipeline (relative to esp_flash/ crate root) ─────────
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let project_root = manifest_dir
        .parent()
        .expect("esp_flash must be inside project root");

    // Model: use the quantized TFLite produced by src/quantize_model.py
    let src_model = project_root.join("models").join("nano_u.tflite");
    let dst_model = manifest_dir.join("models").join("nano_u.tflite");
    fs::create_dir_all(dst_model.parent().unwrap()).expect("Cannot create models/ dir");
    if src_model.exists() {
        fs::copy(&src_model, &dst_model).unwrap_or_else(|e| {
            panic!(
                "Failed to copy {} → {}: {}",
                src_model.display(),
                dst_model.display(),
                e
            )
        });
        println!(
            "cargo:warning=Copied model: {} → {}",
            src_model.display(),
            dst_model.display()
        );
    } else {
        println!(
            "cargo:warning=Source model not found at {}; using existing esp_flash/models/nano_u.tflite",
            src_model.display()
        );
    }
    println!("cargo:rerun-if-changed={}", src_model.display());

    // ── Pack test images into a flat binary (H=96, W=128, C=3, u8) ────────────
    // Images come from the Python processed data path defined in config/config.yaml
    let test_img_dir = project_root
        .join("data")
        .join("botanic_garden")
        .join("test")
        .join("img");

    println!("cargo:rerun-if-changed={}", test_img_dir.display());

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bin_path = out_dir.join("input_images.bin");
    let mut bin_file = fs::File::create(&bin_path).expect("Cannot create input_images.bin");

    // Image dimensions must match the model's expected input (config.yaml: input_shape: [60, 80, 3])
    const IMG_H: u32 = 60;
    const IMG_W: u32 = 80;
    const MAX_IMAGES: usize = 50;

    let mut count = 0usize;

    if test_img_dir.exists() {
        let mut entries: Vec<PathBuf> = fs::read_dir(&test_img_dir)
            .expect("Cannot read test img dir")
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .map(|ext| ext.eq_ignore_ascii_case("png"))
                    .unwrap_or(false)
            })
            .collect();
            
        // Extract the integer from the filename (e.g. image_12.png -> 12) to match python's sorted_by_frame
        entries.sort_by_key(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.split('_').last())
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(0)
        });

        for path in entries.iter().take(MAX_IMAGES) {
            // Decode PNG, resize to model input dimensions, write raw RGB bytes
            let img = image::open(path)
                .unwrap_or_else(|e| panic!("Cannot open {}: {}", path.display(), e))
                .resize_exact(IMG_W, IMG_H, image::imageops::FilterType::Lanczos3)
                .to_rgb8();
            bin_file
                .write_all(img.as_raw())
                .expect("Cannot write image bytes");
            count += 1;
            println!("cargo:rerun-if-changed={}", path.display());
        }
    } else {
        println!(
            "cargo:warning=Test image dir not found: {}; input_images.bin will be all zeros",
            test_img_dir.display()
        );
    }

    // Pad with zero-images if fewer than MAX_IMAGES were found
    let bytes_per_image = (IMG_H * IMG_W * 3) as usize;
    if count < MAX_IMAGES {
        let padding = vec![0u8; bytes_per_image * (MAX_IMAGES - count)];
        bin_file.write_all(&padding).expect("Cannot write padding");
    }

    println!(
        "cargo:warning=Packed {} / {} test images into input_images.bin",
        count, MAX_IMAGES
    );
}
