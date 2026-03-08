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

    // ── Quantization parameters ──────────────────────────────────────────────────
    let quant_params_path = project_root
        .join("models")
        .join("nano_u_quant_params.json");
    println!("cargo:rerun-if-changed={}", quant_params_path.display());

    if quant_params_path.exists() {
        let raw = fs::read_to_string(&quant_params_path)
            .expect("Failed to read nano_u_quant_params.json");
        
        let input_scale      = extract_json_f64(&raw, "input",  "scale");
        let input_zero_point = extract_json_i64(&raw, "input",  "zero_point");
        let output_scale     = extract_json_f64(&raw, "output", "scale");
        let output_zero_point= extract_json_i64(&raw, "output", "zero_point");

        let input_shape      = extract_json_array_i64(&raw, "input", "shape");
        let img_h = input_shape[1];
        let img_w = input_shape[2];

        let mean = extract_json_array_f64(&raw, "normalization", "mean");
        let std  = extract_json_array_f64(&raw, "normalization", "std");

        // Use bitwise float representations to avoid parser errors with scientific notation
        println!("cargo:rustc-env=NANO_U_INPUT_SCALE_BITS={}", (input_scale as f32).to_bits());
        println!("cargo:rustc-env=NANO_U_INPUT_ZERO_POINT={}", input_zero_point);
        println!("cargo:rustc-env=NANO_U_OUTPUT_SCALE_BITS={}", (output_scale as f32).to_bits());
        println!("cargo:rustc-env=NANO_U_OUTPUT_ZERO_POINT={}", output_zero_point);

        println!("cargo:rustc-env=NANO_U_IMG_H={img_h}");
        println!("cargo:rustc-env=NANO_U_IMG_W={img_w}");

        println!("cargo:rustc-env=NANO_U_MEAN_R_BITS={}", (mean[0] as f32).to_bits());
        println!("cargo:rustc-env=NANO_U_MEAN_G_BITS={}", (mean[1] as f32).to_bits());
        println!("cargo:rustc-env=NANO_U_MEAN_B_BITS={}", (mean[2] as f32).to_bits());
        println!("cargo:rustc-env=NANO_U_STD_R_BITS={}",  (std[0] as f32).to_bits());
        println!("cargo:rustc-env=NANO_U_STD_G_BITS={}",  (std[1] as f32).to_bits());
        println!("cargo:rustc-env=NANO_U_STD_B_BITS={}",  (std[2] as f32).to_bits());
        
        println!(
            "cargo:warning=Quant params loaded: input(scale={}, zp={}, {}x{}) output(scale={}, zp={})",
            input_scale, input_zero_point, img_w, img_h, output_scale, output_zero_point
        );
        println!(
            "cargo:warning=Normalization: mean={:?} std={:?}", mean, std
        );
    } else {
        println!("cargo:rustc-env=NANO_U_INPUT_SCALE_BITS=0");
        println!("cargo:rustc-env=NANO_U_INPUT_ZERO_POINT=0");
        println!("cargo:rustc-env=NANO_U_OUTPUT_SCALE_BITS=0");
        println!("cargo:rustc-env=NANO_U_OUTPUT_ZERO_POINT=0");
        println!("cargo:rustc-env=NANO_U_IMG_H=60");
        println!("cargo:rustc-env=NANO_U_IMG_W=80");
        println!("cargo:rustc-env=NANO_U_MEAN_R_BITS=0");
        println!("cargo:rustc-env=NANO_U_MEAN_G_BITS=0");
        println!("cargo:rustc-env=NANO_U_MEAN_B_BITS=0");
        println!("cargo:rustc-env=NANO_U_STD_R_BITS=0");
        println!("cargo:rustc-env=NANO_U_STD_G_BITS=0");
        println!("cargo:rustc-env=NANO_U_STD_B_BITS=0");
        println!(
            "cargo:warning=nano_u_quant_params.json not found at {}; \
             run `python -m src.quantize_model` to generate it.",
            quant_params_path.display()
        );
    }

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
        println!("cargo:warning=Copied model: {} → {}", src_model.display(), dst_model.display());
    }
    println!("cargo:rerun-if-changed={}", src_model.display());

    let test_img_dir = project_root
        .join("data")
        .join("botanic_garden")
        .join("test")
        .join("img");

    println!("cargo:rerun-if-changed={}", test_img_dir.display());

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bin_path = out_dir.join("input_images.bin");
    let mut bin_file = fs::File::create(&bin_path).expect("Cannot create input_images.bin");

    let img_h: u32 = env::var("NANO_U_IMG_H").unwrap_or("60".to_string()).parse().unwrap();
    let img_w: u32 = env::var("NANO_U_IMG_W").unwrap_or("80".to_string()).parse().unwrap();
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
            
        entries.sort_by_key(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .and_then(|s| s.split('_').last())
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(0)
        });

        for path in entries.iter().take(MAX_IMAGES) {
            // FIX: Using FilterType::Triangle (Bilinear) to match OpenCV
            let img = image::open(path)
                .unwrap_or_else(|e| panic!("Cannot open {}: {}", path.display(), e))
                .resize_exact(img_w, img_h, image::imageops::FilterType::Triangle)
                .to_rgb8();
            bin_file
                .write_all(img.as_raw())
                .expect("Cannot write image bytes");
            count += 1;
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }

    let bytes_per_image = (img_h * img_w * 3) as usize;
    if count < MAX_IMAGES {
        let padding = vec![0u8; bytes_per_image * (MAX_IMAGES - count)];
        bin_file.write_all(&padding).expect("Cannot write padding");
    }

    println!("cargo:warning=Packed {} / {} test images into input_images.bin", count, MAX_IMAGES);
}

// ── Lightweight JSON helpers ─────
fn extract_json_f64(json: &str, section: &str, field: &str) -> f64 {
    let section_marker = format!("\"{}\"", section);
    let field_marker   = format!("\"{}\"", field);

    let sec_start = json
        .find(section_marker.as_str())
        .unwrap_or_else(|| panic!("JSON section '{section}' not found"));

    let field_start = json[sec_start..]
        .find(field_marker.as_str())
        .unwrap_or_else(|| panic!("JSON field '{field}' not found in section '{section}'"));
    let after_colon = json[sec_start + field_start..]
        .find(':')
        .unwrap_or_else(|| panic!("':' not found after '{field}'"));
    let value_start = sec_start + field_start + after_colon + 1;

    let raw = json[value_start..].trim_start();
    let end = raw
        .find(|c: char| c == ',' || c == '}' || c == '\n' || c == ']')
        .unwrap_or(raw.len());

    raw[..end].trim().parse::<f64>().unwrap_or_else(|e| {
        panic!("Cannot parse '{section}.{field}' as f64: {e}")
    })
}

fn extract_json_i64(json: &str, section: &str, field: &str) -> i64 {
    extract_json_f64(json, section, field) as i64
}

fn extract_json_array_f64(json: &str, section: &str, field: &str) -> Vec<f64> {
    let section_marker = format!("\"{}\"", section);
    let field_marker   = format!("\"{}\"", field);

    let sec_start = json
        .find(section_marker.as_str())
        .unwrap_or_else(|| panic!("JSON section '{section}' not found"));

    let field_start = json[sec_start..]
        .find(field_marker.as_str())
        .unwrap_or_else(|| panic!("JSON field '{field}' not found in section '{section}'"));
    
    let array_start_rel = json[sec_start + field_start..]
        .find('[')
        .unwrap_or_else(|| panic!("'[' not found for '{field}'"));
    let array_end_rel = json[sec_start + field_start + array_start_rel..]
        .find(']')
        .unwrap_or_else(|| panic!("']' not found for '{field}'"));
    
    let content = &json[sec_start + field_start + array_start_rel + 1 .. sec_start + field_start + array_start_rel + array_end_rel];
    content.split(',')
        .map(|s| s.trim().parse::<f64>().expect("Failed to parse array element"))
        .collect()
}

fn extract_json_array_i64(json: &str, section: &str, field: &str) -> Vec<i64> {
    extract_json_array_f64(json, section, field).into_iter().map(|f| f as i64).collect()
}