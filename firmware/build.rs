use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    // make sure linkall.x is the last linker script (otherwise might cause problems with flip-link)
    println!("cargo:rustc-link-arg=-Tlinkall.x");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let models_dir_default = manifest_dir.parent().unwrap().join("models").join("BotanicGarden");
    let test_img_dir_default = manifest_dir.parent().unwrap().join("data").join("BotanicGarden").join("test").join("img");

    let models_dir = env::var("MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or(models_dir_default);
    println!("cargo:rerun-if-env-changed=MODELS_DIR");
    
    let test_img_dir = env::var("TEST_IMG_DIR")
        .map(PathBuf::from)
        .unwrap_or(test_img_dir_default);
    println!("cargo:rerun-if-env-changed=TEST_IMG_DIR");

    // ── Model Selection ──────────────────────────────────────────────────────────
    let model_name = env::var("MODEL_NAME").unwrap_or_else(|_| "nano_u".to_string());
    println!("cargo:rerun-if-env-changed=MODEL_NAME");
    println!("cargo:warning=Building for model: {}", model_name);

    // ── Quantization parameters ──────────────────────────────────────────────────
    let quant_params_path = models_dir.join(format!("{}_quant_params.json", model_name));
    println!("cargo:rerun-if-changed={}", quant_params_path.display());

    if quant_params_path.exists() {
        let raw = fs::read_to_string(&quant_params_path)
            .expect(&format!("Failed to read {}", quant_params_path.display()));
        
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
        panic!("CRITICAL: {} not found at {}! Run python quantization pipeline first to generate calibration parameters, otherwise the hardware image will silently quantize pixels with NaN errors.", model_name, quant_params_path.display());
    }

    let src_model = models_dir.join(format!("{}.tflite", model_name));
    let dst_model = manifest_dir.join("models").join("nano_u.tflite"); // Keep same destination name for simplicity in code
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
    println!("cargo:rerun-if-env-changed=MODEL_NAME");

    println!("cargo:rerun-if-changed={}", test_img_dir.display());

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bin_path = out_dir.join("input_images.bin");
    let mut bin_file = fs::File::create(&bin_path).expect("Cannot create input_images.bin");
    
    let target_idx = env::var("TARGET_IMG_IDX").unwrap_or("1".to_string());
    println!("cargo:rustc-env=NANO_U_TARGET_IMG_IDX={}", target_idx);
    println!("cargo:rerun-if-env-changed=TARGET_IMG_IDX");

    let img_h: u32 = env::var("NANO_U_IMG_H").unwrap_or("60".to_string()).parse().unwrap();
    let img_w: u32 = env::var("NANO_U_IMG_W").unwrap_or("80".to_string()).parse().unwrap();

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
                // strip leading non-digits so "frame500" -> "500", bare numbers pass through
                .map(|s| s.trim_start_matches(|c: char| !c.is_ascii_digit()))
                .and_then(|s| s.parse::<u32>().ok())
                .unwrap_or(0)
        });
        
        for path in entries {
            let img = image::open(&path)
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

    println!("cargo:warning=Packed {} images into input_images.bin", count);
    println!("cargo:rustc-env=NANO_U_NUM_IMAGES={}", count);
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