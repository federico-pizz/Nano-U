import os
import json
import glob
import numpy as np
import cv2

# --- CONFIGURATION ---
# Where your new unzipped folder is (contains .json and .tif files)
input_dir = "./1005_dalsa_garden_05_labeled" 
output_dir = "./dataset_ready_for_train"

# Define what classes count as "Walkable"
# (Run the scan script first if you aren't sure, but usually 'road', 'paved', 'path')
WALKABLE_CLASSES = ["road", "paved", "path", "grass", "flat_ground"]
# ---------------------

os.makedirs(f"{output_dir}/images", exist_ok=True)
os.makedirs(f"{output_dir}/masks", exist_ok=True)

# Find pairs of JSON and TIF
json_files = glob.glob(os.path.join(input_dir, "**/*.json"), recursive=True)

print(f"Found {len(json_files)} labeled samples. Processing...")

for j_path in json_files:
    # 1. Load the Label
    with open(j_path, 'r') as f:
        data = json.load(f)

    # 2. Load the Corresponding Image
    # The new zip has the .tif image right next to the .json
    image_path = j_path.replace(".json", ".tif")
    
    if not os.path.exists(image_path):
        print(f"Skipping {os.path.basename(j_path)} - Image not found!")
        continue

    # Load TIF image (unchanged to keep quality)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    # 3. Create the Binary Mask
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in data['shapes']:
        if shape['label'] in WALKABLE_CLASSES:
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255) # White = Walkable

    # 4. Save for Training
    base_name = os.path.basename(j_path).replace(".json", "")
    
    # Save Image as PNG (easier for PyTorch/Tensorflow to read than TIF)
    cv2.imwrite(f"{output_dir}/images/{base_name}.png", img)
    # Save Mask
    cv2.imwrite(f"{output_dir}/masks/{base_name}.png", mask)

print(f"Done! Check the '{output_dir}' folder.")