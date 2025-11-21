# Nano-U: Efficient Semantic Segmentation for Robotic Navigation in Outdoor Environments

[![Nano-U Presentation Slide](path/to/title_slide_image.png)](path/to/Tesi_presentazione.pdf)  
*Click the image above to view the full thesis presentation PDF.*

## Overview

This repository contains the implementation and extensions of the Nano-U model, developed as part of my thesis at the University of Padua, Department of Information Engineering (DEI). The project focuses on efficient semantic segmentation for estimating traversable ground in outdoor environments, optimized for resource-constrained microcontrollers like the ESP32.

Key features:
- Lightweight CNN model for on-device inference.
- Extensions including Rust-based embeddings, model retraining, and code refinements.
- Integration with MicroFlow library for microcontroller deployment.

Thesis details:
- **Author**: Federico Pizzolato
- **Supervisor**: Prof. Nicola Bellotto
- **Co-Supervisor**: Francesco Pasti
- **Academic Year**: 2024/2025

[Link to full thesis document (if available)](path/to/full_thesis.pdf)  
[Link to presentation PDF](path/to/Tesi_presentazione.pdf)

## Problem and Motivations

Semantic segmentation in outdoor environments is challenging due to variability in terrain, lighting, and obstacles. This project addresses:

- **Why is it a complex problem?**  
  [Briefly describe the complexities, e.g., real-time processing, diverse outdoor scenes. Include motivations from slide 2.]

- **Why on microcontrollers?**  
  [Explain the need for edge computing on low-power devices like ESP32 for robotic navigation. Reference motivations from slide 2.]

![Motivations Slide](path/to/motivations_slide_image.png)

## Objectives and Contributions

### Objectives
- Estimate traversable ground "on-device".
- Adhere to ESP32 resource constraints.
- Maintain sufficient quality for autonomous navigation.

### Contributions
- **Nano-U**: A CNN model for segmentation.
- Evaluation of compression techniques' effectiveness.
- Extension of the MicroFlow library.

![Objectives Slide](path/to/objectives_slide_image.png)

## Design of Nano-U

Nano-U is derived from a standard U-Net architecture, optimized for efficiency:

- **BU-Net**: Standard U-Net, effective but resource-intensive.
- **Nano-U**: Applies compound scaling and removes skip layers.
- **Training**: Uses knowledge distillation.
- **Nano-U in int8**: Conversion from PyTorch to TensorFlow, followed by post-training quantization (PTQ calibrated).

[Include diagrams or flowcharts from slide 4, e.g., training/validation loss graphs.]

![Design Slide](path/to/design_slide_image.png)

## Implementation in MicroFlow

- **Pre-processing and Input Handling**: [Describe input pre-processing steps.]
- **MicroFlow Features**:
  - Inference engine.
  - Static allocations.
  - Isolated kernels.
  - Essential TFLite operators.
  - Support for RESIZE BILINEAR operator.

[Add details on how the model is deployed on microcontrollers. Mention any extensions you've made.]

![Implementation Slide](path/to/implementation_slide_image.png)

### Rust Embeddings Extension
[Placeholder for your additions: Describe the integration of embeddings in Rust, why Rust was chosen (e.g., performance, safety), and how it interfaces with the existing Python/TensorFlow code.]

### Model Retraining
[Placeholder: Detail the retraining process, including new datasets, hyperparameters, and improvements over the original model.]

### Code Refinements
[Placeholder: List general code improvements, such as optimizations, bug fixes, or modularization.]

## Dataset and Experiments

- **Datasets Used**: [Describe datasets like Crops and Tomatoes from slide 6. Mention any new datasets added during retraining.]
- **Experiments**: Predictions on complex (e.g., Crops) and simpler (e.g., Tomatoes) frames.

[Include example predictions or tables comparing model outputs.]

![Dataset Slide](path/to/dataset_slide_image.png)

## Results

[Summarize measurements from slide 7, e.g., model size, inference time, IoU comparisons.]

- **Model Performance Comparison**:  
  | Model       | Size (KB) | Inference Time (ms) | Mean IoU |
  |-------------|-----------|---------------------|----------|
  | BU-Net     | [Value]  | [Value]            | [Value] |
  | Nano-U     | [Value]  | [Value]            | [Value] |
  | Nano-U int8| [Value]  | [Value]            | [Value] |

- **IoU for MicroFlow Predictions**: [Describe graph trends.]

![Results Slide](path/to/results_slide_image.png)

## Conclusions and Future Work

On-device ground segmentation is feasible with good accuracy, lightweight models, and minimal runtime footprint.

### Future Projects
- More varied datasets.
- Use Neural Architecture Search (NAS) techniques.
- Experiment with advanced quantizations like f4 (further MicroFlow extension).

### Possible Applications
- Outdoor robotic navigation (thesis goal).
- Agricultural monitoring (with microcontrollers).
- Surveillance of "no-go" areas (unstructured environments).

[Add any new conclusions from your extensions.]

![Conclusions Slide](path/to/conclusions_slide_image.png)

## Installation

### Prerequisites
- Python 3.x
- Rust (for embeddings extension)
- [List libraries: e.g., PyTorch, TensorFlow, TensorFlow Lite, etc.]
- ESP32 development tools (if deploying on hardware)

### Setup
1. Clone the repository:
  git clone https://github.com/yourusername/nano-u-repo.git
  cd nano-u-repo

2. Install Python dependencies:
   pip install -r requirements.txt

3. For Rust components:
   cargo build --release

[Add hardware setup instructions for ESP32 if applicable.]

## Usage

### Running the Model
[Provide example commands, e.g.:]
python scripts/train.py --dataset path/to/dataset
python scripts/infer.py --model path/to/model.tflite --image path/to/test_image.png


### Deploying on ESP32
[Describe steps to flash the model using MicroFlow.]

### Example Output
[Placeholder for sample segmentation results.]

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

[Add any specific guidelines, e.g., coding standards.]

## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## Acknowledgments

- University of Padua, DEI.
- Prof. Nicola Bellotto and Francesco Pasti for supervision.
- [Any other credits, e.g., libraries or datasets used.]

[Feel free to add badges, e.g., for build status or coverage if you set them up.]
