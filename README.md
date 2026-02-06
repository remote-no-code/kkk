Based on your existing README for SiliconSight and industry best practices, here's a comprehensive, professional-grade README with an integrated virtual environment setup section: [purpletutor](https://purpletutor.com/python-virtual-environment-best-practices/)

```markdown
# SiliconSight
**Autonomous Semiconductor Wafer Defect Detection using Vision Transformers**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

**SiliconSight** is an autonomous computer vision system for **binary defect classification** in semiconductor wafer manufacturing [file:1].

The system performs wafer-level **PASS / REJECT** decisions by analyzing **wafer map representations**, a critical quality-control step in modern fabrication pipelines [file:1]. The primary objective is **zero missed defects (false negatives)** under strict industrial constraints [file:1].

Instead of traditional CNN-based approaches, SiliconSight leverages **Self-Supervised Vision Transformers (ViT)**—specifically **Meta's DINOv2**—to capture **global, non-linear spatial defect patterns** (rings, clusters, arcs, edge defects) that CNNs often struggle to model [file:1].

This repository demonstrates a **production-oriented, end-to-end ML lifecycle**, including [file:1]:

- Domain-aware preprocessing  
- Extreme class-imbalance handling  
- Transformer fine-tuning  
- Manufacturing-focused evaluation  
- Robustness stress testing  
- Near real-time inference simulation  

All experiments are optimized for **Apple Silicon (MPS)** and **NVIDIA CUDA** hardware [file:1].

---

## Table of Contents

- [Objective](#objective)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [Virtual Environment Setup](#-virtual-environment-setup-recommended)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Performance Benchmark](#performance-benchmark)
- [Limitations](#limitations)
- [Acknowledgements](#acknowledgements)
- [Contributing](#contributing)
- [License](#license)

---

## Objective

Autonomous **binary classification** of semiconductor wafers [file:1]:

- **Clean (PASS)**
- **Defected (REJECT)**

with **zero tolerance for false negatives** [file:1].

---

## Features

- ✅ **Vision Transformer backbone** (DINOv2) for global pattern recognition
- ✅ **100% accuracy** on balanced test set with perfect recall
- ✅ **~29ms inference latency** (~34 FPS) on Apple M4 MacBook Air
- ✅ **Robustness to Gaussian noise** (>99% accuracy up to σ ≈ 0.1–0.2)
- ✅ **Production-ready demo** with live wafer streaming simulation
- ✅ **Hardware acceleration** via MPS (Apple Silicon) and CUDA (NVIDIA)

---

## Prerequisites

- **Python**: 3.9 or higher [web:4]
- **Hardware**: 
  - Apple Silicon Mac (automatically uses MPS acceleration) [file:1]
  - NVIDIA GPU with CUDA support (optional) [file:1]
  - CPU fallback available
- **Operating System**: macOS, Linux, or Windows

---

## Installation

### 🧪 Virtual Environment Setup (Recommended)

To ensure dependency isolation and reproducibility, this project should be run inside a Python virtual environment [web:2][web:4]. Creating a virtual environment prevents conflicts with system-wide packages and guarantees consistent behavior across machines [web:4].

#### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/siliconsight.git
cd siliconsight
```

#### Step 2: Create a Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv venv
```

**Windows:**
```bash
python -m venv venv
```

#### Step 3: Activate the Virtual Environment

**macOS / Linux:**
```bash
source venv/bin/activate
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```bash
venv\Scripts\activate.bat
```

After activation, your terminal prompt should display `(venv)` [web:4].

#### Step 4: Upgrade pip and Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 5: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

If this runs without errors and displays hardware acceleration status, the environment is set up correctly [web:4].

#### Step 6: Deactivate the Environment (When Done)

```bash
deactivate
```

> **📌 Important Notes:**
> - Always activate your environment before running any scripts [web:6]
> - Add `venv/` to your `.gitignore` file to avoid committing environment files [web:9]
> - Apple Silicon users automatically use MPS acceleration if available [file:1]
> - NVIDIA users will use CUDA if properly installed [file:1]

---

## Dataset

### Source
- **File**: `Wafer_Map_Datasets.npz`
- **Total Samples**: ~38,000 wafers
- **Class Distribution**: ~95% Defected, <5% Clean [file:1]

### Structure

| Array  | Description                          |
|-------:|--------------------------------------|
| `arr_0`| Wafer maps (2D integer grids)        |
| `arr_1`| Defect annotations (multi-label)     |

Each wafer map is a sparse categorical matrix [file:1]:

| Value | Meaning        |
|------:|----------------|
| 0     | Background     |
| 1     | Normal Die     |
| 2     | Defect         |

---

## Repository Structure

```
DEEPTECH/
├── demo/
│   ├── assets/
│   │   ├── clean/           # Archived clean wafers
│   │   └── defected/        # Archived defective wafers
│   └── demo.py              # Production-line simulation
├── figures/                 # Generated visualizations
├── hackathon_model/         # Trained model checkpoints
├── results/                 # Evaluation outputs
├── src/
│   ├── dataset.py           # Dataset loader
│   ├── model.py             # DINOv2 + classification head
│   ├── preprocess.py        # RGB encoding & preprocessing
│   ├── train.py             # Training pipeline
│   ├── inference.py         # Single wafer inference
│   ├── evaluate.py          # Metrics computation
│   ├── generate_figures.py # Visualization generation
│   └── generate_results.py # Full evaluation report
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
```

---

## Usage

### Train the Model

```bash
python -m src.train
```

Trains DINOv2 on balanced wafer dataset [file:1]. Model checkpoints are saved to `./hackathon_model/` [file:1].

### Evaluate Performance

```bash
python src/generate_results.py
```

Generates comprehensive evaluation metrics and confusion matrix [file:1]. Results are saved to `results/` [file:1].

### Generate Figures

```bash
python src/generate_figures.py
```

Creates ROC curves, confidence histograms, and visualization plots [file:1].

### Run Production Demo

```bash
python demo/demo.py
```

Launches live wafer streaming simulation with dual-view rendering [file:1]. Predictions are archived to `demo/assets/` [file:1].

### Single Wafer Inference

```bash
python src/inference.py --random
```

Performs inference on a randomly selected wafer from the test set [file:1].

---

## Model Architecture

### Backbone
- **Model**: `facebook/dinov2-base` [file:1]
- **Architecture**: Vision Transformer (ViT) [file:1]
- **Training Paradigm**: Self-Supervised Learning (DINO) [file:1]

### Why DINOv2?
DINOv2 excels at wafer map defect analysis because it [file:1]:
- Learns **global structural features**
- Avoids over-reliance on local texture
- Excels at **non-local pattern detection**

### Classification Head
- Linear layer on the `[CLS]` token [file:1]
- Output: 2 logits → Clean / Defected [file:1]

### Training Configuration

| Component        | Value          |
|------------------|----------------|
| Loss             | Cross-Entropy  |
| Optimizer        | AdamW          |
| Learning Rate    | 2e-5           |
| Batch Size       | 32             |
| Epochs           | 3              |
| Hardware         | MPS / CUDA     |

*A low learning rate preserves pretrained DINOv2 representations while adapting to domain-specific patterns* [file:1].

---

## Evaluation

### Metrics (Balanced Test Set)

| Metric                                 | Value   |
| -------------------------------------- | ------- |
| Accuracy                               | 100.00% |
| Precision                              | 1.0000  |
| Recall (Sensitivity)                   | 1.0000  |
| F1 Score                               | 1.0000  |
| Matthews Correlation Coefficient (MCC) | 1.0000  |

**Primary metric**: MCC, due to robustness under class imbalance [file:1].

### Diagnostic Outputs
- Confusion Matrix (zero misclassifications) [file:1]
- ROC Curve (AUC = 1.00) [file:1]
- Prediction confidence histograms [file:1]

All artifacts are stored in `results/` [file:1].

---

## Performance Benchmark

**Hardware**: Apple M4 MacBook Air (MPS) [file:1]

| Metric            | Value          |
| ----------------- | -------------- |
| Inference Latency | ~29 ms / wafer |
| Throughput        | ~34 FPS        |

Suitable for **near real-time production screening** [file:1].

### Robustness Stress Testing

Gaussian noise injection during inference [file:1]:
- Accuracy remained >99% up to σ ≈ 0.1–0.2 [file:1]
- Confirms reliance on **structural patterns** rather than pixel artifacts [file:1]

---

## Limitations

- Operates on wafer map data (not raw microscopy) [file:1]
- Binary classification only [file:1]
- Dataset-specific preprocessing [file:1]
- Generalization may vary across fabs and processes [file:1]

---

## Acknowledgements

This project makes use of publicly available wafer map datasets from the following repositories [file:1]:

- **WaferMap (MixedWM38 Dataset)**  
  Repository: [https://github.com/Junliangwangdhu/WaferMap](https://github.com/Junliangwangdhu/WaferMap)  
  Provides wafer map data with multiple single and mixed defect patterns [file:1].

These datasets were used **strictly for research and educational purposes** [file:1].  
All rights and credits belong to the original authors and contributors [file:1].

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add YourFeature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

For major changes, please open an issue first to discuss proposed changes [web:11].

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

**Project Maintainer**: [Your Name]  
**Email**: your.email@example.com  
**Project Link**: [https://github.com/yourusername/siliconsight](https://github.com/yourusername/siliconsight)

---

<div align="center">
  <sub>Built with ❤️ for the semiconductor industry</sub>
</div>
```

## Key Improvements

This industry-standard README incorporates: [github](https://github.com/RichardLitt/standard-readme)

### Structure Enhancements
- **Table of Contents** for easy navigation [hatica](https://www.hatica.io/blog/best-practices-for-github-readme/)
- **Badges** showing Python version, framework, and license compatibility [github](https://github.com/RichardLitt/standard-readme)
- **Clear sections** following the Diátaxis framework (learning, problem-solving, information, understanding) [realpython](https://realpython.com/python-project-documentation-with-mkdocs/)

### Virtual Environment Section
- **Step-by-step instructions** for all major operating systems [oneuptime](https://oneuptime.com/blog/post/2026-01-24-create-virtual-environments-python/view)
- **Verification command** to check installation success [oneuptime](https://oneuptime.com/blog/post/2026-01-24-create-virtual-environments-python/view)
- **Best practices** prominently displayed (gitignore, activation reminders) [codefixeshub](https://www.codefixeshub.com/programming/python-virtual-environment-best-practices-a-compre)
- **Hardware acceleration** detection included in verification [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/72371279/8b87df89-9f13-499e-b273-5d9c3094fdfb/README-10.md)

### Professional Features
- **Features section** with checkmarks highlighting key capabilities [hatica](https://www.hatica.io/blog/best-practices-for-github-readme/)
- **Repository structure** with inline comments explaining each directory [github](https://github.com/RichardLitt/standard-readme)
- **Contributing guidelines** to encourage open-source collaboration [realpython](https://realpython.com/documenting-python-code/)
- **Contact information** section for maintainer details [hatica](https://www.hatica.io/blog/best-practices-for-github-readme/)
- **Consistent formatting** using tables, code blocks, and proper Markdown hierarchy [hatica](https://www.hatica.io/blog/best-practices-for-github-readme/)

The virtual environment setup is now prominently integrated into the Installation section with platform-specific commands, following 2026 best practices. [purpletutor](https://purpletutor.com/python-virtual-environment-best-practices/)
