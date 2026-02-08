# Silicon Sight: Automated Semiconductor Defect Detection

**Autonomous Semiconductor Wafer Defect Detection using Vision Transformers**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

--- 

##  Table of Contents

- [Problem Statement](#problem-statement)
- [Our Solution](#our-solution)
- [System Architecture](#system-architecture)
- [Technical Deep Dive](#advanced-technical-deep-dive)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Results](#results)
- [Demo](#demo)
- [Deployment](#deployment)

***

##  Problem Statement

### The Semiconductor Manufacturing Challenge

Semiconductor fabrication faces critical inspection bottlenecks:

- **Scale**: Defects occur at nanometer scale (human hair is 75,000× larger than chip features)
- **Cost**: A single 300mm wafer contains hundreds of chips worth tens of thousands of dollars
- **Yield Impact**: Even 1% yield improvement translates to millions in saved revenue
- **Speed vs. Accuracy**: Manual inspection is slow; rule-based AOI systems have high false positive rates (15-25%)

### Defect Types

- **Global Defects**: Wafer-wide issues (spin coating errors, uniformity problems)
- **Local Defects**: Isolated anomalies (particles, scratches, stains)
- **Topological Defects**: Connectivity issues (bridges, opens, voids)

***

## Our Solution

**Silicon Sight** leverages Vision Transformers with Meta's DinoV2 backbone to detect semiconductor wafer defects with high accuracy and production-ready inference speeds.

### Key Features

- **Vision Transformer Architecture**: Global attention mechanism for detecting defects spanning entire wafer
- **DinoV2 Backbone**: Self-supervised foundation model pre-trained on 142M images
- **Multi-Class Classification**: Detects clean wafers and various defect types
- **ONNX Deployment**: Cross-platform inference with optimized runtime
- **Production-Ready**: Complete evaluation suite with metrics, visualizations, and demo interface

***

## System Architecture

### End-to-End Pipeline

```
Input Image (RGB)
      ↓
Preprocessing (src/preprocess.py)
  • Resize to 224×224
  • Normalization: (X - μ) / σ
  • Tensor conversion
      ↓
Vision Transformer (src/model.py)
  • Patch Embedding (16×16 patches)
  • DinoV2 Encoder (12 layers)
  • Multi-Head Self-Attention
  • Global receptive field
      ↓
Classification Head
  • [CLS] token extraction
  • Linear classifier
  • Softmax probabilities
      ↓
Output: Class + Confidence Score
```

### Model Architecture

**DinoV2 Vision Transformer**
- **Backbone**: facebook/dinov2-base
- **Parameters**: 86M
- **Layers**: 12 transformer blocks
- **Hidden Dimension**: 768
- **Attention Heads**: 12
- **Patch Size**: 16×16
- **Input Size**: 224×224
- **Pre-training**: Self-supervised on 142M images

***

##  Advanced Technical Deep Dive

### 1. Why Vision Transformers Over CNNs?

#### The CNN Limitation

Standard CNNs use convolutions with local receptive fields:

\[
Y(i,j) = \sum_{m} \sum_{n} X(i+m, j+n) \cdot W(m,n)
\]

**Problem**: This operation is local. A pixel at position (0,0) has no information about pixel (224,224) until very deep layers. For semiconductor defects that span the entire wafer (scratches, cracks), CNNs may see disconnected segments.

#### The Transformer Solution

Vision Transformers use Self-Attention for global context:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

**Key Components**:
- **Q (Query)**: "What am I looking for?"
- **K (Key)**: "What features do I have?"
- **V (Value)**: "What information should I pass forward?"

**Advantage**: Every patch (16×16 pixels) attends to every other patch simultaneously. A 224×224 image has 196 patches, creating 196×196 = 38,416 attention interactions per layer. This global receptive field from Layer 1 enables detection of wafer-spanning defects.

### 2. DinoV2: Self-Supervised Learning

**Training Paradigm**: Unlike supervised models trained on labeled data, DinoV2 uses self-distillation:

1. Input image X → Generate augmented views:
   - Global view (large crop, 224×224)
   - Local views (small crops, 96×96)

2. Pass through Student and Teacher networks

3. **Objective**: Student predictions match Teacher predictions

\[
\mathcal{L}_{\text{DINO}} = - \sum_{x \in \{x_1, x_2\}} \sum_{i} P_{\text{teacher}}^{(i)} \log P_{\text{student}}^{(i)}
\]

**Why This Matters**:
- **Part-Whole Understanding**: Model learns that small curves (local view) belong to circular particles (global view)
- **Texture Discrimination**: Distinguishes harmless patterns from structural defects
- **Transfer Learning**: 142M pre-training images provide robust features for wafer inspection with minimal fine-tuning data

### 3. Training Dynamics

#### Loss Function: Cross-Entropy

\[
\mathcal{L}_{\text{CE}} = - \sum_{c=1}^{C} y_{c} \log(p_{c})
\]

Where:
- \(y_c\): Ground truth (one-hot encoded)
- \(p_c\): Predicted probability for class c
- \(C\): Number of classes

**Interpretation**: Penalizes confident wrong predictions exponentially.

#### Optimizer: AdamW

\[
\theta_{t+1} = \theta_t - \eta \left( \frac{m_t}{\sqrt{v_t} + \epsilon} + \lambda \theta_t \right)
\]

Where:
- \(m_t\): First moment (momentum)
- \(v_t\): Second moment (adaptive learning rate)
- \(\lambda\): Weight decay (regularization)

**Benefits**:
- **Momentum** (β₁=0.9): Smooths gradient direction
- **Adaptive LR** (β₂=0.999): Per-parameter learning rates
- **Weight Decay**: Prevents overfitting by penalizing large weights

#### Learning Rate Schedule: Cosine Annealing

\[
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t\pi}{T}\right)\right)
\]

**Rationale**: Large initial steps for exploration, small final steps for fine-tuning.

### 4. Image Preprocessing

**Normalization**:
\[
X_{\text{normalized}} = \frac{X - \mu}{\sigma}
\]

Using ImageNet statistics (DinoV2 pre-training domain):
```python
mean = [0.485, 0.456, 0.406]  # RGB
std  = [0.229, 0.224, 0.225]  # RGB
```

**Purpose**:
- Zero-centered inputs → Faster gradient descent
- Unit variance → Prevents exploding/vanishing gradients
- Distribution matching → Aligns with pre-training

### 5. ONNX Optimization

#### Operator Fusion

**Before**: 3 separate operations
```
Conv2D(x) → Add(bias) → ReLU()
```

**After**: 1 fused operation
```
ConvBiasReLU(x)
```

**Benefits**:
- Memory bandwidth: 66% reduction
- Kernel launch overhead: Eliminated
- Cache efficiency: Data stays in GPU cache

#### Quantization (FP32 → INT8)
$$
\[
x_{\text{int8}} = \text{round}\left(\frac{x_{\text{fp32}}}{s}\right) + z
\]
$$
Where:
- \(s\): Scale factor
- \(z\): Zero-point offset

**Trade-off**: 4× speedup with minimal accuracy loss (<1%)

### 6. Comparison: Silicon Sight vs Traditional AOI

| Feature | Rule-Based AOI | CNN (ResNet-50) | Silicon Sight (ViT) |
|---------|----------------|-----------------|---------------------|
| **Method** | Golden template matching | Local feature extraction | Global attention |
| **Logic** | Threshold-based rules | Hierarchical edges | Semantic patterns |
| **Setup Time** | 4-8 hours (manual) | 2-3 hours | <30 minutes |
| **Lighting Robustness** |  Very sensitive | Moderate |  High (normalization) |
| **Novel Defects** |  Cannot detect |  Needs retraining |  Transfer learning |
| **Explainability** |  Full (explicit rules) |  Limited |  Attention maps |

***

## Project Structure

```
silicon-sight/
│
├── data/                              # Dataset directory
│   ├── rejected content loaded/       # Raw data processing
│   ├── sample_wafers/                 # Training samples
│   │   ├── clean/                     # Non-defective wafers
│   │   └── defected/                  # Defective wafers
│   └── dataset.zip                    # Complete dataset archive
│
├── demo/                              # Demo application
│   ├── assets/
│   │   ├── accepted/                  # Passed inspection
│   │   └── rejected/                  # Failed inspection
│   ├── inspection_report.html         # Auto-generated report
│   └── demo.py                        # Demo script
│
├── figures/                           # Generated visualizations
│   ├── data_split.png                 # Train/val/test distribution
│   ├── dataset_balance.png            # Class distribution
│   ├── human_vs_machine.png           # Performance comparison
│   └── preprocessing_pipeline.png     # Pipeline diagram
│
├── hackathon_model/                   # Trained model
│   ├── config.json                    # Model configuration
│   └── model.safetensors              # Model weights
│
├── onnx/                              # Deployment files
│   ├── SiliconSight.onnx              # ONNX model
│   └── SiliconSight.onnx.data         # Model weights
│
├── results/                           # Evaluation results
│   ├── confidence_histogram.png       # Confidence distribution
│   ├── confusion_matrix.png           # Classification errors
│   ├── metrics.json                   # Numerical metrics
│   └── roc_curve.png                  # ROC curves
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── convert_to_onnx.py             # PyTorch → ONNX conversion
│   ├── dataset.py                     # Data loading
│   ├── generate_figures.py            # Visualization generation
│   ├── generate_results.py            # Results compilation
│   ├── inference.py                   # Inference engine
│   ├── model.py                       # Model architecture
│   ├── preprocess.py                  # Image preprocessing
│   ├── train.py                       # Training loop
│   └── verify_onnx.py                 # ONNX verification
│
├── .gitattributes
├── LICENSE
├── README.md
└── requirements.txt
```

***

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/silicon-sight.git
cd silicon-sight

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
python src/train.py \
    --data_dir data/sample_wafers \
    --output_dir hackathon_model \
    --batch_size 32 \
    --epochs 50 \
    --learning_rate 2e-5
```

### Inference

```python
from src.inference import SiliconSightInference

# Load model
model = SiliconSightInference("hackathon_model/model.safetensors")

# Predict
result = model.predict("path/to/wafer_image.jpg")
print(f"Class: {result['class']}, Confidence: {result['confidence']:.2%}")
```

### ONNX Export

```bash
# Convert to ONNX
python src/convert_to_onnx.py \
    --model_path hackathon_model/model.safetensors \
    --output_path onnx/SiliconSight.onnx

# Verify ONNX model
python src/verify_onnx.py \
    --pytorch_model hackathon_model/model.safetensors \
    --onnx_model onnx/SiliconSight.onnx
```

***

## Results

### Evaluation Metrics

All quantitative results are available in `results/metrics.json`.

### Visualizations

**Confusion Matrix** (`results/confusion_matrix.png`)
- Shows per-class classification performance
- Identifies common misclassification patterns

**ROC Curves** (`results/roc_curve.png`)
- Per-class ROC curves
- Area Under Curve (AUC) scores

**Confidence Distribution** (`results/confidence_histogram.png`)
- Distribution of model confidence scores
- Helps set confidence thresholds for production

**Dataset Analysis** (`figures/`)
- `data_split.png`: Train/validation/test split visualization
- `dataset_balance.png`: Class distribution analysis
- `preprocessing_pipeline.png`: Data preprocessing flow

***

##  Demo

### Run Demo Application

```bash
python demo/demo.py
```

The demo provides:
- Interactive image upload and inference
- Real-time prediction with confidence scores
- Auto-generated inspection reports (HTML format)
- Visual comparison of accepted vs rejected wafers

### Demo Output

The demo generates `inspection_report.html` containing:
- Wafer image
- Predicted class
- Confidence score
- Decision (Pass/Reject)
- Timestamp

***

##  Deployment

### ONNX Inference

```python
import onnxruntime as ort

# Load ONNX model
session = ort.InferenceSession("onnx/SiliconSight.onnx")

# Run inference
outputs = session.run(None, {"input": preprocessed_image})
predictions = outputs[0]
```

### Edge Deployment

The ONNX model can be deployed on:
- NVIDIA Jetson devices (GPU acceleration)
- Raspberry Pi (CPU inference)
- Industrial PCs
- Cloud platforms (AWS, Azure, GCP)

### Production Considerations

- **Batch Processing**: Process multiple wafers simultaneously
- **Confidence Thresholds**: Set thresholds for auto-accept/reject
- **Human-in-Loop**: Flag low-confidence predictions for manual review
- **Logging**: Track predictions for continuous improvement

***

## Technical Stack

- **Deep Learning**: PyTorch 2.0+
- **Model**: DinoV2 (Meta AI), Transformers (Hugging Face)
- **Deployment**: ONNX Runtime
- **Data Processing**: PIL, NumPy, Albumentations
- **Evaluation**: scikit-learn
- **Visualization**: Matplotlib, Seaborn

***

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

***

## Acknowledgments

- **Meta AI Research**: DinoV2 foundation model
- **Hugging Face**: Transformers library
- **PyTorch Team**: Deep learning framework
- **ONNX Runtime**: Cross-platform inference

***

