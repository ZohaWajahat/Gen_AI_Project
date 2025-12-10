
# 🖼️ Project Title: Hybrid VLM for Zero-Shot Image Captioning Refinement

Status: Under Development / Research Project<br><br>

This repository contains the implementation and evaluation scripts for comparing two distinct approaches to image captioning: the ZeroCap Zero-Shot Baseline and a Fine-Tuned Hybrid Vision-Language Model (VLM) using Self-Critical Sequence Training (SCST). The project demonstrates how leveraging modern LLMs and targeted fine-tuning overcomes the primary limitations (fluency, speed) of older guidance-based zero-shot methods.<br><br>

## 🌟 Key Features
• **ZeroCap Baseline:** Implementation of the ZeroCap method using GPT-2 Medium and CLIP ViT-B/32 guided by inference-time logit modification.<br>
• **Hybrid VLM Pipeline:** Implementation of a modern VLM using a frozen Mistral-7B LLM and a frozen ViT-L/14 connected via a fine-tuned MLP Projector (Soft Prompting).<br>
• **Evaluation:** Scripts to assess both models using standard COCO metrics ($\text{CIDEr}$, $\text{SPICE}$, $\text{BLEU}$, $\text{METEOR}$) and the reference-free $\text{CLIPScore}$.<br>
• **Methodology:** Implementation of the two-stage Hybrid Alignment and Refinement (HAR) pipeline, including Maximum Likelihood Estimation (MLE) initialization and Self-Critical Sequence Training (SCST) refinement.<br><br>

## 🚀 Getting Started
### Prerequisites
• Python 3.8
• $\text{PyTorch}$
• $\text{Java Runtime Environment (JRE)}$ (Required for $\text{METEOR}$ and $\text{SPICE}$ evaluation metrics)

### Installation
#### 1. Clone the repository:
```powershell
git clone https://github.com/ZohaWajahat/Gen_AI_Project.git
cd Gen_AI_Project
```

#### 2. Create and activate the environment:
```powershell
conda create -n zerocap_hybrid python=3.10
conda activate zerocap_hybrid
```

#### 3. Install dependencies:
```powershell
pip install -r requirements.txt
# Assuming 'open_clip', 'tqdm', 'transformers', 'torch', 'pandas', 'pycocotools', 'pycocoevalcap' are listed in requirements.txt
```

#### 4. Download COCO Data: Download the $\text{COCO 2017}$ validation images and the $\text{captions\val2017.json}$ annotation file.
```powershell
# Create the data structure
mkdir -p data/coco/annotations
mkdir -p data/coco/val2017

# Place captions_val2017.json in data/coco/annotations/annotations/
# Place validation images in data/coco/val2017/
```

## 🏃 Usage and Evaluation
### 1. Run the ZeroCap Baseline
This script runs the token-by-token guidance method on the COCO validation set.

```powershell
python Zero_Cap_Model.py
```

Output: ./results/original_zerocap_final.json

### 2. Run the Hybrid Pipeline
This script loads the pre-trained Hybrid VLM model and runs the efficient feature-injection generation.

Note: Ensure your model checkpoint is available at ./models/Hybrid_RL_Best.pt.

```powershell
python HybridPipeline.py
```

Output: ./results/hybrid_final.json

### 3. Evaluate and Compare Models
This script loads both sets of generated captions and computes the standard evaluation metrics.
```powershell
python evaluation.py
```

Output: A markdown table comparing $\text{CIDEr}$, $\text{SPICE}$, $\text{CLIPScore}$, etc., for both pipelines.

## 📘 Methodology Breakdown
### ZeroCap Baseline (Inference Guidance)

     | LM                    | VSM               | Integration                           | Training          |
     |:---------------------:|:-----------------:|:-------------------------------------:|:-----------------:|
     | $\text{GPT-2 Medium}$ | $\text{ViT-B/32}$ | Logit Modification at every step.     | None (Zero-Shot)  |

**Limitation Addressed:** Semantic Fidelity.  <br>
**Trade-off:** Low Fluency, Slow Inference.
