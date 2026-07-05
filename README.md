# MoCaf-Mamba: Modality Completion and Alignment in Feature Space

[![MICCAI](https://img.shields.io/badge/Conference-MICCAI_2026-blue)](https://anonymous.4open.science/r/MoCaf-Mamba-467C/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)

This repository contains the official PyTorch implementation for the **MICCAI 2026** paper:

> **"MoCaf-Mamba (Modality Completion and Alignment in Feature Space): A Mamba-based Framework for Missing-Modality Segmentation"**

---

## 🚀 Overview

Multimodal medical image segmentation often suffers from **missing modalities** and **inter-modality misalignment**. To address these challenges, we propose **MoCaf-Mamba**, a unified multi-scale framework that integrates three key components:

- **Feature-space Modality Completion (MCFS)** — predicts missing modality representations using bidirectional Mamba blocks with reconstruction supervision, avoiding expensive pixel-level synthesis.
- **Deformable Feature Alignment (DFA)** — hierarchically aligns multi-scale features via bounded dense displacement fields to mitigate residual geometric discrepancies.
- **Space Token Mixer (STM)** — aggregates completed and aligned features through a dual-branch design with a global consensus token for robust decoding.

### Pipeline

```
Input (arbitrary missing modalities)
        │
        ▼
  ┌─────────────────────┐
  │  Mamba Encoder      │  ← per-modality multi-scale encoding
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  MCFS (Completion)  │  ← bidirectional Mamba, reconstruction loss
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  DFA (Alignment)    │  ← deformable feature alignment
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  STM (Fusion)       │  ← dual-branch token mixer
  └─────────┬───────────┘
            │
            ▼
        Segmentation Output
```

Our method achieves **state-of-the-art** performance on **BraTS2023**, **Prostate158**, and **MM-WHS** benchmarks under various missing patterns.

---

## ✨ Main Features

- ✅ **Unified pipeline** for arbitrary missing-modality inputs (any subset of available modalities).
- ✅ **Efficient feature-space completion** without heavy GANs or Diffusion models.
- ✅ **Deformable alignment** at the feature level for non-rigid distortion correction.
- ✅ **Lightweight cross-modal fusion (STM)** with linear complexity.
- ✅ **Pre-trained models** and training scripts provided.

---

## 📁 Project Structure

```
MoCaf-Mamba/
├── README.md
├── LICENSE
├── requirements.txt
├── CITATION.cff
├── train_mamba.py                  # Main training script (DDP)
├── test_all.py                     # Evaluation script
├── data_set.py                     # Dataset loader with augmentation
├── scripts/
│   ├── train.sh                    # Training launch example
│   └── test.sh                     # Testing launch example
└── model/
    ├── __init__.py
    ├── mocaf_mamba.py              # MoCaf-Mamba model
    ├── layers.py                   # Shared building blocks
    └── utils/
        ├── __init__.py
        ├── criterions.py           # Loss functions
        ├── generate.py             # Visualization utilities
        ├── initialization.py       # Weight initialization
        ├── lr_scheduler.py         # LR schedulers
        ├── parser.py               # Configuration parser
        ├── random_seed.py          # Seed setting
        └── str2bool.py             # String-to-bool helper
```

---

## 🛠️ Environment and Dependencies

### Hardware Requirements
- **GPU**: NVIDIA V100 (16GB+) or equivalent (24GB recommended for 3D volumes).
- **RAM**: 32GB+ recommended.

### Installation

```bash
# 1. Create conda environment
conda create -n mocaf python=3.8
conda activate mocaf

# 2. Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Install other dependencies
pip install -r requirements.txt
```

> **Note**: `mamba-ssm` may require specific CUDA toolkit setup. Refer to [mamba-ssm official docs](https://github.com/state-spaces/mamba) for installation guidance.

---

## 📊 Data Preparation

Organize your dataset in the following structure:

```
data/
├── Prostate_image/      # modality images (.nii.gz)
│   ├── case001.nii.gz
│   ├── case002.nii.gz
│   └── ...
└── Prostate_label/      # corresponding labels (.nii.gz)
    ├── case001.nii.gz
    ├── case002.nii.gz
    └── ...
```

The dataset loader (`data_set.py`) expects paired `_image` and `_label` directories. By default, 3 modalities and 3 segmentation classes are used. You can modify `class_index` in `data_set.py` for custom datasets.

---

## 🏋️ Training

### Single-node Multi-GPU (DDP)

```bash
torchrun --nproc_per_node=4 train_mamba.py \
    --data_dir ./data \
    --output_dir ./checkpoints \
    --fold 0 \
    --epochs 300 \
    --lr 1e-4
```

### Key Training Details
- **Distributed Training**: DDP with NCCL backend, mixed precision (AMP).
- **Optimizer**: RAdam with weight decay 3e-5, CosineAnnealingLR with 5-epoch linear warmup.
- **EMA**: Exponential moving average (decay=0.999) for stable validation.
- **Missing Modality Simulation**: Random modality dropout during training (50% probability).
- **Validation**: 7 missing patterns evaluated (full + 6 partial combinations).

---

## 📈 Evaluation

```bash
python test_all.py \
    --checkpoint ./checkpoints/0/last.pth \
    --data_dir ./data
```

The evaluation reports:
- **Dice Score** and **IoU** for each missing modality pattern.
- **Class-wise Dice** for fine-grained analysis.
- **Comparison table** across different models.

---

## 🧩 Missing Modality Patterns

During evaluation, we test 7 modality availability patterns:

| Pattern | Mod1 | Mod2 | Mod3 | Description   |
|---------|------|------|------|---------------|
| 111     | ✓    | ✓    | ✓    | Full modality |
| 110     | ✓    | ✓    |      | Miss Mod3     |
| 101     | ✓    |      | ✓    | Miss Mod2     |
| 011     |      | ✓    | ✓    | Miss Mod1     |
| 100     | ✓    |      |      | Only Mod1     |
| 010     |      | ✓    |      | Only Mod2     |
| 001     |      |      | ✓    | Only Mod3     |

---

## 📦 Pre-trained Models

Pre-trained model weights are available at:

| Model        | Full Dice | Missing Dice | Download |
|--------------|-----------|--------------|----------|
| MoCaf-Mamba  | TBD       | TBD          | [Link]() |

---

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{mocaf2026,
  title     = {MoCaf-Mamba: Modality Completion and Alignment in Feature Space},
  author    = {Anonymized Authors},
  booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year      = {2026}
}
```

---

## 🙏 Acknowledgements

This project builds upon several open-source works, including:
- [Mamba](https://github.com/state-spaces/mamba) (Selective State Space Models)
- [MONAI](https://monai.io/) (Medical Open Network for AI)
- [TorchIO](https://torchio.readthedocs.io/) (Medical image augmentation)

---

## 📧 Contact

For questions and feedback, please open an issue on GitHub or contact the authors.
