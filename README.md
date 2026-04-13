# FW-SNN

Official implementation of **"SNN-Focused Frequency Pruning: When Less is More in Audio Classification"**, submitted to *Information Sciences*.

## Overview

We propose **Frequency-Weighted Input Pruning (FW-SNN)**, a lightweight framework that learns each frequency bin's task-specific importance via end-to-end gradient descent and removes low-importance bands through structured input-axis pruning. Removing noise-dominated bands at the input interface suppresses spurious spike generation in LIF neurons, allowing input compression to also behave as data-driven denoising.

<p align="center">
<img src="./infer_data/overview.png" align="center" width="100%" style="margin: 0 auto">
</p>

## Requirements

```
torch>=2.0.1
torchvision>=0.15.2
torchaudio>=2.0.2
librosa>=0.10.1
spikingjelly==0.0.0.0.14
numpy>=1.23.5
pandas>=1.5.3
scikit-learn>=1.2.1
opencv-python>=4.8.1.78
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

### Phase 1: Train with Learnable Frequency Weights

```bash
# UrbanSound8K (VGG9-SNN)
python fwsnn.py -train -arch=vgg9 -act=snn -device=cuda -data_dir=. -dataset=urbansound -b=128 -use_freq_weights

# GTZAN (VGG9-SNN)
python fwsnn.py -train -arch=vgg9 -act=snn -device=cuda -data_dir=. -dataset=gtzan -b=16 -use_freq_weights
```

### Phase 2: Structured Frequency Pruning

```bash
# Prune 25% of frequency bins
python fwsnn.py -prune -arch=vgg9 -act=snn -device=cuda -data_dir=. -dataset=urbansound -b=128 -prune_ratio=0.25

# Prune 50% of frequency bins
python fwsnn.py -prune -arch=vgg9 -act=snn -device=cuda -data_dir=. -dataset=urbansound -b=128 -prune_ratio=0.50
```

### Phase 3: Fine-tuning on Pruned Input

```bash
python fwsnn.py -train -arch=vgg9 -act=snn -device=cuda -data_dir=. -dataset=urbansound -b=128 -pruned
```

### Energy Estimation

```bash
python energy.py -arch=vgg9 -act=snn -device=cuda -data_dir=. -dataset=urbansound -b=128
```

### Analysis & Visualization

```bash
python analysis.py
```

## Datasets

Download and place datasets in the data directory:

| Dataset | Link |
|---------|------|
| UrbanSound8K | [Kaggle](https://www.kaggle.com/datasets/chrisfilo/urbansound8k) |
| GTZAN | [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) |
| SHD | [Zenke Lab](https://zenkelab.org/datasets/SHD/) |

## Note

This repository currently contains the **VGG9-SNN** implementation used for UrbanSound8K and GTZAN experiments. The **Spikformer**-based implementation for SHD (Spiking Heidelberg Digits) uses a different architecture and will be released in a future update.

## Project Structure

```
FW-SNN/
├── fwsnn.py          # Main training and pruning pipeline
├── model.py          # VGG9-SNN model with learnable frequency weights
├── utils.py          # Data loading, preprocessing, and utilities
├── energy.py         # Energy consumption estimation
├── analysis.py       # Result analysis and visualization
├── photo.py          # Figure generation
├── generate_concept_spectrogram.py  # Concept spectrogram visualization
├── requirements.txt
├── LICENSE
└── README.md
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{guo2026fwsnn,
  title={SNN-Focused Frequency Pruning: When Less is More in Audio Classification},
  author={Guo, Xing and Su, Zhiming and Hu, Jingyang and Zhou, Wangqiu and Li, Wei and Fu, Yaoheng},
  journal={Information Sciences},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
