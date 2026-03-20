# DCRNet – Accelerated QSM and R2* via Deep Complex Residual Network

**Accelerating Quantitative Susceptibility and R2\* Mapping using Incoherent Undersampling and Deep Neural Network Reconstruction**

[NeuroImage 2021](https://doi.org/10.1016/j.neuroimage.2021.118404) &nbsp;|&nbsp; [arXiv](https://arxiv.org/abs/2103.09375) &nbsp;|&nbsp; [data & checkpoints](https://www.dropbox.com/sh/p9k9rq8zux2bkzq/AADSgw3bECQ9o1dPpIoE5b85a?dl=0) &nbsp;|&nbsp; [deepMRI collection](https://github.com/sunhongfu/deepMRI)

DCRNet recovers both MR magnitude and quantitative phase images from compressed-sensing undersampled k-space data, accelerating QSM and R2* acquisitions using a deep complex residual network.

> **Note:** QSM post-processing (from phase images) requires Linux. Magnitude/phase reconstruction from undersampled data works on all platforms.

---

## Overview

### Framework

![Whole Framework](https://www.dropbox.com/s/f729s5l2xvpwjfx/Figs_1.png?raw=1)

Fig. 1: Overview of the proposed QSM acceleration scheme.

### Network Architecture

![Data Flow](https://www.dropbox.com/s/2519jlm4cr8g9cp/Figs_2.png?raw=1)

Fig. 2: DCRNet architecture built on a deep residual network using complex convolutional operations.

---

## Requirements

**For DL-based reconstruction:**
- Python 3.7+, PyTorch 1.8+
- NVIDIA GPU (CUDA 10.0+)
- MATLAB R2017b+

**For QSM post-processing (Linux only):**
- [Hongfu Sun's QSM toolbox](https://github.com/sunhongfu/QSM)
- [FSL v6.0](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL)

Tested on: CentOS 7.8 (Tesla V100), Windows 10 / Ubuntu 19.10 (GTX 1060).

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/sunhongfu/DCRNet.git
cd DCRNet

conda create -n DCRNet python=3.8
conda activate DCRNet
conda install pytorch cudatoolkit=10.2 -c pytorch
conda install scipy
pip install mat73
```

### 2. Download demo data and checkpoints

Download from [Dropbox](https://www.dropbox.com/sh/p9k9rq8zux2bkzq/AADSgw3bECQ9o1dPpIoE5b85a?dl=0) and place in the repo root.

### 3. Run demo

```bash
conda activate DCRNet

# Single-channel
matlab -nodisplay -r demo_single_channel

# Multi-channel
matlab -nodisplay -r demo_multi_channel
```

---

## Reconstruction on Your Own Data

Edit parameters in lines 10–20 of the demo script, then run:

```bash
# Single-channel
conda activate DCRNet
matlab -nodisplay -r demo_single_channel

# Multi-channel
conda activate DCRNet
matlab -nodisplay -r demo_multi_channel
```

---

## Training

```matlab
% 1. Prepare training data
matlab -nodisplay -r PrepareTrainingData
```

```bash
# 2. Train DCRNet
cd PythonCodes/training
python TrainDCRNet.py
```

---

## Citation

```bibtex
@article{dcrnet2021,
  title={Accelerating quantitative susceptibility and R2* mapping using incoherent undersampling and deep neural network reconstruction},
  journal={NeuroImage},
  year={2021},
  doi={10.1016/j.neuroimage.2021.118404}
}
```

---

[⬆ top](#dcrnet--accelerated-qsm-and-r2-via-deep-complex-residual-network) &nbsp;|&nbsp; [deepMRI collection](https://github.com/sunhongfu/deepMRI)
