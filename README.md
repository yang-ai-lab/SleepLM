# SleepLM

This codebase contains an inference pipeline for generating natural language descriptions of sleep physiology signals using the **SleepLM**. This model can generate detailed captions conditioned on specific physiological modalities (brain, cardiac, respiration, somatic) and compute signal-text similarity in a shared embedding space.

---

## 🌟 Overview

This repository provides an interface for SleepLM, capable of:

1. **Targeted Caption Generation**: Generate free-text descriptions of 30-second sleep epochs by conditioning on specific modality tokens.
2. **Signal–Text Retrieval**: Encode both biosignals and text into a shared embedding space and compute cosine similarity for retrieval tasks.
3. **Checkpoint**: Please download the checkpoint from [https://drive.google.com/drive/folders/1G-kECgRcXr9bJhsahnh6RWmzRGK7Rvme?usp=sharing](https://drive.google.com/drive/folders/1G-kECgRcXr9bJhsahnh6RWmzRGK7Rvme?usp=sharing).

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yang-ai-lab/sleep_language_DEMO.git
cd sleep_language_DEMO

# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python ≥ 3.8
- PyTorch ≥ 2.0
- CUDA-capable GPU

### Basic Usage

Download the model checkpoint from [https://drive.google.com/drive/folders/1G-kECgRcXr9bJhsahnh6RWmzRGK7Rvme?usp=sharing](https://drive.google.com/drive/folders/1G-kECgRcXr9bJhsahnh6RWmzRGK7Rvme?usp=sharing).

Launch `demo.ipynb` for an interactive, visual demonstration.

The notebook includes:
- Signal-text similarity calculations
- Targeted caption generation with per-modality conditioning

---

## 📋 Data Preparation Guide

### Channel Order and Encoding

The model expects signals in the following order:

| Index | Channel | Description |
|-------|---------|-------------|
| 0 | ECG | Electrocardiogram |
| 1 | ABD | Abdominal respiratory effort |
| 2 | THX | Thoracic respiratory effort |
| 3 | AF | Airflow |
| 4 | EOG_Left | Left eye movement |
| 5 | EOG_Right | Right eye movement |
| 6 | EEG_C3_A2 | Left central EEG |
| 7 | EEG_C4_A1 | Right central EEG |
| 8 | EMG_Chin | Chin muscle tone |
| 9 | POS | Body position |

### Body Position Encoding

The **POS** channel uses integer encoding:

```python
POSITION_ENCODING = {
    0: "Right",          
    1: "Left",           
    2: "Supine",         
    3: "Prone",          
    4: "Upright",        
   -1: "Other/Unknown"   # Use for missing data
}
```
Convert your position labels to these integer values before saving signals.

---

## 📋 Using Your Own Signals

While we provide a few examples under the `\examples`, You can generate captions for your own sleep recordings by placing pre-processed signals in the `\examples` directory:

### Signal Requirements
```
  - Resample to 64 Hz sampling rate
  - Normalize each channel (z-score normalization recommended)
  - Zero-pad missing channels to 0 (e.g., if your recording lacks airflow, set channel 3 to all zeros)
  - Body Position follows standard encoding:
    - `0` = Right
    - `1` = Left
    - `2` = Supine
    - `3` = Prone
    - `4` = Upright
    - `-1` = Other/Unknown
  - Ensure each epoch is exactly 30 seconds (1920 samples @ 64 Hz)
```
---

## 📝 Citation

If you find this model useful for your research, please consider citing:

```bibtex
@article{your2024sleep,
  title={Sleep Language Model: Natural Language Generation from Biosignals},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## License

[Specify your license here, e.g., MIT, Apache 2.0, etc.]

---

## Acknowledgments

This work was made possible by [your institution/funding sources].

The model architecture is based on [OpenCLIP](https://github.com/mlfoundations/open_clip) and adapted for multi-channel biosignal processing.
