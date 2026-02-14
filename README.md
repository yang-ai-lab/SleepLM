# SleepLM — Sleep Language Model (Inference Demo)

[![Paper](https://img.shields.io/badge/paper-arXiv-blue)](#citation)
[![License](https://img.shields.io/badge/license-TBD-lightgrey)](#license)
[![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)](#installation)

SleepLM is a **sleep–language foundation model** that maps **multimodal polysomnography (PSG)** to natural language.  
This repository is an **inference-focused demo** for (1) **targeted caption generation** and (2) **signal↔text retrieval** using a shared embedding space.

> **Research-only.** SleepLM is **not** clinically validated and must not be used for diagnosis or medical decision-making.

---

## ✨ What you can do with this repo

- **Targeted caption generation** for 30-second sleep epochs using modality tokens (brain / cardiac / respiration / somatic).
- **Signal–text retrieval** by encoding signals and text into a shared embedding space and computing cosine similarity.
- Run an interactive demo in **`demo.ipynb`** (recommended).

---

## 🚀 Quickstart (5 minutes)

### 1) Install

```bash
git clone https://github.com/yang-ai-lab/sleep_language_DEMO.git
cd sleep_language_DEMO
pip install -r requirements.txt
```

**Requirements**
- Python ≥ 3.8
- PyTorch ≥ 2.0
- A CUDA-capable GPU is recommended for smooth inference (CPU may work but will be slower)

### 2) Download checkpoint

Download the model checkpoint here:  
- Google Drive: https://drive.google.com/drive/folders/1G-kECgRcXr9bJhsahnh6RWmzRGK7Rvme?usp=sharing

Place the checkpoint in the expected location used by `demo.ipynb` (see notebook cell that loads weights).

### 3) Run the demo

Open and run:

- `demo.ipynb`

The notebook includes:
- signal↔text similarity calculations
- targeted caption generation with per-modality conditioning

---

## 📦 Repository contents

- `demo.ipynb` — interactive inference + visualization
- `requirements.txt` — dependencies
- `examples/` — example preprocessed epochs (see “Using your own signals” below)

---

## 🧾 Input format

SleepLM expects a **30-second epoch**, sampled at **64 Hz** → **1920 samples/channel**, with **10 channels** in the order below.

### Channel order

| Index | Channel | Description |
|------:|---------|-------------|
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

### Body position encoding (POS channel)

```python
POSITION_ENCODING = {
    0: "Right",
    1: "Left",
    2: "Supine",
    3: "Prone",
    4: "Upright",
   -1: "Other/Unknown",  # Use for missing data
}
```

---

## 🧪 Using your own signals

You can generate captions for your own sleep recordings by placing **preprocessed** epochs in `examples/` and pointing the notebook to them.

**Signal requirements**
- Resample to **64 Hz**
- Normalize each channel (**z-score** recommended)
- If a channel is missing, **zero-pad** it (e.g., no airflow → channel 3 all zeros)
- POS must follow the integer encoding above
- Each epoch must be exactly **30 seconds** (**1920 samples @ 64 Hz**)

---

## 🔁 Reproducibility notes (inference demo)

This repo is intentionally lightweight and focuses on **inference**. If you plan to:
- reproduce paper benchmarks,
- train on NSRR cohorts,
- or evaluate cross-cohort generalization,

consider adding (or linking to) a separate training/eval codebase with:
- dataset loaders + subject-level splits
- configs matching the paper
- scripts to run each benchmark and produce tables/figures

(If you maintain that code elsewhere, link it prominently here.)

---

## ⚠️ Limitations & responsible use

- **Not a medical device**: do not use for clinical diagnosis, triage, or treatment decisions.
- Outputs can be **wrong, incomplete, or misleading**, especially under domain shift (sensor montage differences, noise/artifacts, missing channels).
- Use human oversight and validate on your own data distribution.

---

## 🛠️ Contributing

Contributions are welcome (bugs, docs, examples, usability improvements).

Suggested workflow:
1. Fork the repo and create a feature branch
2. Make changes + update README/examples if needed
3. Open a PR with a short description and screenshots/logs when relevant

If you plan to open-source broadly, consider adding:
- a code style tool (ruff/black) + minimal CI
- issue templates (bug report / feature request)
- a roadmap section (next features/checkpoints)

---

## 📝 Citation

If you use SleepLM in your research, please cite the paper:

```bibtex
@article{sleeplm2024,
  title   = {SleepLM: Natural Language Generation and Retrieval from Sleep Physiology Signals},
  author  = {TBD},
  journal = {arXiv preprint arXiv:TBD},
  year    = {2024}
}
```

---

## 📄 License

**TBD.**  
If you intend to be open-source, pick a license explicitly (e.g., **Apache-2.0** or **MIT**) and add a `LICENSE` file.

---

## 🙏 Acknowledgments

- Data sources and cohort infrastructure: **NSRR** (if applicable to your paper/training pipeline)
- Model architecture inspiration: OpenCLIP (https://github.com/mlfoundations/open_clip)
