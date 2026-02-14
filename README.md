# SleepLM
[![Paper](https://img.shields.io/badge/paper-arXiv-blue)](#citation)
[![License](https://img.shields.io/badge/license-TBD-lightgrey)](#license)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen)](#installation)

SleepLM is, to our knowledge, the first sleep-language foundation model family that enables targeted natural language generation from multimodal polysomnography (PSG) while also learning a shared signal–text embedding space for retrieval and open vocabulary sleep understanding. It is trained on the largest paired sleep–text corpus to date, built from five NSRR cohorts totaling 100K+ hours of PSG from 10,000+ individuals.

SleepLM supports controllable, domain-specific generation (brain, cardiac, respiration, somatic) as well as holistic summaries, moving beyond fixed label spaces like sleep stages and events. The model combines contrastive alignment, captioning, and signal reconstruction to preserve physiological fidelity while learning strong cross-modal semantics. Across a broad benchmark, SleepLM enables sleep-text retrieval, zero-shot and few-shot generalization, and robust transfer to unseen concepts.

---

## ✨ What you can do with this repo

- **Targeted caption generation** for 30-second sleep epochs using modality tokens (brain / cardiac / respiration / somatic).
- **Signal–text retrieval** by encoding signals and text into a shared embedding space and computing cosine similarity.
- Run an interactive demo in **`demo.ipynb`**.

---

## 🚀 Quickstart

### 1) Install

```bash
git clone https://github.com/yang-ai-lab/sleep_language_DEMO.git
cd sleep_language_DEMO
pip install -r requirements.txt
```

### 2) Download checkpoint

Download the model checkpoint here:  
- Google Drive: https://drive.google.com/drive/folders/1G-kECgRcXr9bJhsahnh6RWmzRGK7Rvme?usp=sharing

Place the checkpoint in the expected location used by `demo.ipynb` (see notebook cell that loads weights).

### 3) Run the demo

Open and run:

- `demo.ipynb`

The notebook includes:
- signal to text similarity calculations
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
- Normalize each channel (**z-score**)
- If a channel is missing, **zero-pad** it
- POS must follow the integer encoding above
- Each epoch must be exactly **30 seconds** (**1920 samples @ 64 Hz**)

---

## 🔁 Reproducibility notes

This repo is intentionally lightweight and focuses on **inference**. If you plan to:
- reproduce paper benchmarks,
- train on NSRR cohorts,
- or evaluate cross-cohort generalization,

We are planning to opensource it upon the acceptance of the paper

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
