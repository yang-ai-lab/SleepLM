# SleepLM: Natural-Language Intelligence for Human Sleep
[![Paper](https://img.shields.io/badge/paper-arXiv-red)](https://arxiv.org/abs/2602.23605)
[![Webpage](https://img.shields.io/badge/website-demo-blue)](https://yang-ai-lab.github.io/SleepLM/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-SleepLM--Base-FFD21E)](https://huggingface.co/yang-ai-lab/SleepLM-Base)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)


SleepLM is the first sleep-language foundation model family that enables targeted natural language generation from multimodal polysomnography (PSG) while also learning a shared signal–text embedding space for retrieval and open vocabulary sleep understanding. It is trained on the largest paired sleep–text corpus to date, built from five NSRR cohorts totaling 100K+ hours of PSG from 10,000+ individuals.

SleepLM supports controllable, domain-specific generation (brain, cardiac, respiration, somatic) as well as holistic summaries, moving beyond fixed label spaces like sleep stages and events. The model combines contrastive alignment, captioning, and signal reconstruction to preserve physiological fidelity while learning strong cross-modal semantics. Across a broad benchmark, SleepLM enables sleep-text retrieval, zero-shot and few-shot generalization, and robust transfer to unseen concepts.

---

## 📰 News
- **[2026-03-02]** Paper released on [arXiv](https://arxiv.org/abs/2602.23605)!
- **[2026-02-23]** Code released on GitHub, and model released on [HuggingFace](https://huggingface.co/yang-ai-lab/SleepLM-Base)!
- **[2026-02-23]** [Project website](https://yang-ai-lab.github.io/SleepLM/) is live!

---

## ✨ What you can do with this repo

- **Targeted caption generation** for 30-second sleep epochs using modality tokens (brain / cardiac / respiration / somatic).
- **Cross modal retrieval** by encoding signals and text into a shared embedding space and computing cosine similarity.
- Run an interactive demo in **`demo.ipynb`**.

---

## 🚀 Quickstart

### 1) Install

```bash
git clone https://github.com/yang-ai-lab/SleepLM
cd SleepLM
pip install -r requirements.txt
```

### 2) Download checkpoint

The model checkpoint is hosted on Hugging Face Hub:

```python
from huggingface_hub import hf_hub_download
checkpoint_path = hf_hub_download(repo_id="yang-ai-lab/SleepLM-Base", filename="model_checkpoint.pt")
```

Or via the CLI:
```bash
huggingface-cli download yang-ai-lab/SleepLM-Base model_checkpoint.pt
```

### 3) Prepare your data

Preprocess your PSG recordings into a float32 PyTorch tensor of shape `[N, 10, 1920]`
(N epochs × 10 channels × 1920 samples) following the channel order and signal requirements
in [Using your own signals](#-using-your-own-signals) below.
Save it as a `.pt` file and update the path in `demo.ipynb`.

### 4) Run the demo

Open and run:

- `demo.ipynb`

The notebook includes:
- similarity calculation between signal and text embeddings
- targeted caption generation with per-modality conditioning

---

## 📦 Repository contents

- `demo.ipynb` — interactive inference + visualization
- `requirements.txt` — dependencies

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

You can generate captions for your own sleep recordings by loading **preprocessed** epochs directly in `demo.ipynb`.

**Signal requirements**
- Resample to **64 Hz**
- Normalize each channel (**z-score**)
- If a channel is missing, **zero-pad** it
- POS must follow the integer encoding above
- Each epoch must be exactly **30 seconds** (**1920 samples @ 64 Hz**)
- Pack epochs into a float32 PyTorch tensor of shape `[N, 10, 1920]`

---

## 🔁 Reproducibility notes

This repo is intentionally lightweight and focuses on **inference**. If you plan to:
- reproduce paper benchmarks,
- train on NSRR cohorts,
- or evaluate cross-cohort generalization,

We are planning to opensource our training pipeline upon the acceptance of the paper. 
Note that the training data will not be opensourced due credential issue. If you wish to use the same NSRR datasets, [please apply here](https://sleepdata.org/).

---

## 📝 Citation

If you use SleepLM in your research, please cite the paper:

```bibtex
@article{xu2026sleeplm,
  title={SleepLM: Natural-Language Intelligence for Human Sleep},
  author={Xu, Zongzhe and Shuai, Zitao and Mozaffari, Eideen and Aysola, Ravi Shankar and Kumar, Rajesh and Yang, Yuzhe},
  journal={arXiv preprint arXiv:2602.23605},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Data sources and cohort infrastructure: **NSRR** (if applicable to your paper/training pipeline)
- Model architecture inspiration: OpenCLIP (https://github.com/mlfoundations/open_clip)
