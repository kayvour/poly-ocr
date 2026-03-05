# PolyOCR-Bench

PolyOCR-Bench is a multilingual OCR benchmarking framework for comparing different OCR engines, both traditional and transformer-based across languages, scripts, and document conditions.

The framework supports evaluating multiple models out-of-the-box and generates unified CSV/JSON reports alongside performance and accuracy comparison plots.

## Supported Engines

**Traditional OCR:**
- Tesseract OCR
- EasyOCR

**Deep Learning / Transformer OCR:**
- TrOCR (Microsoft)
- Donut (Naver CLOVA)
- DocTR (Mindee)
- Nougat (Meta)

---

## What This Does

Given a dataset of images and ground truth text, the framework:

- Runs multiple selected OCR engines dynamically (lazy-loading).
- Computes comprehensive **Text Accuracy** metrics (CER, WER, Exact Match, Levenshtein Distance, Token Accuracy).
- Computes **Semantic Similarity** metrics (BLEU, ROUGE).
- Evaluates **Performance** (Inference Time per image, Memory RSS usage).
- Evaluates **Robustness** under synthetic image corruptions (Gaussian noise, Blur, Rotation, Compression, Low resolution).
- Exports results as comprehensive CSV and JSON reports.
- Generates automatic comparison plots (e.g., Accuracy vs. Speed, Robustness degradation curves, Metric comparisons).

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

*Note: Tesseract must be installed separately and available in your system PATH.*

---

## Usage

Run a standard benchmark with default engines (all engines):

```bash
python cli.py --dataset datasets/eng --lang eng
```

Specify exactly which engines to evaluate:

```bash
python cli.py --dataset tmp_dataset --engines easyocr,trocr,doctr
```

Enable **Robustness Evaluation** (tests engines under corrupted conditions):

```bash
python cli.py --dataset tmp_dataset --engines easyocr --corrupt
```

### Outputs

Results are saved in:
`results/reports/` (Contains `benchmark.csv` and `benchmark.json`)

Visualization plots are saved in:
`results/plots/` (Contains metric comparions, Speed vs Accuracy charts, and Robustness degradation graphs).

---

## Datasets

This project is designed to work with multilingual OCR datasets. Some publicly available datasets you can use include:

**English**
- ICDAR 2013 / ICDAR 2015 Scene Text datasets  
- IIIT5K Word Dataset

**Chinese**
- RCTW-17 (Reading Chinese Text in the Wild)

**Arabic**
- ALIF Arabic Scene Text Dataset

**Hindi / Devanagari**
- IIIT-HW Devnagari Handwritten Dataset

**Multilingual**
- MLT (ICDAR Multi-Lingual Text Dataset)

These datasets contain labeled text images that can be used as input for benchmarking OCR engines.

Dataset folders should follow this structure:

datasets/<lang>/
    images/
    ground_truth/

Each image should have a corresponding `.txt` file with the same name containing the ground truth text.

---

## Project Focus

- Modular OCR engine interface
- Integration of state-of-the-art transformer models (TrOCR, Donut, Nougat, DocTR) alongside traditional systems
- Deep evaluation pipeline covering exact extraction, semantics, time, and memory constraints
- Automated result reporting and graphical visualization
