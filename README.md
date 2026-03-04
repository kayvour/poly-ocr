# PolyOCR-Bench

PolyOCR-Bench is a multilingual OCR benchmarking framework for comparing different OCR engines across languages and scripts.

The current implementation compares:

- Tesseract OCR
- EasyOCR

The goal is to evaluate OCR systems using standard metrics instead of just extracting text.

---

## What This Does

Given a dataset of images and ground truth text, the framework:

- Runs multiple OCR engines
- Computes Character Error Rate (CER)
- Computes Word Error Rate (WER)
- Measures inference time
- Saves predicted text and ground truth
- Exports results as CSV and JSON
- Generates comparison plots

---

## Installation

Install dependencies:

pip install -r requirements.txt

Tesseract must be installed separately and available in your system PATH.

---

## Usage

Run a benchmark:

python cli.py --lang eng --dataset datasets/eng

Results are saved in:

results/reports/

Plots are saved in:

results/plots/

---

## Metrics

CER (Character Error Rate)  
WER (Word Error Rate)  
Inference time per engine  

Lower error means better accuracy.

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
- Multilingual benchmarking structure
- Evaluation pipeline for OCR systems
- Automated result reporting

---

## Future Improvements

- Engine selection through CLI
- Automatic language mapping
- Summary statistics per engine
- Visualization dashboard
- Docker support
- Integration of additional open source OCR models
- OCR to translation pipeline for multilingual document understanding

---

Built as a project exploring OCR evaluation and benchmarking pipelines.



