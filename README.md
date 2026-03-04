# PolyOCR-Bench

PolyOCR-Bench is a modular multilingual OCR benchmarking framework built to compare OCR engines across different languages and scripts.

Instead of just extracting text, this project focuses on evaluation — measuring accuracy and performance across engines like Tesseract and EasyOCR using standard metrics.

---

## What This Does

Given a dataset of images and ground-truth text, the framework:

- Runs multiple OCR engines
- Computes Character Error Rate (CER)
- Computes Word Error Rate (WER)
- Measures inference time
- Exports results as CSV and JSON reports

The goal is not just OCR — but structured benchmarking.

---

## Why This Exists

Most OCR demos stop at:
"Here is the extracted text."

This project asks:
- How accurate is the engine?
- How does it perform across different languages?
- Which engine is faster?
- How does preprocessing affect results?

It is designed as a systems + evaluation oriented project.

---

## Installation

Install dependencies:

pip install -r requirements.txt

Note:
Tesseract must be installed separately and available in your system PATH.

---

## Usage

Run a benchmark:

python cli.py --lang eng --dataset datasets/eng

Results are saved in:

results/reports/

---

## Metrics

CER (Character Error Rate)  
WER (Word Error Rate)  
Inference time per engine  

Lower error = better accuracy.

---

## Project Focus

- Modular engine abstraction
- Multilingual-ready structure
- Extensible benchmarking pipeline
- Evaluation-first design

This project is intentionally backend-heavy and evaluation-focused rather than UI-driven.

---

## Future Improvements

- Engine selection via CLI
- Automatic language mapping
- Summary statistics per engine
- Visualization dashboard
- Docker support
- Additional OCR engine integrations

---

Built as a systems-oriented project exploring multilingual OCR benchmarking and evaluation pipelines.
