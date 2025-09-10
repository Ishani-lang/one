# Toda Language POS Tagger

A cross-lingual adapter-based transfer learning model for Part-of-Speech (POS) tagging of Toda language using Tamil language resources and few-shot learning.

## Overview

This project implements a POS tagger for the Toda language using:
- Cross-lingual adapter-based transfer learning
- Tamil language datasets from Hugging Face
- Few-shot learning for low-resource language adaptation
- MAD-X framework for modular adapters

## Features

- Pre-trained multilingual models (XLM-RoBERTa, mBERT)
- Language-specific adapters for Tamil
- Task-specific adapters for POS tagging
- Few-shot learning for Toda language adaptation
- Comprehensive evaluation and inference scripts

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. **Train on Tamil data:**
   ```bash
   python train_tamil_adapter.py
   ```

2. **Fine-tune for Toda with few-shot learning:**
   ```bash
   python train_toda_fewshot.py
   ```

3. **Evaluate the model:**
   ```bash
   python evaluate.py
   ```

4. **Run inference:**
   ```bash
   python inference.py
   ```

## Project Structure

```
├── requirements.txt
├── README.md
├── config/
│   ├── __init__.py
│   ├── model_config.py
│   └── training_config.py
├── data/
│   ├── __init__.py
│   ├── tamil_processor.py
│   ├── toda_processor.py
│   └── utils.py
├── models/
│   ├── __init__.py
│   ├── adapter_model.py
│   └── pos_tagger.py
├── training/
│   ├── __init__.py
│   ├── trainer.py
│   └── few_shot_trainer.py
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   └── evaluator.py
├── scripts/
│   ├── train_tamil_adapter.py
│   ├── train_toda_fewshot.py
│   ├── evaluate.py
│   └── inference.py
└── notebooks/
    ├── data_exploration.ipynb
    └── model_analysis.ipynb
```

## Datasets

- Tamil POS tagging data from Universal Dependencies
- Custom Toda language few-shot datasets
- Cross-lingual evaluation datasets

## Models

- Base: XLM-RoBERTa (xlm-roberta-base)
- Language Adapter: Tamil-specific adapter
- Task Adapter: POS tagging adapter
- Transfer: Cross-lingual transfer to Toda