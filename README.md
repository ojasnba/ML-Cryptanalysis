# ML Cryptanalysis of ASCON320

Implementation and evaluation of machine learning-based differential distinguishers for the **ASCON320** authenticated encryption algorithm.

This repository extends the baseline neural-network implementation by incorporating additional classical machine learning models, performing hyperparameter optimization, and comparing their effectiveness for differential cryptanalysis.

---

## Project Overview

The objective of this project is to investigate whether machine learning models can distinguish **real ciphertext pairs** from **random ciphertext pairs** generated using differential cryptanalysis on ASCON320.

The repository currently contains implementations of:

- Multi-Layer Perceptron (MLP) baseline
- Random Forest distinguisher
- XGBoost distinguisher
- Hyperparameter optimization for classical ML models
- Validation and evaluation pipeline
- ASCON320 dataset generation

Work is currently in progress on reproducing and improving CNN-based distinguishers reported in the literature.

---

# Repository Structure

```
ML-Cryptanalysis
│
├── ascon.py                          # ASCON320 implementation and data generation
├── gift.py                           # GIFT128 implementation
│
├── train_distinguisher.py            # Original MLP training
├── improve_distinguisher.py          # Original MLP distinguisher
│
├── train_random_forest.py
├── improve_distinguisher_rf.py
├── improve_distinguisher_rf_optimised.py
│
├── train_xgboost.py
├── improve_distinguisher_xgb.py
│
├── train_1dcnn_ascon.py              # CNN implementation (work in progress)
│
├── rf_validation.py                  # Random Forest validation pipeline
│
├── docs/
│   ├── XGBOOST_EXPERIMENTS.md
│   ├── RANDOM_FOREST_EXPERIMENTS.md
│   └── MODEL_COMPARISON.md
│
├── ExecutionParameters.txt
├── requirements_used.txt
├── README.md
└── .gitignore
```

---

# Implemented Models

| Model | Status |
|--------|--------|
| Multi-Layer Perceptron (MLP) | Complete |
| Random Forest | Complete |
| XGBoost | Complete |
| 1D CNN | In Progress |
| 2D CNN | Planned |

---

# Experimental Configuration

Unless stated otherwise, experiments were performed using the following configuration:

| Parameter | Value |
|-----------|-------|
| Cipher | ASCON320 |
| Output Bits | 40 |
| Rounds | 4 |
| Training Samples | 2¹⁶ |
| Validation Samples | 2¹⁴ |
| Prediction Samples | 2¹⁹ |
| Experiments | 100 |

Input Difference:

```
0000000001
```

---

# Implemented Improvements

Compared to the baseline implementation, the repository includes:

- Random Forest based differential distinguisher
- XGBoost based differential distinguisher
- Hyperparameter tuning
- Comparative evaluation of multiple ML models
- Improved validation pipeline
- Performance benchmarking

---

# Results

The best-performing XGBoost configuration obtained during experimentation was:

| Hyperparameter | Value |
|---------------|------:|
| Max Depth | 6 |
| Learning Rate | 0.10 |
| Number of Estimators | 100 |

Detailed tuning experiments and observations are available in the `docs/` directory.

---

# Installation

Clone the repository

```bash
git clone <repository-url>
cd ML-Cryptanalysis
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

**Windows**

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements_used.txt
```

---

# Running Experiments

Train Random Forest

```bash
python train_random_forest.py
```

Evaluate Random Forest

```bash
python improve_distinguisher_rf_optimised.py
```

Train XGBoost

```bash
python train_xgboost.py
```

Evaluate XGBoost

```bash
python improve_distinguisher_xgb.py
```

---

# Documentation

Additional experiment logs are available in:

```
docs/
```

including:

- Random Forest hyperparameter tuning
- XGBoost hyperparameter tuning
- Model comparison

---

# Current Work

Current research focuses on reproducing and improving CNN-based differential distinguishers for the 4-round ASCON320 setting and comparing their performance against the implemented classical machine learning models.

---

# Future Work

- Reproduce published 1D CNN results
- Reproduce published 2D CNN results
- Improve CNN validation accuracy for 4-round ASCON320
- Compare classical ML and deep learning approaches
- Explore additional machine learning architectures

---

# Acknowledgements

This project builds upon prior research on neural differential distinguishers for ASCON and GIFT block ciphers. The repository extends the baseline implementation with additional machine learning models and systematic hyperparameter optimization.