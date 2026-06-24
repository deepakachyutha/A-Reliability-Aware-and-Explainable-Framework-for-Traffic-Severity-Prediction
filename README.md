# STRADA: System for Traffic Risk Assessment & Dynamic Analysis

> A reliability-aware and explainable framework for traffic accident severity prediction using machine learning and contextual risk interpretation.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Research](https://img.shields.io/badge/Research-Project-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

STRADA (**System for Traffic Risk Assessment & Dynamic Analysis**) is a research-oriented machine learning framework designed to study accident severity prediction in safety-critical environments.

Unlike conventional prediction systems that focus solely on maximizing accuracy, STRADA emphasizes **reliability, interpretability, and robustness**. The framework combines statistical machine learning with reliability-aware risk interpretation and explainable recommendations to provide meaningful safety insights.

The project supports the research paper:

**"A Reliability-Aware and Explainable Framework for Vehicular Accident Severity Prediction"** :contentReference[oaicite:0]{index=0}

---

## Motivation

Traffic accident datasets are highly imbalanced. Most accidents belong to moderate severity classes, causing machine learning models to become overconfident and often fail to recognize rare but critical events.

STRADA aims to address these challenges by:

- Building a reliable accident severity prediction framework.
- Studying uncertainty in safety-critical machine learning.
- Introducing reliability-aware confidence interpretation.
- Providing explainable and human-readable safety recommendations.
- Creating a modular research platform for future experimentation.

---

# Features

### Accident Severity Prediction

Predicts accident severity from environmental and temporal conditions.

### Reliability-Aware Framework

Transforms model confidence into an interpretable reliability score instead of relying solely on raw probabilities.

### Explainable AI

Provides feature importance analysis and SHAP-based explanations.

### Temporal Risk Modeling

Captures accident patterns across hours, weekdays, and seasons.

### Regional Risk Modeling

Uses clustering techniques to incorporate geographical characteristics.

### Retrieval-Based Safety Recommendations

Generates contextual safety guidance using knowledge retrieval methods.

### Interactive Dashboard

Includes a Streamlit interface for exploration and experimentation.

---

# Dataset

This project utilizes the **US Accidents Dataset**, a large-scale public repository containing traffic accidents collected across the United States.

### Dataset Statistics

| Property | Value |
|------------|--------|
| Total records | ~7.7 Million |
| Training subset | ~500,000 samples |
| Time period | 2021–2023 |
| States covered | 50 |
| Severity classes | 4 |

The dataset itself is excluded from this repository due to its size.

Dataset source:

- US Accidents Dataset (Moosavi et al.)
- https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

---

# Features Used

The model uses structured environmental and temporal variables:

- Weather conditions
- Temperature
- Visibility
- Hour of day
- Day of week
- Month
- Regional clusters

These variables capture both environmental and behavioral patterns associated with traffic accidents.

---

# Models Evaluated

Multiple models were benchmarked:

- Random Forest
- Support Vector Machine (SVM)
- AdaBoost
- Multi-Layer Perceptron (MLP)
- XGBoost + SMOTE

Random Forest was selected as the primary model due to its robustness, stability, and interpretability.

---

# Project Architecture

```
Input Data
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ▼
Random Forest Model
     │
     ▼
Reliability Index
     │
     ▼
Explainability Module
     │
     ▼
Safety Recommendation System
```

---

# Technologies Used

### Language

- Python

### Machine Learning

- Scikit-Learn
- XGBoost

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Explainability

- SHAP

### Database

- PostgreSQL

### User Interface

- Streamlit

---

# Repository Structure

```
STRADA
│
├── data/                  # Dataset (excluded)
├── notebook/              # Exploratory analysis and experiments
├── paper/                 # Research paper and documents
├── research_files/        # Figures and generated outputs
├── Plots and Images/      # Visualizations
├── src/
│   ├── run_xgboost.py
│   ├── monte_carlo.py
│   └── training scripts
│
├── models/                # Saved models
├── app.py                 # Streamlit application
├── requirements.txt
└── README.md
```

---

# Experimental Results

The framework was evaluated on approximately **500,000 post-pandemic accident records**.

### Performance

| Model | Accuracy | Weighted F1 |
|---------|---------|---------|
| Random Forest | 88.03% | 84.68% |
| SVM | 90.21% | 85.57% |
| AdaBoost | 90.21% | 85.57% |
| MLP | 90.00% | 85.00% |
| XGBoost + SMOTE | 85.00% | 83.00% |

Although some models achieved slightly higher accuracy, STRADA prioritizes reliability and interpretability over raw performance.

---

# Future Work

Planned extensions include:

- Real-time traffic prediction
- OBD-II vehicle telemetry integration
- Edge deployment using Raspberry Pi
- Probability calibration techniques
- Advanced uncertainty estimation
- Deep learning approaches
- Live explainable intelligent transportation systems

---

# Research Focus

- Machine Learning
- Explainable AI (XAI)
- Intelligent Transportation Systems
- Traffic Safety Analytics
- Reliability Modeling
- Risk Assessment
- Safety-Critical AI

---

# Author

**Deepak Battula**

Computer Science Engineering – Artificial Intelligence (CAI)  
G Pullaiah College of Engineering and Technology  
Kurnool, India

📧 deepakbattula@outlook.com

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{battula2026,
title={A Reliability-Aware and Explainable Framework for Vehicular Accident Severity Prediction},
author={Deepak Battula},
year={2026}
}
```

---

## Disclaimer

This project is intended for research and educational purposes. It should not be used as the sole decision-making system in real-world autonomous driving or emergency response systems without additional validation and safety mechanisms.