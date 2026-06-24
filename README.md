# A Reliability-Aware and Explainable Framework for Vehicular Accident Severity Prediction

> Reliability-aware machine learning framework for traffic accident severity prediction with uncertainty estimation and explainable safety guidance.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Random Forest](https://img.shields.io/badge/Model-Random%20Forest-green)
![XAI](https://img.shields.io/badge/Explainable-AI-purple)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## Overview

This project presents a **reliability-aware and explainable machine learning framework** for predicting vehicular accident severity using large-scale traffic data.

Unlike conventional classification systems that focus solely on accuracy, this framework integrates:

- Probabilistic machine learning
- Reliability estimation through the Relative Reliability Index (RRI)
- Deterministic safety rules for fail-safe behavior
- Explainable safety recommendations
- Cross-region robustness evaluation

The framework was developed using a post-pandemic subset of the **US Accidents Dataset (2021–2023)** containing approximately **500,000 records**.

---

## Key Features

- **Accident severity prediction**
- **Relative Reliability Index (RRI)** for uncertainty-aware inference
- **Hybrid architecture combining ML and rule-based safety logic**
- **Explainable safety recommendations**
- **Cross-state robustness validation across 50 U.S. states**
- **Feature importance and SHAP analysis**
- **Fast inference (~3 ms/sample)**

---

## System Architecture

Traffic Data
│
├── Environmental Features
├── Temporal Features
└── Regional Features
↓
Random Forest Classifier
↓
Relative Reliability Index (RRI)
↓
Deterministic Safety Layer
↓
Explainable Recommendation Module
↓
Risk Assessment + Safety Advisory


---

## Dataset

**Source:** US Accidents Dataset

**Period Used:** 2021–2023

**Records:** ~500,000

### Features

- Visibility
- Temperature
- Weather conditions
- Hour of day
- Weekday
- Regional clusters

### Target Classes

| Severity | Description |
|------------|-------------|
| 1 | Low |
| 2 | Moderate |
| 3 | High |
| 4 | Critical |

---

## Relative Reliability Index (RRI)

Traditional models often produce overconfident predictions.

This framework introduces the **Relative Reliability Index (RRI)**, which transforms model confidence into a bounded risk score:

RRI = 0 if P < Pmin
RRI = normalized if Pmin ≤ P < Pmax
RRI = 100 if P ≥ Pmax


The RRI provides:

- Better uncertainty interpretation
- Conservative risk signaling
- Improved edge-case awareness
- Fail-safe decision support

---

## Models Evaluated

| Model | Accuracy | Weighted F1 |
|---------|---------|---------|
| Random Forest (Proposed) | 88.03% | 84.68% |
| SVM | 90.21% | 85.57% |
| AdaBoost | 90.21% | 85.57% |
| MLP | 90.00% | 85.00% |
| XGBoost + SMOTE | 85.00% | 83.00% |

Although several models achieved slightly higher accuracy, Random Forest was selected due to:

- Better stability
- Higher interpretability
- Probability outputs enabling reliability-aware inference
- Compatibility with deterministic safety rules

---

## Performance

### Weighted F1 Score

**0.8468**

### Accuracy

**88.03%**

### Stability

**±0.0024**

### Average Inference Time

**3.1 ms per sample**

### Training Time

**38.2 seconds**

---

## Explainability

The framework incorporates:

- Feature importance analysis
- SHAP interaction analysis
- Context-aware safety recommendations
- Human-readable explanations

Major contributing factors:

1. Hour of day
2. Region
3. Weekday
4. Visibility
5. Weather
6. Temperature

---

## Reliability-Oriented Design

Unlike conventional classifiers, this framework prioritizes:

- Safety over raw accuracy
- Conservative behavior under uncertainty
- Edge-case awareness
- Interpretability
- Robustness across geographical regions

This makes the system suitable for:

- Intelligent Transportation Systems
- Driver Assistance Systems
- Fleet Risk Assessment
- Insurance Analytics
- Smart Mobility Platforms

---

## Tech Stack

### Languages

- Python

### Libraries

- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn

### Techniques

- Random Forest
- SMOTE
- Feature Engineering
- Explainable AI (XAI)
- Reliability Modeling
- SHAP Analysis

---

## Future Work

- Probability calibration methods
- Real-time streaming inference
- OBD-II vehicle integration
- Edge deployment
- Lightweight APIs
- LLM-assisted explanation generation
- Real-time traffic risk monitoring

---

## Author

**Deepak Battula**

Computer Science Engineering (Artificial Intelligence)

Email: deepakbattula@outlook.com

LinkedIn: *Add your profile*

GitHub: *Add your GitHub URL*

---

## Citation

If you use this work, please cite:

> Deepak Battula, *A Reliability-Aware and Explainable Framework for Vehicular Accident Severity Prediction*, 2026.
