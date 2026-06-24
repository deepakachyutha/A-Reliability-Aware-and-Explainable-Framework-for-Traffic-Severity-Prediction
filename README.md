# A Reliability-Aware and Explainable Framework for Vehicular Accident Severity Prediction

> A machine learning framework for traffic accident severity prediction that combines reliability-aware risk assessment and explainable safety recommendations.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Research](https://img.shields.io/badge/Research-Framework-green)

---

## Overview

Road accidents remain a major public safety challenge worldwide. Although machine learning has demonstrated strong predictive capabilities, many existing approaches prioritize accuracy while overlooking reliability and interpretability.

This repository contains the implementation accompanying the research paper:

**"A Reliability-Aware and Explainable Framework for Vehicular Accident Severity Prediction."**

The framework combines machine learning, reliability modeling, and explainable safety support to provide a robust foundation for intelligent transportation systems.

---

## Motivation

Traffic accident datasets are highly imbalanced and contain significant uncertainty. Traditional machine learning models often become overconfident and struggle to recognize rare but critical events.

This work aims to:

- Predict accident severity using environmental and temporal data.
- Introduce reliability-aware confidence interpretation.
- Provide explainable and human-readable safety guidance.
- Study machine learning behavior in safety-critical domains.
- Build a modular framework for future intelligent transportation systems.

---

# Features

### Reliability-Aware Prediction Framework

Transforms raw model confidence into interpretable reliability scores.

### Accident Severity Prediction

Predicts accident severity from contextual traffic and environmental information.

### Explainable AI

Provides feature importance analysis and SHAP-based explanations.

### Temporal Risk Analysis

Captures accident trends across hours, weekdays, and seasons.

### Regional Risk Modeling

Uses clustering techniques to account for geographical variations.

### Safety Recommendation Module

Generates contextual and human-readable safety suggestions.

### Interactive Dashboard

Streamlit-based interface for experimentation and visualization.

---

# Dataset

The project utilizes the **US Accidents Dataset**, a large-scale collection of accident records across the United States.

| Property | Value |
|------------|--------|
| Total records | ~7.7 Million |
| Training subset | ~500,000 |
| Time period | 2021–2023 |
| States covered | 50 |
| Severity classes | 4 |

Severity labels:

| Class | Meaning |
|-------|---------|
| 1 | Minor |
| 2 | Moderate |
| 3 | Severe |
| 4 | Fatal |

> The dataset is not included in this repository due to size constraints.

---

# Features Used

The framework incorporates:

- Weather conditions
- Temperature
- Visibility
- Hour of day
- Day of week
- Month
- Regional clusters

These variables capture environmental and temporal factors affecting accident severity.

---

# Models Evaluated

Several machine learning models were benchmarked:

- Random Forest
- Support Vector Machine (SVM)
- AdaBoost
- Multi-Layer Perceptron (MLP)
- XGBoost + SMOTE

Random Forest was selected as the primary model because of its stability, robustness, and interpretability.

---

# Framework Architecture

```
Input Data
     │
     ▼
Preprocessing
     │
     ▼
Feature Engineering
     │
     ▼
Random Forest Classifier
     │
     ▼
Reliability Index
     │
     ▼
Explainability Layer
     │
     ▼
Safety Recommendation Module
```

---

# Technologies Used

### Programming

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
.
├── data/                  # Dataset (excluded)
├── notebook/              # EDA and experiments
├── paper/                 # Research paper
├── research_files/        # Figures and outputs
├── Plots and Images/
├── src/
│   ├── monte_carlo.py
│   ├── run_xgboost.py
│   └── training scripts
├── models/
├── app.py
├── requirements.txt
└── README.md
```

---

# Experimental Results

Performance on approximately 500,000 post-pandemic accident records:

| Model | Accuracy | Weighted F1 |
|---------|---------|---------|
| Random Forest | 88.03% | 84.68% |
| SVM | 90.21% | 85.57% |
| AdaBoost | 90.21% | 85.57% |
| MLP | 90.00% | 85.00% |
| XGBoost + SMOTE | 85.00% | 83.00% |

The proposed framework prioritizes reliability and interpretability over purely maximizing accuracy.

---

# Future Work

Future extensions include:

- Real-time vehicle telemetry integration (OBD-II)
- Edge deployment using Raspberry Pi
- Probability calibration techniques
- Advanced uncertainty estimation
- Real-time traffic risk monitoring
- Calibration-aware learning
- Intelligent transportation systems

---

# Research Areas

- Machine Learning
- Explainable AI
- Reliability Modeling
- Intelligent Transportation Systems
- Traffic Safety Analytics
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

# Citation

```bibtex
@article{battula2026,
title={A Reliability-Aware and Explainable Framework for Vehicular Accident Severity Prediction},
author={Deepak Battula},
year={2026}
}
```

---

# Disclaimer

This repository is intended for research and educational purposes. It should not be used as the sole decision-making system in autonomous vehicles or emergency-response systems without additional validation and safety mechanisms.