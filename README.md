# 🚦 Reliability-Aware and Explainable Framework for Vehicular Accident Severity Prediction

A machine learning framework for **traffic accident severity prediction** that combines **Random Forest-based risk assessment**, a **Relative Reliability Index (RRI)**, deterministic safety rules, and explainable safety recommendations. The framework emphasizes **reliability, interpretability, and safety-aware decision support** rather than optimizing only predictive accuracy.

---

## 📌 Overview

Road accident severity prediction models often prioritize accuracy while neglecting uncertainty and interpretability. This project introduces a **Reliability-Aware Framework** that combines probabilistic machine learning with deterministic safety constraints and retrieval-based explanations.

The framework:

- Predicts accident severity using a Random Forest model.
- Quantifies confidence using the Relative Reliability Index (RRI).
- Applies fail-safe deterministic safety rules during hazardous conditions.
- Generates human-readable safety recommendations.
- Improves reliability under uncertainty and distribution shifts.

---

## ✨ Features

- 🌲 Random Forest-based severity prediction
- 📊 Relative Reliability Index (RRI)
- ⚠️ Deterministic safety override mechanism
- 🔍 Explainable AI with retrieval-based recommendations
- 🌎 Cross-region validation across all 50 U.S. states
- 📈 Feature importance and SHAP analysis
- 🔄 Reliability-aware decision support
- 🚗 Designed for future real-time deployment

---

## 🏗 Architecture

```text
                     Input Features
                             │
          ┌──────────────────┴─────────────────┐
          │                                    │
          ▼                                    ▼
 Random Forest Classifier          Deterministic Safety Rules
          │                                    │
          └─────────────┬──────────────────────┘
                        ▼
            Relative Reliability Index (RRI)
                        │
                        ▼
          Retrieval-Based Explanation Module
                        │
                        ▼
              Risk Score + Safety Advisory
```

---

## 📂 Dataset

The framework uses the **US Accidents Dataset**, containing more than **7.7 million traffic records**.

For experimentation, a cleaned **post-pandemic subset (2021–2023)** containing approximately **500,000 records** was used to better represent modern traffic conditions.

### Features Used

- Weather conditions
- Visibility
- Temperature
- Hour of day
- Weekday
- Regional clustering

---

## 🧠 Methodology

### 1. Random Forest Prediction

A Random Forest classifier is used to model nonlinear interactions among environmental and temporal features.

### 2. Relative Reliability Index (RRI)

Instead of interpreting raw probabilities directly, predictions are transformed into a reliability score ranging from **0–100**.

The RRI:

- Suppresses low-confidence predictions.
- Emphasizes high-certainty situations.
- Improves interpretability.
- Enables reliability-aware decision making.

### 3. Deterministic Safety Rules

Extreme conditions such as poor visibility can trigger conservative safety overrides to reduce dangerous false negatives.

### 4. Explainable Recommendation Module

Retrieval-based explanations provide context-aware safety advisories in human-readable form.

---

## 📊 Models Evaluated

| Model | Accuracy | Weighted F1 Score |
|---------|---------:|---------:|
| Proposed Framework (Random Forest) | 88.03% | 84.68% |
| SVM | 90.21% | 85.57% |
| AdaBoost | 90.21% | 85.57% |
| MLP | 90.00% | 85.00% |
| XGBoost + SMOTE | 85.00% | 83.00% |

Although SVM and AdaBoost achieve slightly higher accuracy, they exhibit strong majority-class bias, making them less suitable for safety-critical applications.

---

## 📈 Key Results

### Performance

- **Accuracy:** 88.03%
- **Weighted F1 Score:** 84.68%
- **Stability:** ±0.0024 across randomized runs
- **Average inference time:** 3.1 ms/sample
- **Training time:** 38.2 seconds

### Reliability

- RRI identifies uncertain predictions.
- Deterministic rules improve safety under hazardous conditions.
- More than 90% of severe-weather false negatives are flagged by the reliability mechanism.

---

## 🔬 Feature Importance

Most influential features:

1. Hour of day
2. Region
3. Weekday
4. Visibility
5. Weather conditions
6. Temperature

Temporal and regional patterns dominate accident severity prediction, while environmental variables provide additional context.

---

## 🛠 Tech Stack

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

### Tools

- Jupyter Notebook
- Git
- GitHub

---

## 📁 Project Structure

```text
TrafficSeverity-Prediction/
│
├── data/
│
├── notebooks/
│
├── models/
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   ├── reliability.py
│   ├── rules.py
│   └── explainability.py
│
├── results/
├── figures/
├── requirements.txt
└── README.md
```

---

## 🎯 Applications

- Intelligent Transportation Systems
- Risk Assessment Platforms
- Driver Assistance Systems
- Explainable AI Research
- Reliability-Aware Machine Learning
- Traffic Safety Analytics

---

## 🔮 Future Work

- Probability calibration techniques
- OBD-II vehicle integration
- Real-time deployment
- Edge-device optimization
- Advanced uncertainty quantification
- Graph Neural Networks
- Transformer-based architectures

---

## 📚 References

- Breiman, L. (2001). Random Forests.
- Moosavi et al. (2019). US Accidents Dataset.
- Arrieta et al. (2020). Explainable Artificial Intelligence.
- Lewis et al. (2020). Retrieval-Augmented Generation.
- Wang et al. (2019). Road Traffic Accident Severity Prediction Using Machine Learning Models.

---

## 📖 Citation

```bibtex
@article{battula2026trafficseverity,
  title={A Reliability-Aware and Explainable Framework for Vehicular Accident Severity Prediction},
  author={Battula, Deepak},
  year={2026}
}
```

---

## 👨‍💻 Author

**Deepak Battula**  
B.Tech Computer Science Engineering (Artificial Intelligence)  
G Pullaiah College of Engineering and Technology  
Kurnool, India

📧 deepakbattula@outlook.com

---

## 📜 License

This project is intended for research and educational purposes.

---