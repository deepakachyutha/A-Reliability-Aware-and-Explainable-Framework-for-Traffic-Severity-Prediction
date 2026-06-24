from ast import mod
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

import seaborn as sns
import matplotlib as mpl

sns.set_theme(style="whitegrid")

mpl.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 120,
    "savefig.dpi": 600,

    "font.family": "serif",
    "font.size": 12,

    "axes.titlesize": 14,
    "axes.labelsize": 12,

    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    "axes.spines.top": False,
    "axes.spines.right": False,

    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

print("Training Random Forest (Safety-Optimized)...")


# 1. Load & Sample
data_path = "data/modern_training_data.csv"
if not os.path.exists(data_path): data_path = "notebook/modern_training_data.csv"
df = pd.read_csv(data_path)

if len(df) > 100000: df = df.sample(100000, random_state=42)

X = df.drop(columns=['Severity'], errors='ignore')
y = df['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average='weighted')

print(f"Random Forest Accuracy: {acc:.4f} | F1: {f1:.4f}")

os.makedirs("models", exist_ok=True)
os.makedirs("research_files", exist_ok=True)

joblib.dump(model, "models/random_forest_model.pkl")

with open("research_files/report_Random_Forest.txt", "w") as f:
    f.write(classification_report(y_test, preds))

plt.figure(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(y_test, preds, cmap='Greens', values_format='d')
plt.title(f"Random Forest (Balanced)\nAccuracy: {acc:.2%}")
plt.savefig("research_files/confusion_matrix_Random_Forest.png")
print("Done.")

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True,
        linewidths=0.5,
        linecolor='gray'
    )

    plt.title(title, weight='bold')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    plt.savefig(f"{filename}.png", bbox_inches='tight')
    plt.savefig(f"{filename}.pdf", bbox_inches='tight')

    plt.close()

plot_confusion_matrix(
    y_test,
    preds,
    f"Random Forest (Balanced)\nAccuracy: {acc:.2%} | F1: {f1:.2%}",
    "research_files/confusion_matrix_random_forest"
)


from sklearn.metrics import classification_report

y_pred_rf = model.predict(X_test)

print("=== STRADA (Random Forest) Detailed Report ===")
print(classification_report(y_test, y_pred_rf))

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# 1. Apply SMOTE to generate synthetic severe accidents for training
smote = SMOTE(sampling_strategy='not majority', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
y_train_resampled = y_train_resampled - 1

# 2. Train XGBoost (Modern Tree Baseline)
xgb_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_resampled, y_train_resampled)
print("=== XGBoost + SMOTE Report ===")
print(classification_report(y_test, xgb_model.predict(X_test)))

# 3. Train MLP (Deep Learning Baseline)
mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=300, random_state=42)
mlp_model.fit(X_train_resampled, y_train_resampled)
print("=== Deep MLP + SMOTE Report ===")
print(classification_report(y_test, mlp_model.predict(X_test)))


xgb_preds = xgb_model.predict(X_test) + 1

plot_confusion_matrix(
    y_test,
    xgb_preds,
    "XGBoost + SMOTE",
    "research_files/confusion_matrix_xgboost"
)

mlp_preds = mlp_model.predict(X_test) + 1

plot_confusion_matrix(
    y_test,
    mlp_preds,
    "Deep MLP + SMOTE",
    "research_files/confusion_matrix_mlp"
)


def plot_normalized_cm(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap='Greens',
        cbar=True,
        linewidths=0.5
    )

    plt.title(title + " (Normalized)", weight='bold')
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    plt.savefig(f"{filename}_normalized.png", bbox_inches='tight')
    plt.savefig(f"{filename}_normalized.pdf", bbox_inches='tight')

    plt.close()

plot_normalized_cm(y_test, preds, "Random Forest", "research_files/rf")
plot_normalized_cm(y_test, xgb_preds, "XGBoost", "research_files/xgb")
plot_normalized_cm(y_test, mlp_preds, "MLP", "research_files/mlp")