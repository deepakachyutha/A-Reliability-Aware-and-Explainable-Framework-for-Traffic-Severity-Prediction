from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import os

data_path = "data/modern_training_data.csv"
if not os.path.exists(data_path): data_path = "notebook/modern_training_data.csv"
df = pd.read_csv(data_path)

if len(df) > 100000: df = df.sample(100000, random_state=42)    

X = df.drop(columns=['Severity'], errors='ignore')
y = df['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



smote = SMOTE(sampling_strategy='not majority', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 2. Shift labels from [1, 2, 3, 4] to [0, 1, 2, 3] just for XGBoost
y_train_xgb = y_train_resampled - 1 

# 3. Train XGBoost
xgb_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_resampled, y_train_xgb)

# 4. Predict and shift labels BACK to [1, 2, 3, 4] to match y_test
y_pred_xgb_raw = xgb_model.predict(X_test)
y_pred_xgb = y_pred_xgb_raw + 1 

# 5. Evaluate
print("=== XGBoost + SMOTE Report ===")
print(classification_report(y_test, y_pred_xgb))