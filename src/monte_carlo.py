import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("/Users/deepakachyutha/deepak/STARDA-traffic risk analysis project/data/modern_training_data.csv")

X = df.drop(columns=['Severity'])
y = df['Severity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


state_results = []
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 2. Loop through every unique state in your test set
for state in X_test['Region_Cluster'].unique(): # Change 'State' to your actual column name
    # Filter the test data for just this state
    state_mask = X_test['Region_Cluster'] == state
    X_state = X_test[state_mask]
    y_state_true = y_test[state_mask]
    
    # Skip states with too little data in the test set to be statistically significant
    if len(y_state_true) < 50:
        continue
        
    # Get predictions and calculate the F1 score for this specific state
    y_state_pred = rf_model.predict(X_state)
    state_f1 = f1_score(y_state_true, y_state_pred, average='weighted')
    
    state_results.append({'State': state, 'Weighted_F1': state_f1})

df_results = pd.DataFrame(state_results)

# 3. Plot the "Monte Carlo" style distribution graph
plt.figure(figsize=(10, 5))
sns.boxplot(x=df_results["Weighted_F1"], color="lightgray", showfliers=False)
sns.swarmplot(x=df_results["Weighted_F1"], color="crimson", size=7, alpha=0.7)

# Add the Global Mean line
plt.axvline(x=0.85, color='black', linestyle='--', label='Global Mean (0.85)')

plt.title("Geospatial Robustness: Model F1-Score Across 50 U.S. States", fontsize=14)
plt.xlabel("Weighted F1-Score", fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("/Users/deepakachyutha/deepak/STARDA-traffic risk analysis project/research_files/state_distribution.pdf", dpi=300)
plt.show()

# 4. Print the standard deviation for the paper
print(f"State F1 Standard Deviation: ±{df_results['Weighted_F1'].std():.4f}")
