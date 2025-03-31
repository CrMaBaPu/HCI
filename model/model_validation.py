import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Switch to Agg backend for non-GUI plotting
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
print("Loading dataset...")
data = pd.read_csv('model/features_dataset.csv')

# Encode categorical columns
print("Encoding categorical features...")
categorical_cols = ['category', 'criticality', 'segment', 'most_frequent_class']
for col in categorical_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

# Separate features and target
X = data.drop(columns=["most_common_arduino", "person_id", "category", "criticality", "file_id", "segment"])
y = data['most_common_arduino']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training Random Forest model...")
rf = RandomForestRegressor(max_depth=15, min_samples_split=2, n_estimators=1000, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Create directory for plots
plots_dir = Path('plots_validation')
plots_dir.mkdir(parents=True, exist_ok=True)

# Plot Ground Truth vs. Predictions
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', label='Perfect Fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Values')
plt.legend()
plt.grid(True)
plt.savefig(plots_dir / 'ground_truth_vs_predicted.png')
plt.close()

# Plot Residuals
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Histogram')
plt.savefig(plots_dir / 'residual_histogram.png')
plt.close()

# Feature Importance
importances = rf.feature_importances_
feature_names = X_train.columns
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10,8))
plt.bar(feature_names[sorted_idx], importances[sorted_idx], color='blue')
plt.xticks(rotation=90)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.savefig(plots_dir / 'feature_importance.png')
plt.close()

# SHAP Analysis
explainer = shap.Explainer(rf, X_train)
shap_values = explainer(X_test)

plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig(plots_dir / 'shap_summary.png', bbox_inches='tight')
plt.close()

# Learning Curves
print("Generating learning curves...")

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestRegressor(n_estimators=1000, max_depth=15, min_samples_split=2, random_state=42, n_jobs=-1),
    X_train, y_train, scoring="neg_mean_absolute_error", train_sizes=np.linspace(0.1, 0.9, 10), cv=5, n_jobs=-1
)

train_scores_mean = -train_scores.mean(axis=1)
val_scores_mean = -val_scores.mean(axis=1)

# Plot Learning Curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, label='Training Error', color='blue')
plt.plot(train_sizes, val_scores_mean, label='Validation Error', color='red')
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plt.savefig(plots_dir / 'learning_curves.png')
plt.close()

print("All plots saved successfully.")
