import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Create output directory
output_dir = 'model/plots'
os.makedirs(output_dir, exist_ok=True)

# Load dataset
features_df = pd.read_csv('model/features_dataset.csv')
X = features_df.drop('label', axis=1)
y = features_df['label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize model
rf_model = RandomForestRegressor(n_estimators=2000, random_state=42)

# Balance the dataset using resampling
balanced_X, balanced_y = resample(X_scaled, y, replace=True, random_state=42)

# Train the model on the balanced dataset
rf_model.fit(balanced_X, balanced_y)
y_pred = rf_model.predict(X_scaled)
residuals = y - y_pred

# Calculate metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

# Plot Actual vs. Predicted with Residuals
plt.figure(figsize=(6, 5))
plt.scatter(y, y_pred, alpha=0.6, label='Predicted')
plt.axline([0, 0], slope=1, color='red', linestyle='--', label='Perfect Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs. Predicted')
plt.legend()
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted.png'))
plt.close()

# Line Plot: Prediction vs. Ground Truth over Samples
plt.figure(figsize=(8, 5))
plt.plot(range(len(y)), y, label='Actual', marker='o')
plt.plot(range(len(y)), y_pred, label='Predicted', marker='s')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Prediction vs. Actual')
plt.legend()
plt.savefig(os.path.join(output_dir, 'prediction_vs_actual.png'))
plt.close()

# Feature Importance
importances = rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(X.columns[sorted_idx], importances[sorted_idx])
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance")
plt.xticks(rotation=90)
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.close()

# SHAP Analysis
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_scaled)
shap.summary_plot(shap_values, X, show=False)
plt.savefig(os.path.join(output_dir, 'shap_summary.png'))
plt.close()

# Print Results
print("Model Performance on Balanced Dataset:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
