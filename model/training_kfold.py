import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store metrics
mae_scores, mse_scores, rmse_scores, r2_scores = [], [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    residuals = y_test - y_pred
    
    # Store metrics
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    mse_scores.append(mean_squared_error(y_test, y_pred))
    rmse_scores.append(np.sqrt(mse_scores[-1]))
    r2_scores.append(r2_score(y_test, y_pred))
    
    # Plot Actual vs. Predicted with Residuals
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred, alpha=0.6, label='Predicted')
    plt.axline([0, 0], slope=1, color='red', linestyle='--', label='Perfect Fit')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Fold {fold}: Actual vs. Predicted')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'fold_{fold}_actual_vs_predicted.png'))
    plt.close()
    
    # Line Plot: Prediction vs. Ground Truth over Samples
    plt.figure(figsize=(8, 5))
    plt.plot(range(len(y_test)), y_test, label='Actual', marker='o')
    plt.plot(range(len(y_test)), y_pred, label='Predicted', marker='s')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title(f'Fold {fold}: Prediction vs. Actual')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'fold_{fold}_prediction_vs_actual.png'))
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
print("Cross-Validation Results (Average Across Folds):")
print(f"Average MAE: {np.mean(mae_scores):.4f}")
print(f"Average MSE: {np.mean(mse_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f}")
print(f"Average R²: {np.mean(r2_scores):.4f}")
