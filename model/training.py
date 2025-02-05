import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Set the Agg backend for headless environments
import matplotlib.pyplot as plt  # Use the Agg backend for headless environments
import os

# Create a directory to save plots (if it doesn't exist)
output_dir = 'model/plots'
os.makedirs(output_dir, exist_ok=True)

# Load dataset (replace with your actual path)
features_df = pd.read_csv('model/features_dataset.csv')

# Separate features and labels
X = features_df.drop('label', axis=1)
y = features_df['label']

# Preprocess the data (scaling features)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the Random Forest model
rf_model = RandomForestRegressor(n_estimators=1000, criterion="squared_error", random_state=42)

# K-Fold Cross-Validation (5 folds in this case)
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Initialize lists to store results for each fold
mae_scores = []
mse_scores = []
rmse_scores = []
r2_scores = []

# Create subplots grid (5 folds, 2 plots per fold)
fig, axes = plt.subplots(5, 2, figsize=(14, 20))  # 5 rows and 2 columns
axes = axes.ravel()  # Flatten the 2D array of axes for easier indexing

fold = 1
for train_index, test_index in kf.split(X_scaled):
    # Split data into training and test sets for the current fold
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = rf_model.predict(X_test)
    
    # Calculate Residuals
    residuals = y_test - y_pred
    
    # --- Plot Actual vs Predicted --- #
    axes[fold * 2 - 2].scatter(y_test, y_pred, color='blue', alpha=0.6, label="Predicted vs Actual")
    axes[fold * 2 - 2].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label="Perfect Prediction")
    axes[fold * 2 - 2].set_title(f"Fold {fold}: Actual vs Predicted (Observed) Values")
    axes[fold * 2 - 2].set_xlabel("Actual (True) Values")
    axes[fold * 2 - 2].set_ylabel("Predicted (Observed) Values")
    axes[fold * 2 - 2].legend()
    
    # --- Plot Residuals --- #
    axes[fold * 2 - 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
    axes[fold * 2 - 1].axvline(0, color='red', linestyle='--')
    axes[fold * 2 - 1].set_title(f"Fold {fold}: Residuals Distribution")
    axes[fold * 2 - 1].set_xlabel("Residuals")
    axes[fold * 2 - 1].set_ylabel("Frequency")
    
    # Calculate evaluation metrics for this fold
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Append the metrics for this fold
    mae_scores.append(mae)
    mse_scores.append(mse)
    rmse_scores.append(rmse)
    r2_scores.append(r2)
    
    fold += 1

# Average results across all folds
avg_mae = np.mean(mae_scores)
avg_mse = np.mean(mse_scores)
avg_rmse = np.mean(rmse_scores)
avg_r2 = np.mean(r2_scores)

# Adjust layout for subplots to avoid overlap
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'folds_actual_vs_predicted_and_residuals.png'))  # Save the full figure
plt.close()

# Feature Importance Plot (sorted by importance)
importance = rf_model.feature_importances_
sorted_idx = np.argsort(importance)[::-1]  # Sort importance from most to least

plt.figure(figsize=(10, 6))
plt.bar(X.columns[sorted_idx], importance[sorted_idx])
plt.title("Feature Importance (Sorted)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))  # Save plot
plt.close()  # Close the plot to free memory

# Accuracy and Prediction Plot (Observed vs Predicted)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, label="Predicted vs Observed", color="blue", alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='-', label="Perfect Prediction")
relative_margin = 0.10
plt.plot([min(y_test), max(y_test)], [min(y_test) * (1 + relative_margin), max(y_test) * (1 + relative_margin)], 
         color='blue', linestyle='--', label="Relative Error (+10%)")
plt.plot([min(y_test), max(y_test)], [min(y_test) * (1 - relative_margin), max(y_test) * (1 - relative_margin)], 
         color='blue', linestyle='--', label="Relative Error (-10%)")
absolute_margin = 5
plt.plot([min(y_test), max(y_test)], [min(y_test) + absolute_margin, max(y_test) + absolute_margin], 
         color='green', linestyle='--', label="Absolute Error (+5)")  
plt.plot([min(y_test), max(y_test)], [min(y_test) - absolute_margin, max(y_test) - absolute_margin], 
         color='green', linestyle='--', label="Absolute Error (-5)")  
plt.xlabel("Observed Values")
plt.ylabel("Predicted Values")
plt.title("Observed vs Predicted Values")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'observed_vs_predicted.png'))  # Save plot
plt.close()

# Accuracy and Prediction Plot with Dotted Lines for Absolute Accuracy
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, label="Predicted vs Observed", color="blue", alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test) + absolute_margin, max(y_test) + absolute_margin], 
         color='green', linestyle='--', label="Absolute Error (+5)")  
plt.plot([min(y_test), max(y_test)], [min(y_test) - absolute_margin, max(y_test) - absolute_margin], 
         color='green', linestyle='--', label="Absolute Error (-5)")  
plt.xlabel("Observed Values")
plt.ylabel("Predicted Values")
plt.title("Observed vs Predicted Values with Absolute Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'observed_vs_predicted_absolute_accuracy.png'))  # Save plot
plt.close()

# SHAP values plot
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_scaled)

# SHAP summary plot (global feature importance)
shap.summary_plot(shap_values, X, show=False)  # Disable interactive plot
plt.savefig(os.path.join(output_dir, 'shap_summary_plot.png'))  # Save plot
plt.close()

# Print average results across all folds
print("=== Cross-Validation Results (Average Across All Folds) ===")
print(f"Average MAE: {avg_mae:.4f}")
print(f"Average MSE: {avg_mse:.4f}")
print(f"Average RMSE: {avg_rmse:.4f}")
print(f"Average R²: {avg_r2:.4f}")
