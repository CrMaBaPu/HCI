# Missing imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV, learning_curve, cross_val_score

# Create output directory
output_dir = 'model/plots'
os.makedirs(output_dir, exist_ok=True)

# Load dataset
features_df = pd.read_csv('model/features_dataset.csv')
X = features_df.drop('most_common_arduino', axis=1)
y = features_df['most_common_arduino']

# Define a function to randomly modify the labels within a ±2.5 range to change the clinic setup that rounded the labels
def modify_labels(y_true):
    modified_labels = []
    for true_value in y_true:
        modified_value = true_value + random.uniform(-2.5, 2.5)
        modified_labels.append(modified_value)
    return np.array(modified_labels)
y = modify_labels(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance the dataset using resampling 
balanced_X, balanced_y = resample(X_scaled, y, replace=True, random_state=42)

# Hyperparameter Grids for Random Forest
rf_param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [5, 10],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search for Random Forest
grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search_rf.fit(balanced_X, balanced_y)
best_rf_model = grid_search_rf.best_estimator_

# Evaluate the Random Forest Model
y_pred = best_rf_model.predict(X_scaled)
residuals = y - y_pred
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)
mape = mean_absolute_percentage_error(y, y_pred)

# Print performance results for Random Forest
print(f"Model Performance: Random Forest")
print(f"Best Parameters: {grid_search_rf.best_params_}")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")
print(f"MAPE: {mape:.4f}\n")

# Add a line plot showing ground truth (y) on the x-axis and predicted values (y_pred) on the y-axis
plt.figure(figsize=(10, 6))
plt.plot(y, y_pred, 'o', color='blue', alpha=0.6)  # Scatter plot with blue dots
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')  # Ideal line (y = x)
plt.xlabel('Ground Truth (Target Values)')
plt.ylabel('Predicted Values')
plt.title('Ground Truth vs Predicted Values (Random Forest)')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'ground_truth_vs_predicted_RandomForest.png'))
plt.close()

# Residual Plot
plt.figure(figsize=(6, 5))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot (Random Forest)')
plt.savefig(os.path.join(output_dir, 'residual_plot_RandomForest.png'))
plt.close()

# Add a line plot showing the target (y) values against the sample index (x-axis)
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(y)), y, label="Target Values", color='blue', linestyle='-', marker='o')
plt.plot(np.arange(len(y)), y_pred, label="Predicted Values", color='red', linestyle='--', marker='x')  # Add predicted values line
plt.xlabel('Sample Index')
plt.ylabel('Target Value')
plt.title('Target vs Predicted Values (Random Forest)')
plt.legend(loc='upper right')  # Add a legend to differentiate the lines
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'target_vs_predicted_values_RandomForest.png'))
plt.close()

# Learning Curve
train_sizes, train_scores, val_scores = learning_curve(best_rf_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, -np.mean(train_scores, axis=1), label='Training Error', color='blue', linestyle='--')
plt.plot(train_sizes, -np.mean(val_scores, axis=1), label='Validation Error', color='red', linestyle='-')
plt.xlabel('Training Set Size')
plt.ylabel('Error (MSE)')
plt.title('Learning Curve for Random Forest')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'learning_curve_RandomForest.png'))
plt.close()

# Feature Importance
importances = best_rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(X.columns[sorted_idx], importances[sorted_idx])
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.xticks(rotation=90)
plt.savefig(os.path.join(output_dir, 'feature_importance_RandomForest.png'))
plt.close()

# Cross-validation error plot
cv_scores = cross_val_score(best_rf_model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
plt.figure(figsize=(8, 6))
plt.boxplot(cv_scores, vert=False, patch_artist=True, 
            boxprops=dict(facecolor='skyblue', color='blue'), 
            whiskerprops=dict(color='blue'), flierprops=dict(markerfacecolor='r', marker='o'))
plt.xlabel('Negative Mean Squared Error')
plt.title('Cross-validation Performance for Random Forest')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'cv_error_plot_RandomForest.png'))
plt.close()
