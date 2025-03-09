import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import random
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split, learning_curve
from scipy.stats import uniform, randint
from sklearn.model_selection import cross_val_score

# Create output directory
output_dir = 'model/plots'
os.makedirs(output_dir, exist_ok=True)

# Load dataset
features_df = pd.read_csv('model/features_dataset.csv')
X = features_df.drop('label', axis=1)
y = features_df['label']

# Define a function to randomly modify the labels within a ±2.5 range to change the clinic setup that rounded the labels
def modify_labels(y_true):
    modified_labels = []
    for true_value in y_true:
        # Add random noise between -2.5 and 2.5 for each ground truth value
        modified_value = true_value + random.uniform(-2.5, 2.5)
        modified_labels.append(modified_value)
    return np.array(modified_labels)
y = modify_labels(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balance the dataset using resampling
balanced_X, balanced_y = resample(X_scaled, y, replace=True, random_state=42)

# Hyperparameter Grids for Randomized Search
gb_param_dist = {
    'n_estimators': randint(100, 1000),  # Reduced the number of estimators
    'learning_rate': uniform(0.01, 0.1),  # Reduced the learning rate
    'max_depth': randint(3, 10),  # Reduced max depth
    'min_samples_split': randint(5, 15),  # Increased min_samples_split
    'min_samples_leaf': randint(2, 6),  # Increased min_samples_leaf
    'subsample': uniform(0.7, 0.3),  # Added subsampling for regularization
    'max_features': uniform(0.1, 0.9)  # Regularize by limiting max features
}

rf_param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [5, 10],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Randomized Search for Gradient Boosting
random_search_gb = RandomizedSearchCV(GradientBoostingRegressor(random_state=42),
                                      gb_param_dist, n_iter=100, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search_gb.fit(balanced_X, balanced_y)
best_gb_model = random_search_gb.best_estimator_

# Grid Search for Random Forest
grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search_rf.fit(balanced_X, balanced_y)
best_rf_model = grid_search_rf.best_estimator_

# Evaluate models
models = {"Random Forest": best_rf_model, "Gradient Boosting": best_gb_model}
for model_name, model in models.items():
    y_pred = model.predict(X_scaled)
    residuals = y - y_pred
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    mape = mean_absolute_percentage_error(y, y_pred)
    
    # Print performance results
    print(f"Model Performance: {model_name}")
    print(f"Best Parameters: {random_search_gb.best_params_ if model_name == 'Gradient Boosting' else grid_search_rf.best_params_}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.4f}\n")
    
    # Plot Actual vs. Predicted
    plt.figure(figsize=(6, 5))
    plt.scatter(y, y_pred, alpha=0.6, label='Predicted')
    plt.axline([0, 0], slope=1, color='red', linestyle='--', label='Perfect Fit')  # Perfect Fit line
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs. Predicted ({model_name})')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'actual_vs_predicted_{model_name}.png'))
    plt.close()

    # Add a line plot to show predictions vs ground truth (actual vs predicted)
    plt.figure(figsize=(6, 5))
    plt.plot(y, y_pred, 'o', label='Predictions', alpha=0.6)
    plt.plot([min(y), max(y)], [min(y), max(y)], '--', color='red', label='Perfect Fit')  # Line for perfect prediction
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Predictions vs Ground Truth ({model_name})')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'predictions_vs_ground_truth_{model_name}.png'))
    plt.close()

    # Add line plot for ground truth vs predictions over sample index
    sample_indices = np.arange(len(y))  # Generate sample indices (from 0 to len(y)-1)
    plt.figure(figsize=(10, 6))
    plt.plot(sample_indices, y, label='Ground Truth', color='blue', linestyle='-', marker='o')
    plt.plot(sample_indices, y_pred, label='Predictions', color='red', linestyle='--', marker='x')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title(f'Ground Truth vs Predictions Over Sample Index ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'ground_truth_vs_predictions_line_plot_{model_name}.png'))
    plt.close()

    # Residual Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'Residual Plot ({model_name})')
    plt.savefig(os.path.join(output_dir, f'residual_plot_{model_name}.png'))
    plt.close()

    # Learning Curve
    train_sizes, train_scores, val_scores = learning_curve(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, -np.mean(train_scores, axis=1), label='Training Error', color='blue', linestyle='--')
    plt.plot(train_sizes, -np.mean(val_scores, axis=1), label='Validation Error', color='red', linestyle='-')
    plt.xlabel('Training Set Size')
    plt.ylabel('Error (MSE)')
    plt.title(f'Learning Curve for {model_name}')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'learning_curve_{model_name}.png'))
    plt.close()

    # Feature Importance
    if model_name == "Random Forest":
        importances = model.feature_importances_
    else:
        importances = model.feature_importances_

    sorted_idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.bar(X.columns[sorted_idx], importances[sorted_idx])
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title(f"Feature Importance ({model_name})")
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(output_dir, f'feature_importance_{model_name}.png'))
    plt.close()

    # SHAP Summary Plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(os.path.join(output_dir, f'shap_summary_{model_name}.png'))
    plt.close()

    # Cross-validation error plot
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
    plt.figure(figsize=(8, 6))
    plt.boxplot(cv_scores, vert=False, patch_artist=True, 
                boxprops=dict(facecolor='skyblue', color='blue'), 
                whiskerprops=dict(color='blue'), flierprops=dict(markerfacecolor='r', marker='o'))
    plt.xlabel('Negative Mean Squared Error')
    plt.title(f'Cross-validation Performance for {model_name}')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'cv_error_plot_{model_name}.png'))
    plt.close()


# Stacking Model (Ensemble)
stack_estimators = [
    ('rf', RandomForestRegressor(random_state=42, max_depth=10, n_estimators=300)),
    ('gb', GradientBoostingRegressor(random_state=42, learning_rate=0.05, n_estimators=500))
]
stack_model = StackingRegressor(estimators=stack_estimators, final_estimator=Ridge(alpha=1.0))
stack_model.fit(X_scaled, y)

# Evaluate Stacking Model
y_pred_stack = stack_model.predict(X_scaled)
stack_mae = mean_absolute_error(y, y_pred_stack)
stack_mse = mean_squared_error(y, y_pred_stack)
stack_rmse = np.sqrt(stack_mse)
stack_r2 = r2_score(y, y_pred_stack)
stack_mape = mean_absolute_percentage_error(y, y_pred_stack)

print(f"\nStacking Model Performance")
print(f"MAE: {stack_mae:.4f}")
print(f"MSE: {stack_mse:.4f}")
print(f"RMSE: {stack_rmse:.4f}")
print(f"R²: {stack_r2:.4f}")
print(f"MAPE: {stack_mape:.4f}")
