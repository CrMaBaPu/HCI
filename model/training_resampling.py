# Missing imports
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
import random
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, learning_curve, cross_val_score
from scipy.stats import randint, uniform

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
    'n_estimators': randint(100, 1000),
    'learning_rate': uniform(0.01, 0.1),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(5, 15),
    'min_samples_leaf': randint(2, 6),
    'subsample': uniform(0.7, 0.3),
    'max_features': uniform(0.1, 0.9),
}

rf_param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [5, 10],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# ElasticNet model (Combination of Ridge and Lasso)
elastic_net_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
elastic_net_model.fit(balanced_X, balanced_y)

# Randomized Search for Gradient Boosting with Early Stopping
random_search_gb = RandomizedSearchCV(GradientBoostingRegressor(random_state=42),
                                      gb_param_dist, n_iter=100, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
random_search_gb.fit(balanced_X, balanced_y)
best_gb_model = random_search_gb.best_estimator_

# Grid Search for Random Forest
grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search_rf.fit(balanced_X, balanced_y)
best_rf_model = grid_search_rf.best_estimator_

# Linear Regression Model (Ridge and Lasso for regularization)
ridge_model = Ridge(alpha=1.0)  # Ridge Regression (L2 regularization)
lasso_model = Lasso(alpha=0.1)  # Lasso Regression (L1 regularization)
ridge_model.fit(balanced_X, balanced_y)
lasso_model.fit(balanced_X, balanced_y)

# Evaluate models
models = {
    "Random Forest": best_rf_model,
    "Gradient Boosting": best_gb_model,
    "Ridge Regression": ridge_model,
    "Lasso Regression": lasso_model,
    "ElasticNet": elastic_net_model
}

# Perform evaluation for all models
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
    print(f"Best Parameters: {random_search_gb.best_params_ if model_name == 'Gradient Boosting' else grid_search_rf.best_params_ if model_name == 'Random Forest' else 'N/A'}")
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
    plt.title(f'Ground Truth vs Predicted Values ({model_name})')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'ground_truth_vs_predicted_{model_name}.png'))
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

    # Add a line plot showing the target (y) values against the sample index (x-axis)
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(y)), y, label="Target Values", color='blue', linestyle='-', marker='o')
    plt.plot(np.arange(len(y)), y_pred, label="Predicted Values", color='red', linestyle='--', marker='x')  # Add predicted values line
    plt.xlabel('Sample Index')
    plt.ylabel('Target Value')
    plt.title(f'Target vs Predicted Values ({model_name})')
    plt.legend(loc='upper right')  # Add a legend to differentiate the lines
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'target_vs_predicted_values_{model_name}.png'))
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
    if model_name == "Random Forest" or model_name == "Gradient Boosting":
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
    if model_name == "Random Forest" or model_name == "Gradient Boosting":
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
