import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV

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

# Balance the dataset using resampling
balanced_X, balanced_y = resample(X_scaled, y, replace=True, random_state=42)

# Define models and hyperparameter grids
rf_param_grid = {
    'n_estimators': [500, 1000, 2000],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

boosting_param_grid = {
    'n_estimators': [500, 1000, 2000],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search for Random Forest
rf_model = RandomForestRegressor(random_state=42)
grid_search_rf = GridSearchCV(rf_model, rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search_rf.fit(balanced_X, balanced_y)

# Perform Grid Search for Gradient Boosting
gb_model = GradientBoostingRegressor(random_state=42)
grid_search_gb = GridSearchCV(gb_model, boosting_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
grid_search_gb.fit(balanced_X, balanced_y)

# Select the best models
best_rf_model = grid_search_rf.best_estimator_
best_gb_model = grid_search_gb.best_estimator_

# Evaluate models
models = {"Random Forest": best_rf_model, "Gradient Boosting": best_gb_model}
for model_name, model in models.items():
    y_pred = model.predict(X_scaled)
    residuals = y - y_pred
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    
    # Print results
    print(f"Model Performance: {model_name}")
    print(f"Best Parameters: {grid_search_rf.best_params_ if model_name == 'Random Forest' else grid_search_gb.best_params_}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}\n")
    
    # Plot Actual vs. Predicted
    plt.figure(figsize=(6, 5))
    plt.scatter(y, y_pred, alpha=0.6, label='Predicted')
    plt.axline([0, 0], slope=1, color='red', linestyle='--', label='Perfect Fit')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs. Predicted ({model_name})')
    plt.legend()
    plt.savefig(os.path.join(output_dir, f'actual_vs_predicted_{model_name}.png'))
    plt.close()

# Feature Importance (Random Forest only)
importances = best_rf_model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.bar(X.columns[sorted_idx], importances[sorted_idx])
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.xticks(rotation=90)
plt.savefig(os.path.join(output_dir, 'feature_importance_rf.png'))
plt.close()

# SHAP Analysis RF only
explainer = shap.TreeExplainer(best_rf_model)
shap_values = explainer.shap_values(X_scaled)
shap.summary_plot(shap_values, X, show=False)
plt.savefig(os.path.join(output_dir, 'shap_summary_rf.png'))
plt.close()
