import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler 

# Load dataset
data = pd.read_csv('model/features_dataset1.csv')

# Encode categorical columns 
categorical_cols = ['category', 'criticality', 'segment','most_frequent_class']   
label_encoders = {col: LabelEncoder() for col in categorical_cols}
for col, le in label_encoders.items():
    data[col] = le.fit_transform(data[col])
# Separate features and target
X = data.drop(columns=["most_common_arduino", "person_id", "category" ,"criticality" ,"file_id","segment"]) 
y = data['most_common_arduino']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models and hyperparameter grids
models = {
    "RandomForest": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
}

param_grids = {
    "RandomForest": {'n_estimators': [100, 500, 1000], 'max_depth': [5, 10, 15], 'min_samples_split': [2, 5, 10]},
    "GradientBoosting": {'n_estimators': [100, 500, 1000], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 6, 9]},
    "Ridge": {'alpha': [0.01, 0.1, 1, 10]},
    "Lasso": {'alpha': [0.01, 0.1, 1, 10]},
}

# Hyperparameter tuning and evaluation
best_models = {}
metrics = {}


for name, model in models.items():
    print(f"Tuning {name}...")
    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    best_models[name] = best_model
    
    # Predictions
    y_pred = best_model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    metrics[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    
    print(f"{name}: Best Params: {grid_search.best_params_}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# Plot training vs validation loss, residuals, and predictions
# Create the plots directory if it doesn't exist
plots_dir = Path('plots_tuning')
plots_dir.mkdir(parents=True, exist_ok=True)
### LEARNING CURVES ###
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
for i, (name, model) in enumerate(best_models.items()):
    # Learning curves
    train_sizes = np.linspace(0.1, 0.9, 10)
    train_scores, val_scores = [], []
    for size in train_sizes:
        X_train_subset, _, y_train_subset, _ = train_test_split( X_train, y_train, train_size=float(size), random_state=42)
        model.fit(X_train_subset, y_train_subset)
        train_pred = model.predict(X_train_subset)
        val_pred = model.predict(X_test)
        train_scores.append(mean_absolute_error(y_train_subset, train_pred))
        val_scores.append(mean_absolute_error(y_test, val_pred))
    
    ax = axes[i]
    ax.plot(train_sizes, train_scores, label='Training Error', color='blue')
    ax.plot(train_sizes, val_scores, label='Validation Error', color='red')
    ax.set_title(f'{name} Learning Curve')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('MSE')
    ax.legend()

plt.tight_layout()
learning_curve_path = plots_dir / 'learning_curves.png'
plt.savefig(learning_curve_path)
plt.close()

### RESIDUAL PLOTS ###
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
for i, (name, model) in enumerate(best_models.items()):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    ax = axes[i]
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_title(f'{name} Residual Plot')
    ax.set_xlabel('Residuals')
plt.tight_layout()
residual_plot_path = plots_dir / 'residuals.png'
plt.savefig(residual_plot_path)
plt.close()

### PREDICTIONS VS ACTUAL ###
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()
for i, (name, model) in enumerate(best_models.items()):
    y_pred = model.predict(X_test)
    ax = axes[i]
    ax.scatter(y_test, y_pred, alpha=0.8, label='Predictions')
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], '--r', label='Perfect Fit')
    ax.set_title(f'{name} Predictions vs Actual')
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.legend()
plt.tight_layout()
prediction_vs_actual_path = plots_dir / 'predictions_vs_actual.png'
plt.savefig(prediction_vs_actual_path)
plt.close()

### SHAP SUMMARY AND FEATURE IMPORTANCE PLOTS ###

for name, model in best_models.items():
    # Initialize SHAP explainer
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    
    # SHAP Summary Plot
    shap_summary_path = plots_dir / f'{name}_shap.png'
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False) 
    plt.savefig(shap_summary_path)
    plt.close()