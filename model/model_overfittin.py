import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.utils import resample
from pathlib import Path
from PIL import Image

# Load dataset
print("Loading dataset...")
data = pd.read_csv('model/features_dataset1.csv')

# Encode categorical columns
print("Encoding categorical features...")
categorical_cols = ['category', 'criticality', 'segment', 'most_frequent_class']
label_encoders = {col: LabelEncoder() for col in categorical_cols}
for col, le in label_encoders.items():
    data[col] = le.fit_transform(data[col])

# Separate features and target
X = data.drop(columns=["most_common_arduino", "person_id", "category", "criticality", "file_id", "segment"])
y = data['most_common_arduino']

# StandardScaler to numeric features
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create a directory for plots
plots_dir = Path('plots_overfitting')
plots_dir.mkdir(parents=True, exist_ok=True)

# Function to train and evaluate a model
def train_evaluate_model(model, X_train, y_train, X_val, y_val, title):
    print(f"Training model: {title}...")
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Compute metrics for validation data
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)
    
    return {'title': title, 'val_mae': val_mae, 'val_rmse': val_rmse, 'val_r2': val_r2, 'y_val': y_val, 'y_val_pred': y_val_pred}

# Base Model
print("Splitting dataset for base model...")
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
rf_base = RandomForestRegressor(max_depth=15, min_samples_split=2, n_estimators=1000, random_state=42)
results = []
results.append(train_evaluate_model(rf_base, X_train, y_train, X_val, y_val, "Base Model"))

# Feature Selection
print("Performing feature selection...")
selector = SelectKBest(score_func=f_regression, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_val_selected = selector.transform(X_val)
rf_feature = RandomForestRegressor(max_depth=15, min_samples_split=2, n_estimators=1000, random_state=42)
results.append(train_evaluate_model(rf_feature, X_train_selected, y_train, X_val_selected, y_val, "Feature Selection"))

# 5-Fold Cross-Validation
print("Running 5-Fold Cross-Validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Prepare lists to store the scores
fold_mae = []
fold_rmse = []
fold_r2 = []

# Perform cross-validation manually
for train_index, val_index in skf.split(X_train, y_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    rf_base.fit(X_train_fold, y_train_fold)

    y_val_pred = rf_base.predict(X_val_fold)

    fold_mae.append(mean_absolute_error(y_val_fold, y_val_pred))
    fold_rmse.append(np.sqrt(mean_squared_error(y_val_fold, y_val_pred)))
    fold_r2.append(r2_score(y_val_fold, y_val_pred))

avg_mae = np.mean(fold_mae)
avg_rmse = np.mean(fold_rmse)
avg_r2 = np.mean(fold_r2)

results.append({
    'title': '5-Fold Cross-Validation',
    'val_mae': avg_mae,
    'val_rmse': avg_rmse,
    'val_r2': avg_r2
})

# Leave-One-Person-Out Cross-Validation (LOPO)
print("Running Leave-One-Person-Out Cross-Validation...")
unique_persons = data['person_id'].unique()
lopo_results = []

for person_id in unique_persons:
    train_data = data[data['person_id'] != person_id]
    val_data = data[data['person_id'] == person_id]
    
    X_train_person = train_data.drop(columns=["most_common_arduino", "person_id", "category", "criticality", "file_id", "segment"])
    y_train_person = train_data['most_common_arduino']
    X_val_person = val_data.drop(columns=["most_common_arduino", "person_id", "category", "criticality", "file_id", "segment"])
    y_val_person = val_data['most_common_arduino']
    
    X_train_person_scaled = scaler.fit_transform(X_train_person)
    X_val_person_scaled = scaler.transform(X_val_person)
    
    rf_person = RandomForestRegressor(max_depth=15, min_samples_split=2, n_estimators=1000, random_state=42)
    rf_person.fit(X_train_person_scaled, y_train_person)
    
    y_train_pred = rf_person.predict(X_train_person_scaled)
    y_val_pred = rf_person.predict(X_val_person_scaled)
    
    val_mae = mean_absolute_error(y_val_person, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val_person, y_val_pred))
    val_r2 = r2_score(y_val_person, y_val_pred)
    
    lopo_results.append({'person_id': person_id, 'val_mae': val_mae, 'val_rmse': val_rmse, 'val_r2': val_r2})

results.append({'title': 'Leave-One-Person-Out Cross-Validation', 'val_mae': np.mean([r['val_mae'] for r in lopo_results]),
                'val_rmse': np.mean([r['val_rmse'] for r in lopo_results]), 'val_r2': np.mean([r['val_r2'] for r in lopo_results])})

# Bootstrap Resampling
print("Performing bootstrap resampling...")
X_resampled, y_resampled = resample(X_train, y_train, replace=True, n_samples=len(y_train), random_state=42)
rf_bootstrap = RandomForestRegressor(max_depth=15, min_samples_split=2, n_estimators=1000, random_state=42)
results.append(train_evaluate_model(rf_bootstrap, X_resampled, y_resampled, X_val, y_val, "Bootstrap Resampling"))

# Person ID-based Split
print("Splitting dataset for person_id-based method...")
unique_person_ids = data['person_id'].unique()
train_person_ids, test_person_ids = train_test_split(unique_person_ids, test_size=0.3, random_state=42)

train_data = data[data['person_id'].isin(train_person_ids)]
test_data = data[data['person_id'].isin(test_person_ids)]

X_train_person = train_data.drop(columns=["most_common_arduino", "person_id", "category", "criticality", "file_id", "segment"])
y_train_person = train_data['most_common_arduino']
X_test_person = test_data.drop(columns=["most_common_arduino", "person_id", "category", "criticality", "file_id", "segment"])
y_test_person = test_data['most_common_arduino']

X_train_person_scaled = scaler.fit_transform(X_train_person)
X_test_person_scaled = scaler.transform(X_test_person)

rf_person_model = RandomForestRegressor(max_depth=15, min_samples_split=2, n_estimators=1000, random_state=42)
results.append(train_evaluate_model(rf_person_model, X_train_person_scaled, y_train_person, X_test_person_scaled, y_test_person, "Person ID-based Split"))

# Evaluation Summary
evaluation_summary = pd.DataFrame(results)
evaluation_summary.dropna(subset=['val_mae'], inplace=True)
evaluation_summary = evaluation_summary[['title', 'val_mae', 'val_rmse', 'val_r2']] 
print("Evaluation Summary:")
print(evaluation_summary)

# Save the evaluation summary as CSV
evaluation_summary.to_csv(plots_dir / 'evaluation_summary.csv', index=False)

# Generating Individual Plots
print("Generating Predicted vs Actual and Residual plots...")
individual_plot_paths = []  # Store paths of individual plots
for i, res in enumerate(results):
    if 'y_val' in res:
        # Predicted vs Actual
        plt.figure(figsize=(7, 5))
        sns.scatterplot(x=res['y_val'], y=res['y_val_pred'])
        plt.plot([min(res['y_val']), max(res['y_val'])], [min(res['y_val']), max(res['y_val'])], '--r')
        plt.title(f'{res["title"]} - Predicted vs Actual')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        predicted_vs_actual_path = plots_dir / f'predicted_vs_actual_{res["title"]}.png'
        plt.savefig(predicted_vs_actual_path)
        plt.close()
        individual_plot_paths.append(predicted_vs_actual_path)
        
        # Residual Histogram
        plt.figure(figsize=(7, 5))
        sns.histplot(res['y_val'] - res['y_val_pred'], kde=True)
        plt.title(f'{res["title"]} - Residual Histogram')
        residual_histogram_path = plots_dir / f'residual_histogram_{res["title"]}.png'
        plt.savefig(residual_histogram_path)
        plt.close()
        individual_plot_paths.append(residual_histogram_path)

#Combine all plots into one overview image
print("Combining all plots into an overview image...")

method_names = [res['title'] for res in results if 'y_val' in res]
plot_rows = []  # Store rows of images

for method_name in method_names:
    scatter_path = plots_dir / f'predicted_vs_actual_{method_name}.png'
    residual_path = plots_dir / f'residual_histogram_{method_name}.png'

    if scatter_path.exists() and residual_path.exists():
        scatter_img = Image.open(scatter_path)
        residual_img = Image.open(residual_path)

        # Ensure all images have the same height
        common_height = min(scatter_img.height, residual_img.height)
        scatter_img = scatter_img.resize((scatter_img.width, common_height))
        residual_img = residual_img.resize((residual_img.width, common_height))
    

        # Create a single row combining all three images
        row_img = Image.new('RGB', (scatter_img.width + residual_img.width , common_height))
        row_img.paste(scatter_img, (0, 0))
        row_img.paste(residual_img, (scatter_img.width, 0))

        plot_rows.append(row_img)

# Combine all rows vertically
total_width = plot_rows[0].width
total_height = sum(img.height for img in plot_rows)

overview_image = Image.new('RGB', (total_width, total_height))

# Paste each row into the final overview image
current_y = 0
for row in plot_rows:
    overview_image.paste(row, (0, current_y))
    current_y += row.height

# Save the overview image
overview_image_path = plots_dir / 'overview_all_plots.png'
overview_image.save(overview_image_path)

print(f"Overview image saved to: {overview_image_path}")