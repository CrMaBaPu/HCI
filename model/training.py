import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the saved extracted features
features_df = pd.read_csv('features_dataset.csv')

# Map labels to integers (e.g., low=0, mid=1, high=2)
label_mapping = {'low': 0, 'mid': 1, 'high': 2}
features_df['label'] = features_df['label'].map(label_mapping)

# Separate features and labels
X = features_df.drop('label', axis=1)
y = features_df['label']

# Handle missing values (filling NaN values with the mean of the column)
X.fillna(X.mean(), inplace=True)

# Preprocess the data (scaling features)
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize the Random Forest model with class weight balancing
rf_model = RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=42, class_weight='balanced')

# Train the model
print("Training the model with class weight balancing...")
rf_model.fit(X_train, y_train)

# Make predictions on the test set
print("Making predictions on test set...")
y_pred = rf_model.predict(X_test)

# Evaluate the model
print("Evaluating the model...")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_mapping.keys()))

# Confusion matrix
print("Generating normalized confusion matrix...")
cm = confusion_matrix(y_test, y_pred, normalize='true')  # Normalize by row
labels_unique = list(label_mapping.keys())  # Use the label names for the heatmap

# Visualization of Normalized Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels_unique, yticklabels=labels_unique)
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Feature importance (to understand which features are most important)
feature_importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features_df.columns[:-1],  # Exclude the 'label' column
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print("Feature Importances:\n", importance_df)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance from Random Forest')
plt.show()
