# HCI

# HCI Project

## Overview
This project involves segmentation, preprocessing, feature extraction, and regression modeling of YOLO, gaze, and video data.

## Pipeline

### 1. Data Processing
Run the following script to segment and preprocess YOLO, gaze, and video data:

```bash
python data_processing/processing_pipeine.py
```

### 2. Feature Extraction
Extract relevant features

  ```bash
  python feature_extraction/feature_extraction_pipeline.py
  ```
  This script generates `yolo_features`, `gaze_features` and `interactive_features`  for each file.

### 3. Model Preparation
Combine extracted features with criticality scores:

```bash
python model/join_features.py
```
This script merges `yolo_features`, `gaze_features`, `interactive_features` and the criticality score into a single `features_dataset`.

### 4. Model Training
Train the regression model using the prepared dataset:

For the model selection based on hyperparameter tuning
```bash
python model/model_tuning.py
```
For overfitting countermeasures
```bash
python model/model_overfitting.py
```
For model training and interpreation:
```bash
python model/model_validation.py
```
Respective plots are saved in the plots folder with the same name. 

## Authors
- Cristina Bayerd



