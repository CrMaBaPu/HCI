# HCI

# HCI Project

## Overview
This project involves segmentation, preprocessing, feature extraction, and regression modeling of YOLO, gaze, and video data.

## Pipeline

### 1. Data Processing
Run the following script to segment and preprocess YOLO, gaze, and video data:

```bash
python data_processing/main.py
```

### 2. Feature Extraction
Extract relevant features and perform an initial analysis:

- **Feature Extraction:**  
  ```bash
  python feature_extraction/feature.py
  ```
  This script generates `yolo_features` and `gaze_features` for each file.

- **Descriptive Analysis:**  
  ```bash
  python feature_extraction/descriptive_analyses.py
  ```
  This provides an overview of the extracted data.

### 3. Model Preparation
Combine extracted features with criticality scores:

```bash
python model/join_features.py
```
This script merges `yolo_features`, `gaze_features`, and the criticality score into a single `features_dataset`.

### 4. Model Training
Train the regression model using the prepared dataset:

```bash
python model/training.py
```


## Authors
- Cristina Bayer



