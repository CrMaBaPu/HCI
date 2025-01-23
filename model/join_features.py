import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import shutil
from helper import zip_folder, unzip_folder

# ===================================================
# Constants and Variables
# ===================================================

yolo_features = [
        'frame', 'total_objects_all', 'total_classes',
        'num_per_class_car', 'average_size_car',
        'num_per_class_bicycle', 'average_size_bicycle',
        'num_per_class_pedestrian', 'average_size_pedestrian',
        'cluster_std_dev', 'central_detection_size'
    ]

# Path to the zip file containing the data
BASE_PATH = Path("C:/Users/bayer/Documents/HCI")
ZIP_FILE_PATH = BASE_PATH /"Data.zip"
OUTPUT_FOLDER = BASE_PATH / "Data/Processed_results"
# ===================================================
# Functions
# ===================================================


# Helper function: Encode labels into low, mid, high
def encode_label(value):
    if value <= 33:
        return "low"
    elif value <= 66:
        return "mid"
    else:
        return "high"

# Function to process YOLO features data (from *_yolo_*_features.csv)
def process_yolo_data(yolo_file: Path) -> pd.DataFrame:
    """
    Extract the required features from the YOLO file, only retaining the last row.
    """
    yolo_data = pd.read_csv(yolo_file)
    required_columns = yolo_features
    
    if not set(required_columns).issubset(yolo_data.columns):
        raise ValueError(f"YOLO file {yolo_file} is missing required feature columns.")
    
    # Return only the last row with required columns
    return yolo_data[required_columns].iloc[[-1]]


# Function to process gaze data
def process_gaze_data(gaze_file: Path) -> str:
    """
    Extract the last label (ArduinoData1) and encode it into a category.
    """
    gaze_data = pd.read_csv(gaze_file)
    if 'ArduinoData1' not in gaze_data.columns:
        raise ValueError(f"Gaze file {gaze_file} is missing 'ArduinoData1' column.")
    
    # Get the last value in the 'ArduinoData1' column
    last_label = gaze_data['ArduinoData1'].iloc[-1]
    return encode_label(last_label)

# Main function to create the dataset
def create_features_dataset(processed_results_folder: Path) -> pd.DataFrame:
    """
    Process all *_yolo_*_features.csv and corresponding gaze files to generate the features dataset.
    """
    processed_files = {file.stem: file for file in processed_results_folder.rglob("*.csv")}
    dataset = []

    for file_stem, yolo_file in processed_files.items():
        # Only process *_yolo_*_features.csv files
        if '_yolo_' not in file_stem or '_features' not in file_stem:
            continue
        
        # Construct the expected gaze file path based on the YOLO file's directory
        yolo_dir = yolo_file.parent
        gaze_file_stem = file_stem.replace('_yolo_', '_gaze_').replace('_features', '')
        gaze_file = yolo_dir / f"{gaze_file_stem}.csv"
        
        if not gaze_file.exists():
            print(f"Skipping {yolo_file}: Corresponding gaze file not found.")
            continue
        
        try:
            # Process YOLO file
            yolo_features = process_yolo_data(yolo_file)
            
            # Process gaze file and get the encoded label
            label = process_gaze_data(gaze_file)
            
            # Combine features and label
            yolo_features['label'] = label
            dataset.append(yolo_features)
        
        except ValueError as e:
            print(f"Error processing files: {e}")
            continue

    # Combine all data into a single DataFrame
    if dataset:
        features_df = pd.concat(dataset, ignore_index=True)
    else:
        features_df = pd.DataFrame()
    
    return features_df

# Main function to handle the workflow
def main(zip_file_path: str, output_folder: Path) -> None:
    """
    Main function to extract YOLO features and build the dataset.
    """
    extract_folder = Path(zip_file_path).parent
    unzip_folder(zip_file_path, extract_folder)

    # Locate the `Processed_results` folder
    processed_results_folder = output_folder
    if not processed_results_folder.exists():
        print(f"Error: '{processed_results_folder}' does not exist.")
        return
    
    # Create the features dataset
    features_df = create_features_dataset(processed_results_folder)
    
    # Save the dataset
    features_df_path = BASE_PATH / "model/features_dataset.csv"
    features_df.to_csv(features_df_path, index=False)
    print(f"Features dataset saved to: {features_df_path}")

    # Re-zip the folder
    zip_folder(extract_folder, zip_file_path)
    print("Re-zipped the folder.")
    unzipped_folder = extract_folder / "Data"
    if unzipped_folder.exists() and unzipped_folder.is_dir():
        shutil.rmtree(unzipped_folder)
        print(f"Deleted the folder: {unzipped_folder}")

if __name__ == "__main__":
    main(ZIP_FILE_PATH, OUTPUT_FOLDER)