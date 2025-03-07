import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import shutil
from helper import zip_folder, unzip_folder

# ===================================================
# Constants and Variables
# ===================================================
BASE_PATH = Path("C:/Users/bayer/Documents/HCI")
ZIP_FILE_PATH = BASE_PATH / "Data.zip"
OUTPUT_FOLDER = BASE_PATH / "Data/Processed_results"

# ===================================================
# Functions
# ===================================================
def create_features_dataset(processed_results_folder: Path) -> pd.DataFrame:
    """
    This function will extract YOLO and gaze features from their respective CSV files,
    match them by frame, and combine them side by side into a single DataFrame.
    """
    # Get all processed files in the folder (excluding already processed feature files)
    print(processed_results_folder)
    processed_files = {file.stem: file for file in processed_results_folder.rglob("*features.csv")}
    print(processed_files)
    if not processed_files:
        print("No processed files found in the directory.")
        return pd.DataFrame()  # Return empty DataFrame if no files are found

    # Get YOLO and gaze feature files
    yolo_files = {name: file for name, file in processed_files.items() 
            if "yolo" in name.lower()}
    
    gaze_files = {name: file for name, file in processed_files.items() 
              if "gaze" in name.lower()}

    print(f"Found {len(yolo_files)} YOLO files and {len(gaze_files)} gaze files.")  # Debug line

    if not yolo_files or not gaze_files:
        print("No YOLO or gaze feature files found.")
        return pd.DataFrame()  # Return empty DataFrame if no feature files are found

    # Initialize an empty list to store the combined rows
    combined_features = []

    # Loop through each YOLO file
    for name, yolo_file in yolo_files.items():
        # Try to match the corresponding gaze file
        matching_gaze_file = gaze_files.get(name.replace("yolo", "gaze"), None)
        
        if matching_gaze_file:
            # Read both YOLO and gaze files
            yolo_df = pd.read_csv(yolo_file)
            gaze_df = pd.read_csv(matching_gaze_file)

            # Check if they have the same number of rows
            if yolo_df.shape[0] != gaze_df.shape[0]:
                print(f"Mismatch in row count between {yolo_file} and {matching_gaze_file}. Skipping.")
                continue  # Skip this pair if row counts do not match

            # Concatenate YOLO and gaze features side by side
            combined_row = pd.concat([yolo_df, gaze_df], axis=1)
            combined_features.append(combined_row)
        else:
            print(f"No matching gaze file found for {yolo_file}")

    # Check if combined_features is empty before concatenating
    if not combined_features:
        print("No valid feature pairs were found.")
        return pd.DataFrame()  # Return empty DataFrame if no valid rows are combined

    print(f"Combining {len(combined_features)} feature pairs.")  # Debug line

    # Create a DataFrame from the combined features
    final_df = pd.concat(combined_features, ignore_index=True)
    
    return final_df


# Main function to handle the workflow
def main(zip_file_path: str, output_folder: Path) -> None:
    """
    Main function to extract YOLO features and build the dataset.
    """
    extract_folder = Path(zip_file_path).parent
    unzip_folder(zip_file_path, extract_folder)

    # Locate the `Processed_results` folder
    processed_results_folder = output_folder

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
