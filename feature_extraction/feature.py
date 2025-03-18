import pandas as pd
from collections import defaultdict
from helper import unzip_folder, zip_folder
import numpy as np
import shutil
from pathlib import Path

from yolo_features import process_yolo_data
from gaze_features import process_gaze_data
from gaze_yolo_features import process_yolo_gaze_data

# ==========================================================
# Constants and variables
# ==========================================================
# Path to the zip file containing the data
BASE_PATH = Path("C:/Users/bayer/Documents/HCI")
ZIP_FILE_PATH = BASE_PATH /"Data.zip"
OUTPUT_FOLDER = BASE_PATH / "Data/Processed_results"


# Dimensions of the video frame
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
# ==========================================================
# Functions
# ==========================================================

# Main function to handle the workflow
def main(zip_file_path: str, output_folder: Path, frame_width: float, frame_height: float) -> None:
    """
    Main function to extract YOLO features from a specific file within a zipped dataset.

    Args:
    - zip_file_path (str): Path to the zip file containing the data folder.
    - output_folder (Path): Folder where the processed files will be saved (unzipped).
    - frame_width (float): Width of the video frame.
    - frame_height (float): Height of the video frame.
    
    Returns:
    - None: Outputs the features to a CSV file and re-zips the folder.
    """

    extract_folder = Path(zip_file_path).parent 
    unzip_folder(zip_file_path, extract_folder)

    # Find all CSV files in the unzipped output folder and its subfolders
    processed_results_folder = output_folder    
    processed_files = {file.stem: file for file in processed_results_folder.rglob("*.csv")}

    # Match YOLO and gaze files by frame, excluding already processed feature files
    yolo_files = {name: file for name, file in processed_files.items() 
                if "yolo" in name.lower() and not name.endswith("features")}

    gaze_files = {name: file for name, file in processed_files.items() 
                if "gaze" in name.lower()and not name.endswith("features")}


    for name, yolo_file in yolo_files.items():

         # Read YOLO data
        yolo_data = pd.read_csv(yolo_file)
        # Rename the 'class' column to 'cls' in the YOLO dataframe
        yolo_data.rename(columns={'class': 'cls'}, inplace=True)
        
        # Process YOLO features
        yolo_features = process_yolo_data(yolo_data, frame_width, frame_height)

        # Save YOLO features
        base_name = yolo_file.stem
        feature_filename = f"{base_name}_yolo_features.csv"
        relative_folder = yolo_file.parent.relative_to(processed_results_folder)
        output_path = output_folder / relative_folder
        output_path.mkdir(parents=True, exist_ok=True)
        yolo_features.to_csv(output_path / feature_filename, index=False)
        print(f"Saved YOLO features to: {output_path / feature_filename}")

        # Find matching gaze file
        matching_gaze_file = gaze_files.get(name.replace("_yolo", "_gaze"), None)
        if matching_gaze_file:
            gaze_data = pd.read_csv(matching_gaze_file)

            # Process gaze features
            gaze_features = process_gaze_data(gaze_data)
            # Save gaze features
            base_name = matching_gaze_file.stem
            feature_filename = f"{base_name}_gaze_features.csv"
            output_path.mkdir(parents=True, exist_ok=True)
            gaze_features.to_csv(output_path / feature_filename, index=False)
            print(f"Saved Gaze features to: {output_path / feature_filename}")

            # #Process gaze yolo features
            # combined_features = process_yolo_gaze_data(gaze_data, yolo_data, target_classes=['car', 'bicycle', 'person'])
            # # Save combined features
            # base_name = matching_gaze_file.stem
            # feature_filename = f"{base_name}_combi_features.csv"
            # relative_folder = yolo_file.parent.relative_to(processed_results_folder)
            # output_path = output_folder / relative_folder
            # output_path.mkdir(parents=True, exist_ok=True)
            # combined_features.to_csv(output_path / feature_filename, index=False)
            # print(f"Saved combined features to: {output_path / feature_filename}")

    # Re-zip the contents of the extracted folder back to Data.zip
    zip_folder(extract_folder, zip_file_path)
    print("Re-zipped the folder.")
    unzipped_folder = extract_folder / "Data"
    if unzipped_folder.exists() and unzipped_folder.is_dir():
        shutil.rmtree(unzipped_folder)
        print(f"Deleted the folder: {unzipped_folder}")

if __name__ == "__main__":
    main(ZIP_FILE_PATH, OUTPUT_FOLDER, FRAME_WIDTH, FRAME_HEIGHT)


