import pandas as pd
from collections import defaultdict
import numpy as np
import shutil
from pathlib import Path
import zipfile
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

# Function to calculate the average size of bounding boxes for a given class.
def calculate_average_size(bboxes: list) -> float:
    """
    Calculate the average size (area) of the bounding boxes for a given class.

    Args:
    - bboxes (list): A list of bounding boxes for a specific class in the format
      (frame, x_min, y_min, x_max, y_max, class).

    Returns:
    - float: The average size (area) of the bounding boxes, or 0 if no boxes are present.
    """
    if len(bboxes) == 0:
        return 0.0
    
    total_area = 0.0
    for _, _, _, x_min, y_min, x_max, y_max in bboxes:
        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        total_area += area
    
    return total_area / len(bboxes)


# Function to calculate the standard deviation of bounding box centroids from the frame center
def calculate_cluster_std_from_center(bboxes: list, frame_width: float, frame_height: float) -> float:
    """
    Calculate the standard deviation of bounding box centroids from the center of the frame.

    Args:
    - bboxes (list): A list of bounding boxes in the format 
      (frame, x_min, y_min, x_max, y_max, class).
    - frame_width (float): The width of the frame.
    - frame_height (float): The height of the frame.

    Returns:
    - float: The standard deviation of centroids' distances from the frame center.
    """
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    
    distances = []
    for _, _, _, x_min, y_min, x_max, y_max in bboxes:
        centroid_x = (x_min + x_max) / 2
        centroid_y = (y_min + y_max) / 2
        distance = np.sqrt((centroid_x - frame_center_x) ** 2 + (centroid_y - frame_center_y) ** 2)
        distances.append(distance)
    
    if distances:
        return np.std(distances)
    else:
        return 0.0


# Function to calculate the size of the most central detection based on its proximity to the center.
def calculate_central_detection_size(bboxes: list, frame_width: float, frame_height: float) -> float:
    """
    Calculate the size (area) of the most central detection by finding the closest detection 
    to the center of the frame.

    Args:
    - bboxes (list): A list of bounding boxes in the format 
      (frame, x_min, y_min, x_max, y_max, class).
    - frame_width (float): The width of the frame.
    - frame_height (float): The height of the frame.

    Returns:
    - float: The size (area) of the most central detection, or 0 if no detections are present.
    """
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    
    closest_detection = None
    closest_distance = float('inf')
    for _, _, _, x_min, y_min, x_max, y_max in bboxes:
        centroid_x = (x_min + x_max) / 2
        centroid_y = (y_min + y_max) / 2
        distance = np.sqrt((centroid_x - frame_center_x) ** 2 + (centroid_y - frame_center_y) ** 2)
        
        if distance < closest_distance:
            closest_distance = distance
            closest_detection = (x_min, y_min, x_max, y_max)
    
    if closest_detection:
        x_min, y_min, x_max, y_max = closest_detection
        width = x_max - x_min
        height = y_max - y_min
        return width * height
    else:
        return 0.0


# Function to process YOLO detection data and calculate features for each frame
def process_yolo_data(yolo_data: pd.DataFrame, frame_width: float, frame_height: float) -> pd.DataFrame:
    """
    Process YOLO detection data to calculate a feature vector for each frame.
    
    Args:
    - yolo_data (pd.DataFrame): A DataFrame containing YOLO detection data 
      with columns: frame, class, confidence, x_min, y_min, x_max, y_max.
    - frame_width (float): The width of the frame (video frame or image).
    - frame_height (float): The height of the frame (video frame or image).

    Returns:
    - pd.DataFrame: A DataFrame where each row corresponds to a frame, 
      containing calculated features such as total objects, 
      class counts, clustering measures, etc.
    """
    frame_features = []
    grouped_by_frame = defaultdict(list)
    for row in yolo_data.itertuples(index=False):
        grouped_by_frame[row.frame].append(row)

    target_classes = ['car', 'bicycle', 'pedestrian']
    for frame, bboxes in grouped_by_frame.items():
        class_counts = defaultdict(int)
        for _, cls, _, _, _, _, _ in bboxes:  # renamed `class` to `cls`
            class_counts[cls] += 1
        
        features = {
            'frame': frame,
            'total_objects_all': len(bboxes),
            'total_classes': len(set([d[1] for d in bboxes]))  # Access class via index
        }
        
        for cls in target_classes:  # renamed `class` to `cls`
            features[f'num_per_class_{cls}'] = class_counts.get(cls, 0)
            class_bboxes = [bbox for bbox in bboxes if bbox[1] == cls]  # Access class via index
            features[f'average_size_{cls}'] = calculate_average_size(class_bboxes)
        
        features['cluster_std_dev'] = calculate_cluster_std_from_center(bboxes, frame_width, frame_height)
        features['central_detection_size'] = calculate_central_detection_size(bboxes, frame_width, frame_height)
        frame_features.append(features)

    return pd.DataFrame(frame_features)


# ==========================================================
# Main
# ==========================================================

# Main function to handle the workflow
def main(zip_file_path: str, target_file_name: str, frame_width: float, frame_height: float) -> None:
    """
    Main function to extract YOLO features from a specific file within a zipped dataset.

    Args:
    - zip_file_path (str): Path to the zip file containing the data folder.
    - target_file_name (str): Name of the file to process within the extracted folder.
    - frame_width (float): Width of the video frame.
    - frame_height (float): Height of the video frame.

    Returns:
    - None: Outputs the features to a CSV file and re-zips the folder.
    """
    extract_folder = Path(zip_file_path).parent 
    unzip_folder(zip_file_path, extract_folder)

    # Find all CSV files in the unzipped output folder and its subfolders
    # Locate the specific `Processed_results` folder
    processed_results_folder = output_folder

    if not processed_results_folder.exists():
        print(f"Error: '{processed_results_folder}' does not exist.")
        return
    
    processed_files = {file.stem: file for file in processed_results_folder.rglob("*.csv")}


    for file in processed_files.values():
        print(f"Processing file: {file}")
        
        # Read YOLO data
        yolo_data = pd.read_csv(file)
        required_columns = {'frame', 'class', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max'}
        if not required_columns.issubset(set(yolo_data.columns)):
            print(f"Skipping {file}: Missing required columns")
            continue
        # Calculate features
        features_df = process_yolo_data(yolo_data, frame_width, frame_height)
        # Derive the output folder structure
        relative_folder = file.parent.relative_to(processed_results_folder)
        output_path = output_folder / relative_folder
        output_path.mkdir(parents=True, exist_ok=True)

        # Save the features DataFrame
        feature_filename = f"{file.stem}_features.csv"
        features_df.to_csv(output_path / feature_filename, index=False)
        print(f"Saved features to: {output_path / feature_filename}")

    # Re-zip the contents of the extracted folder back to Data.zip
    zip_folder(extract_folder, zip_file_path)
    print("Re-zipped the folder.")
if __name__ == "__main__":
    main(ZIP_FILE_PATH, OUTPUT_FOLDER, FRAME_WIDTH, FRAME_HEIGHT)
