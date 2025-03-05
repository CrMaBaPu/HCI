import pandas as pd
from collections import defaultdict
from helper import unzip_folder, zip_folder
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

# Function to check if a gaze point is inside a bounding box
def is_gaze_inside_bbox(gaze_x: float, gaze_y: float, bbox: tuple) -> bool:
    """
    Check if the gaze coordinates fall inside a given bounding box.
    
    Args:
    - gaze_x (float): Gaze X coordinate.
    - gaze_y (float): Gaze Y coordinate.
    - bbox (tuple): Bounding box in the format (x_min, y_min, x_max, y_max).
    
    Returns:
    - bool: True if the gaze point is inside the bounding box, otherwise False.
    """
    x_min, y_min, x_max, y_max = bbox
    return x_min <= gaze_x <= x_max and y_min <= gaze_y <= y_max

# Function to map gaze points to object classes based on bounding boxes
def match_gaze_to_objects(gaze_data: pd.DataFrame, yolo_data: pd.DataFrame) -> dict:
    """
    Matches gaze points to detected objects based on bounding boxes.

    Args:
    - gaze_data (pd.DataFrame): Gaze data with columns ['VideoFrame', 'PixelX', 'PixelY'].
    - yolo_data (pd.DataFrame): YOLO detections with ['frame', 'class', 'x_min', 'y_min', 'x_max', 'y_max'].
    
    Returns:
    - dict: A dictionary mapping frame numbers to a list of object classes the gaze fell upon.
    """
    gaze_to_objects = defaultdict(list)

    for gaze_row in gaze_data.itertuples(index=False):
        frame, gaze_x, gaze_y = int(gaze_row.VideoFrame), gaze_row.PixelX, gaze_row.PixelY

        if frame in yolo_data['frame'].values:
            frame_bboxes = yolo_data[yolo_data['frame'] == frame]

            for bbox_row in frame_bboxes.itertuples(index=False):
                if is_gaze_inside_bbox(gaze_x, gaze_y, (bbox_row.x_min, bbox_row.y_min, bbox_row.x_max, bbox_row.y_max)):
                    gaze_to_objects[frame].append(bbox_row.cls)

    return gaze_to_objects

# Function to compute gaze-based statistics
def compute_gaze_statistics(gaze_to_objects: dict) -> dict:
    """
    Computes statistics on gaze fixations.

    Args:
    - gaze_to_objects (dict): Dictionary where keys are frames and values are lists of detected object classes.
    
    Returns:
    - dict: Contains statistics including average fixation time, most viewed class, 
            and most frequently viewed class.
    """
    gaze_duration_per_class = defaultdict(float)
    gaze_count_per_class = defaultdict(int)

    for frame, classes in gaze_to_objects.items():
        unique_classes = set(classes)  # Avoid counting multiple detections of the same object per frame
        for cls in unique_classes:
            gaze_duration_per_class[cls] += 1
            gaze_count_per_class[cls] += 1

    avg_fixation_time = {cls: duration / gaze_count_per_class[cls] for cls, duration in gaze_duration_per_class.items()}
    most_viewed_class = max(gaze_duration_per_class, key=gaze_duration_per_class.get, default=None)
    most_frequent_class = max(gaze_count_per_class, key=gaze_count_per_class.get, default=None)

    return {
        "average_fixation_time": avg_fixation_time,
        "most_viewed_class": most_viewed_class,
        "most_frequent_class": most_frequent_class,
        "gaze_duration_per_class": gaze_duration_per_class,
        "gaze_count_per_class": gaze_count_per_class,
    }


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
        clscounts = defaultdict(int)
        for _, cls, _, _, _, _, _ in bboxes:  # renamed `class` to `cls`
            clscounts[cls] += 1
        
        features = {
            'frame': frame,
            'total_objects_all': len(bboxes),
            'total_classes': len(set([d[1] for d in bboxes]))  # Access class via index
        }
        
        for cls in target_classes:  # renamed `class` to `cls`
            features[f'num_per_cls{cls}'] = clscounts.get(cls, 0)
            clsbboxes = [bbox for bbox in bboxes if bbox[1] == cls]  # Access class via index
            features[f'average_size_{cls}'] = calculate_average_size(clsbboxes)
        
        features['cluster_std_dev'] = calculate_cluster_std_from_center(bboxes, frame_width, frame_height)
        features['central_detection_size'] = calculate_central_detection_size(bboxes, frame_width, frame_height)
        frame_features.append(features)

    return pd.DataFrame(frame_features)

# Function to process gaze data alongside YOLO data
def process_gaze_data(gaze_data: pd.DataFrame, yolo_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process gaze data to calculate gaze-related features for each frame.

    Args:
    - gaze_data (pd.DataFrame): A DataFrame containing gaze data with columns: VideoFrame, PixelX, PixelY.
    - yolo_data (pd.DataFrame): A DataFrame containing YOLO detection data with columns: frame, class, confidence, x_min, y_min, x_max, y_max.

    Returns:
    - pd.DataFrame: A DataFrame where each row corresponds to a frame, containing gaze-related features.
    """
    # Initialize the list to hold features for each frame
    frame_features = []

    # Match gaze data to YOLO detections by frame
    gaze_to_objects = match_gaze_to_objects(gaze_data, yolo_data)

    # Calculate statistics on the matched gaze-to-objects mapping
    gaze_stats = compute_gaze_statistics(gaze_to_objects)

    # Loop through each frame in the gaze data (unique frames)
    for frame in gaze_data['VideoFrame'].unique():
        # Count the total number of gaze points for this frame
        total_gaze_points = len(gaze_data[gaze_data['VideoFrame'] == frame])
        
        # Prepare the features dictionary for this frame
        gaze_features = {
            'frame': frame,  # Ensure the frame column is named 'frame'
            'total_gaze_points': total_gaze_points
        }

        # Add gaze statistics to the features dictionary
        for key, value in gaze_stats.items():
            gaze_features[key] = value

        # Append the features for this frame to the list
        frame_features.append(gaze_features)

    # Convert the frame_features list to a DataFrame
    gaze_features_df = pd.DataFrame(frame_features)

    # Ensure the column is named 'frame' for consistency with YOLO data
    gaze_features_df.rename(columns={'VideoFrame': 'frame'}, inplace=True)

    return gaze_features_df

# ==========================================================
# Main
# ==========================================================

# Main function to handle the workflow
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
    # Locate the specific `Processed_results` folder
    processed_results_folder = output_folder

    if not processed_results_folder.exists():
        print(f"Error: '{processed_results_folder}' does not exist.")
        return
    
    processed_files = {file.stem: file for file in processed_results_folder.rglob("*.csv")}

    # Match YOLO and gaze files by frame, excluding already processed feature files
    yolo_files = {name: file for name, file in processed_files.items() 
                if "yolo" in name.lower() and not name.endswith("features")}

    gaze_files = {name: file for name, file in processed_files.items() 
                if "gaze" in name.lower()}
    
        # Print the lists of YOLO and gaze files found
    print(f"Found {len(yolo_files)} YOLO files:")
    for name, file in yolo_files.items():
        print(f"  YOLO file: {file}")

    print(f"Found {len(gaze_files)} Gaze files:")
    for name, file in gaze_files.items():
        print(f"  Gaze file: {file}")


    for name, yolo_file in yolo_files.items():
        #Read data
        yolo_data = pd.read_csv(yolo_file)
        # Rename the 'class' column to 'cls' in the YOLO dataframe
        yolo_data.rename(columns={'class': 'cls'}, inplace=True)

        # Find matching gaze file
        matching_gaze_file = gaze_files.get(name.replace("_yolo", "_gaze"), None)
        gaze_data = pd.read_csv(matching_gaze_file)

        # Calculate features
        yolo_features = process_yolo_data(yolo_data, frame_width, frame_height)
        gaze_features = process_gaze_data(gaze_data, yolo_data)
        # Merge features on 'frame'
        combined_features = pd.merge(yolo_features, gaze_features, on="frame", how="left").fillna(0)

        # Generate output file name
        base_name = yolo_file.stem.replace("_yolo_", "_features_")
        feature_filename = f"{base_name}.csv"
        # Derive the output folder structure
        relative_folder = yolo_file.parent.relative_to(processed_results_folder)
        output_path = output_folder / relative_folder
        output_path.mkdir(parents=True, exist_ok=True)

        # Save the merged features DataFrame
        combined_features.to_csv(output_path / feature_filename, index=False)
        print(f"Saved combined features to: {output_path / feature_filename}")

    # Re-zip the contents of the extracted folder back to Data.zip
    zip_folder(extract_folder, zip_file_path)
    print("Re-zipped the folder.")
    unzipped_folder = extract_folder / "Data"
    if unzipped_folder.exists() and unzipped_folder.is_dir():
        shutil.rmtree(unzipped_folder)
        print(f"Deleted the folder: {unzipped_folder}")

if __name__ == "__main__":
    main(ZIP_FILE_PATH, OUTPUT_FOLDER, FRAME_WIDTH, FRAME_HEIGHT)


