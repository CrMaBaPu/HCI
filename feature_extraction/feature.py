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

        # Flip the gaze Y-coordinate to match the YOLO coordinate system
        gaze_y_flipped = FRAME_HEIGHT - gaze_y

        if frame in yolo_data['frame'].values:
            frame_bboxes = yolo_data[yolo_data['frame'] == frame]

            for bbox_row in frame_bboxes.itertuples(index=False):
                if is_gaze_inside_bbox(gaze_x, gaze_y_flipped, (bbox_row.x_min, bbox_row.y_min, bbox_row.x_max, bbox_row.y_max)):
                    gaze_to_objects[frame].append(bbox_row.cls)

    return gaze_to_objects


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

    target_classes = ['car', 'bicycle', 'person']
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
    Process gaze data to calculate gaze-related features for each frame, 
    including gaze shifts, duration counters, and class-specific gaze tracking.
    
    Args:
    - gaze_data (pd.DataFrame): Gaze data with ['VideoFrame', 'PixelX', 'PixelY'].
    - yolo_data (pd.DataFrame): YOLO detections with ['frame', 'cls', 'x_min', 'y_min', 'x_max', 'y_max'].

    Returns:
    - pd.DataFrame: DataFrame with features for each frame, merged with YOLO.
    """
    # Sort gaze data by frame
    gaze_data = gaze_data.sort_values(by="VideoFrame").reset_index(drop=True)

    # Trackers
    shift_counter = 0
    duration_counter = 0
    prev_objects = set()
    
    # Track class-specific features
    target_classes = ['car', 'bicycle', 'person']
    class_duration = {cls: 0 for cls in target_classes}
    class_shift_count = {cls: 0 for cls in target_classes}
    
    frame_features = []

    for frame in gaze_data['VideoFrame'].unique():
        # Get gaze point for this frame
        gaze_row = gaze_data[gaze_data['VideoFrame'] == frame].iloc[0]
        gaze_x, gaze_y = gaze_row.PixelX, gaze_row.PixelY
        
        # Find objects looked at in this frame
        objects_looked_at = set()
        if frame in yolo_data['frame'].values:
            frame_bboxes = yolo_data[yolo_data['frame'] == frame]
            for bbox_row in frame_bboxes.itertuples(index=False):
                if is_gaze_inside_bbox(gaze_x, gaze_y, (bbox_row.x_min, bbox_row.y_min, bbox_row.x_max, bbox_row.y_max)):
                    objects_looked_at.add(bbox_row.cls)
        
        # Detect shift (if gaze moves to different objects)
        if objects_looked_at != prev_objects:
            shift_counter = 1  # Mark shift event
            duration_counter = 1  # Reset duration counter
        else:
            shift_counter = 0
            duration_counter += 1  # Continue counting duration
        
        # Update class-specific counters
        for cls in target_classes:
            if cls in objects_looked_at:
                class_duration[cls] += 1  # Increase duration counter for this class
                if cls not in prev_objects:
                    class_shift_count[cls] += 1  # Increase shift count when first looking at this class
            else:
                class_duration[cls] = 0  # Reset duration counter if not being looked at
        
        # Store frame features
        frame_features.append({
            "frame": frame,
            "shift": shift_counter,
            "duration": duration_counter,
            "objects_looked_at": list(objects_looked_at),  # Can be used later
            **{f"{cls}_duration": class_duration[cls] for cls in target_classes},
            **{f"{cls}_shift_count": class_shift_count[cls] for cls in target_classes}
        })

        # Update previous objects looked at
        prev_objects = objects_looked_at

    return pd.DataFrame(frame_features)


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


