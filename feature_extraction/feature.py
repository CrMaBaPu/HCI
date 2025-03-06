import pandas as pd
from collections import defaultdict
from helper import unzip_folder, zip_folder
import numpy as np
import shutil
from pathlib import Path

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

# ==========================================================
# Functions
# ==========================================================
# how frequently gaze locations change significantly between consecutive gaze points.
def gaze_change_frequency(gaze_data: pd.DataFrame, threshold: float = 50) -> float:
    """
    Calculate the frequency of significant gaze location changes (i.e., gaze shifts).
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    - threshold (float): Minimum distance (in pixels) between two consecutive gaze points to consider it a change.
    
    Returns:
    - float: Frequency of significant gaze location changes.
    """
    gaze_changes = 0
    previous_gaze = None

    for gaze_row in gaze_data.itertuples(index=False):
        current_gaze = (gaze_row.PixelX, gaze_row.PixelY)
        
        if previous_gaze is not None:
            distance = np.sqrt((current_gaze[0] - previous_gaze[0])**2 + (current_gaze[1] - previous_gaze[1])**2)
            if distance >= threshold:
                gaze_changes += 1
        
        previous_gaze = current_gaze
    
    # Return the frequency of gaze changes (scaled by the number of gaze points)
    return gaze_changes / len(gaze_data)
# average duration (in frames) that the gaze stays in a particular location.
def average_gaze_duration(gaze_data: pd.DataFrame, fixation_threshold: float = 10) -> float:
    """
    Calculate the average gaze duration (i.e., how long the gaze stays on a particular point).
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    - fixation_threshold (int): The minimum number of frames required to consider a gaze as a fixation.
    
    Returns:
    - float: The average gaze duration (in frames).
    """
    gaze_durations = []
    current_fixation = None

    for gaze_row in gaze_data.itertuples(index=False):
        current_gaze = (gaze_row.PixelX, gaze_row.PixelY)
        
        if current_fixation is None:
            current_fixation = (current_gaze, 1)
        elif current_gaze == current_fixation[0]:
            current_fixation = (current_gaze, current_fixation[1] + 1)
        else:
            if current_fixation[1] >= fixation_threshold:
                gaze_durations.append(current_fixation[1])
            current_fixation = (current_gaze, 1)
    
    if current_fixation and current_fixation[1] >= fixation_threshold:
        gaze_durations.append(current_fixation[1])
    
    if gaze_durations:
        return np.mean(gaze_durations)
    else:
        return 0.0
    
# average Euclidean distance between consecutive gaze points.   
def distance_between_consecutive_gaze_points(gaze_data: pd.DataFrame) -> float:
    """
    Calculate the average distance between consecutive gaze points.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    
    Returns:
    - float: The average distance between consecutive gaze points.
    """
    distances = []
    previous_gaze = None

    for gaze_row in gaze_data.itertuples(index=False):
        current_gaze = (gaze_row.PixelX, gaze_row.PixelY)
        
        if previous_gaze is not None:
            distance = np.sqrt((current_gaze[0] - previous_gaze[0])**2 + (current_gaze[1] - previous_gaze[1])**2)
            distances.append(distance)
        
        previous_gaze = current_gaze
    
    if distances:
        return np.mean(distances)
    else:
        return 0.0

# counts how many fixations and average duration for each object class.
def fixation_count_per_object_class(gaze_data: pd.DataFrame, yolo_data: pd.DataFrame, target_classes: list, threshold: float = 10) -> pd.DataFrame:
    """
    Count how many fixations and average duration for each object class detected by YOLO.
    
    Args:
    - gaze_data (pd.DataFrame): Gaze data with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    - yolo_data (pd.DataFrame): YOLO data with columns: ['frame', 'cls', 'x_min', 'y_min', 'x_max', 'y_max'].
    - target_classes (list): List of target object classes to track fixations for.
    - threshold (int): Number of frames to consider for a gaze fixation.
    
    Returns:
    - pd.DataFrame: A DataFrame containing fixation count and average duration per target class.
    """
    fixation_counts = defaultdict(int)
    fixation_durations = defaultdict(list)
    current_fixation = None

    for gaze_row in gaze_data.itertuples(index=False):
        frame = gaze_row.VideoFrame
        gaze_x, gaze_y = gaze_row.PixelX, gaze_row.PixelY

        # Get object classes intersecting with gaze point
        object_classes = get_intersecting_object_classes(gaze_x, gaze_y, frame, yolo_data)

        if current_fixation is None:
            current_fixation = (gaze_x, gaze_y, 1, object_classes)
        elif (gaze_x, gaze_y) == (current_fixation[0], current_fixation[1]):
            current_fixation = (gaze_x, gaze_y, current_fixation[2] + 1, object_classes)
        else:
            if current_fixation[2] >= threshold:
                for cls in current_fixation[3]:
                    fixation_counts[cls] += 1
                    fixation_durations[cls].append(current_fixation[2])
            current_fixation = (gaze_x, gaze_y, 1, object_classes)

    if current_fixation and current_fixation[2] >= threshold:
        for cls in current_fixation[3]:
            fixation_counts[cls] += 1
            fixation_durations[cls].append(current_fixation[2])

    # Prepare the output DataFrame
    fixation_data = []
    for cls in target_classes:
        num_fixations = fixation_counts.get(cls, 0)
        avg_duration = np.mean(fixation_durations.get(cls, [])) if fixation_durations.get(cls) else 0.0
        fixation_data.append({'class': cls, 'fixation_count': num_fixations, 'average_duration': avg_duration})

    return pd.DataFrame(fixation_data)

#how much time the gaze spends inside the bounding box of each object class.
def time_spent_in_each_bbox(gaze_data: pd.DataFrame, yolo_data: pd.DataFrame) -> dict:
    """
    Calculate how much time the gaze spends in each object class's bounding box.
    
    Args:
    - gaze_data (pd.DataFrame): Gaze data with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    - yolo_data (pd.DataFrame): YOLO detections with columns: ['frame', 'cls', 'x_min', 'y_min', 'x_max', 'y_max'].
    
    Returns:
    - dict: A dictionary with object class labels as keys and time spent (in frames) as values.
    """
    time_spent = defaultdict(int)
    
    for gaze_row in gaze_data.itertuples(index=False):
        frame = gaze_row.VideoFrame
        gaze_x, gaze_y = gaze_row.PixelX, gaze_row.PixelY
        
        # Get the object classes the gaze point intersects with
        object_classes = get_intersecting_object_classes(gaze_x, gaze_y, frame, yolo_data)
        
        for cls in object_classes:
            time_spent[cls] += 1
    
    return time_spent


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
# Function to get the object classes that a gaze point intersects with
def get_intersecting_object_classes(gaze_x: float, gaze_y: float, frame: int, yolo_data: pd.DataFrame) -> list:
    """
    This function checks which object classes the gaze point intersects with based on YOLO detections.

    Args:
    - gaze_x (float): Gaze X-coordinate.
    - gaze_y (float): Gaze Y-coordinate.
    - frame (int): Frame number to check for object detections.
    - yolo_data (pd.DataFrame): YOLO detections for all frames with columns: ['frame', 'cls', 'x_min', 'y_min', 'x_max', 'y_max'].

    Returns:
    - list: A list of object classes the gaze point intersects with.
    """
    intersecting_classes = []

    # Flip the gaze Y-coordinate to match YOLO coordinate system
    gaze_y_flipped = FRAME_HEIGHT - gaze_y

    # Get all the detections for the current frame
    frame_bboxes = yolo_data[yolo_data['frame'] == frame]

    # Iterate over all YOLO detections in this frame
    for bbox_row in frame_bboxes.itertuples(index=False):
        if is_gaze_inside_bbox(gaze_x, gaze_y_flipped, (bbox_row.x_min, bbox_row.y_min, bbox_row.x_max, bbox_row.y_max)):
            intersecting_classes.append(bbox_row.cls)

    return intersecting_classes

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
        yolo_features_df = pd.DataFrame(frame_features)

        # Aggregate the features to have one row per file
        aggregated_features = {
            'total_objects_all': yolo_features_df['total_objects_all'].sum(),  # Sum makes sense for total counts
            'total_classes': yolo_features_df['total_classes'].nunique(),  # Unique count of different detected classes
            'cluster_std_dev': yolo_features_df['cluster_std_dev'].mean(),  # Mean makes sense for variability
            'central_detection_size': yolo_features_df['central_detection_size'].mean(),  # Mean for central object size
        }

        # Aggregate per class
        target_classes = ['car', 'bicycle', 'person']
        for cls in target_classes:
            aggregated_features[f'num_per_cls{cls}'] = yolo_features_df[f'num_per_cls{cls}'].sum()  # Sum makes sense
            aggregated_features[f'average_size_{cls}'] = yolo_features_df[f'average_size_{cls}'].mean()  # Mean makes sense

        return pd.DataFrame([aggregated_features])


# # Function to process gaze data alongside YOLO data
def process_gaze_data(gaze_data: pd.DataFrame, yolo_data: pd.DataFrame, target_classes: list) -> pd.DataFrame:
    """
    Process gaze data to calculate the relevant features for specific object classes.

    Args:
    - gaze_data (pd.DataFrame): Gaze data with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    - yolo_data (pd.DataFrame): YOLO detections with columns: ['frame', 'cls', 'x_min', 'y_min', 'x_max', 'y_max'].
    - target_classes (list): List of target object classes to track.

    Returns:
    - pd.DataFrame: A DataFrame containing calculated gaze features for each target class.
    """
    # Feature 1: Gaze Change Frequency
    gaze_change_freq = gaze_change_frequency(gaze_data)
    
    # Feature 2: Average Gaze Duration
    avg_gaze_duration = average_gaze_duration(gaze_data)
    
    # Feature 3: Distance Between Consecutive Gaze Points
    gaze_dispersion = distance_between_consecutive_gaze_points(gaze_data)
    
    # Feature 4: Fixation Count and Duration per Object Class
    fixation_data = fixation_count_per_object_class(gaze_data, yolo_data, target_classes)
    
    # Feature 5: Time Spent in Each Object's Bounding Box
    time_spent_in_bbox = time_spent_in_each_bbox(gaze_data, yolo_data)

    # Prepare final features
    features = {
        'gaze_change_frequency': gaze_change_freq,
        'average_gaze_duration': avg_gaze_duration,
        'gaze_dispersion': gaze_dispersion,
    }

    # Extract time spent in relevant classes (car, person, bicycle)
    features['time_spent_in_car'] = time_spent_in_bbox.get('car', 0)
    features['time_spent_in_person'] = time_spent_in_bbox.get('person', 0)
    features['time_spent_in_bicycle'] = time_spent_in_bbox.get('bicycle', 0)

    # Extract fixation count for relevant classes (car, person, bicycle)
    features['fixation_count_car'] = fixation_data[fixation_data['class'] == 'car']['fixation_count'].values[0] if 'car' in fixation_data['class'].values else 0
    features['fixation_count_person'] = fixation_data[fixation_data['class'] == 'person']['fixation_count'].values[0] if 'person' in fixation_data['class'].values else 0
    features['fixation_count_bicycle'] = fixation_data[fixation_data['class'] == 'bicycle']['fixation_count'].values[0] if 'bicycle' in fixation_data['class'].values else 0

    # Merge fixation data for target classes into the features
    features_df = pd.DataFrame([features])
    
    return features_df



# ==========================================================
# Main
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
            gaze_features = process_gaze_data(gaze_data, yolo_data, target_classes = ['car', 'bicycle', 'person'])

            # Save gaze features
            base_name = matching_gaze_file.stem
            feature_filename = f"{base_name}_gaze_features.csv"
            output_path.mkdir(parents=True, exist_ok=True)
            gaze_features.to_csv(output_path / feature_filename, index=False)
            print(f"Saved Gaze features to: {output_path / feature_filename}")

    # Re-zip the contents of the extracted folder back to Data.zip
    zip_folder(extract_folder, zip_file_path)
    print("Re-zipped the folder.")
    unzipped_folder = extract_folder / "Data"
    if unzipped_folder.exists() and unzipped_folder.is_dir():
        shutil.rmtree(unzipped_folder)
        print(f"Deleted the folder: {unzipped_folder}")

if __name__ == "__main__":
    main(ZIP_FILE_PATH, OUTPUT_FOLDER, FRAME_WIDTH, FRAME_HEIGHT)


