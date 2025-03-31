import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# Feature Explanation:
# 1. **most_frequent_class**:
#    - **Description**: This is the object class that was most frequently looked at by the gaze across all frames. It is calculated by counting the number of times each object class is looked at and selecting the class with the highest count.
#    - **Example**: If in the entire video, the gaze looked at "Car" 50 times, "Person" 30 times, and "Bicycle" 20 times, then `most_frequent_class = "Car"`.

# 2. **avg_duration_near_object**:
#    - **Description**: This is the average number of frames where the gaze was near a bounding box of a specific object class. The duration is represented by the **count of frames** where the gaze was close to the object. For each object class, we calculate the total number of frames and divide by the number of occurrences of that class to get the average.
#    - **Example**: If the gaze was near a "Car" bounding box for frames 1, 3, 5, the `avg_duration_near_object` for "Car" would be `3` (since there are 3 frames in which the gaze was near the "Car").

# 3. **avg_distance_to_closest_object**:
#    - **Description**: This is the average distance between the gaze point and the closest bounding box of an object class, across all frames where the gaze was near a detection. The distance is calculated from the gaze point to the closest edge of the bounding box, with the threshold for considering the object "near" being configurable (e.g., 100 pixels).
#    - **Example**: If in three frames, the closest object "Person" had distances of 20, 15, and 10 pixels from the gaze point, the `avg_distance_to_closest_object` for "Person" would be `(20 + 15 + 10) / 3 = 15.0` pixels.

# ==========================================================
# Constants and variables
# ==========================================================
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# ==========================================================
# helper functions
# ==========================================================
def calculate_distance_to_bbox(gaze_x: float, gaze_y: float, bbox: tuple) -> float:
    """
    Calculate the distance from the gaze point to the closest edge of the bounding box.
    
    Args:
    - gaze_x (float): Gaze X coordinate.
    - gaze_y (float): Gaze Y coordinate.
    - bbox (tuple): Bounding box in the format (x_min, y_min, x_max, y_max).
    
    Returns:
    - float: The distance from the gaze point to the closest edge of the bounding box.
    """
    # Adjust gaze_y to match YOLO coordinate system (flip vertically)
    gaze_y = FRAME_HEIGHT - gaze_y
    
    x_min, y_min, x_max, y_max = bbox
    # Calculate the distance to each of the four bounding box edges
    dx = max(x_min - gaze_x, 0, gaze_x - x_max)
    dy = max(y_min - gaze_y, 0, gaze_y - y_max)
    return np.sqrt(dx**2 + dy**2)

def get_closest_detection_and_distance(gaze_x: float, gaze_y: float, frame: int, yolo_data: pd.DataFrame):
    """
    Calculate the closest object class(es) and the distance to the closest bounding box, without any distance threshold.
    
    Args:
    - gaze_x (float): Gaze X coordinate.
    - gaze_y (float): Gaze Y coordinate.
    - frame (int): Frame number to check for object detections.
    - yolo_data (pd.DataFrame): YOLO detections with columns: ['frame', 'cls', 'x_min', 'y_min', 'x_max', 'y_max'].
    
    Returns:
    - list: A list of (class, distance) tuples for the closest detected bounding boxes.
    - float: The minimum distance to the closest bounding box.
    """
    frame_bboxes = yolo_data[yolo_data['frame'] == frame]
    closest_bboxes = []
    min_distance = float('inf')
    
    # Loop through the bounding boxes in the current frame and find the closest one
    for bbox_row in frame_bboxes.itertuples(index=False):
        bbox = (bbox_row.x_min, bbox_row.y_min, bbox_row.x_max, bbox_row.y_max)
        distance = calculate_distance_to_bbox(gaze_x, gaze_y, bbox)
        
        # Always update the closest bounding box
        if distance < min_distance:
            min_distance = distance
            closest_bboxes = [(bbox_row.cls, distance)]
        elif distance == min_distance:
            closest_bboxes.append((bbox_row.cls, distance))
    
    # Return the closest bounding box(es) and the minimum distance
    return closest_bboxes, min_distance

# ==========================================================
# feature functions
# ==========================================================
def calculate_most_frequent_class(class_counts: Counter) -> str:
    """
    Calculate the most frequently looked at object class.
    
    Args:
    - class_counts (Counter): A Counter object containing the count of class appearances.
    
    Returns:
    - str: The class that was most frequently looked at.
    """
    most_frequent_class = class_counts.most_common(1)[0][0] if class_counts else None
    return most_frequent_class

def calculate_avg_duration(looked_at_classes: defaultdict) -> dict:
    """
    Calculate the average number of frames the gaze was near an object class.
    
    Args:
    - looked_at_classes (defaultdict): A dictionary containing the list of frames where each object class was looked at.
    
    Returns:
    - dict: A dictionary with average gaze durations (number of frames) for each class.
    """
    return {cls: len(frames) for cls, frames in looked_at_classes.items()}

def calculate_avg_distance(closest_distances: defaultdict) -> dict:
    """
    Calculate the average distance to the closest bounding box for each object class.
    
    Args:
    - closest_distances (defaultdict): A dictionary containing the list of distances to the closest object for each class.
    
    Returns:
    - dict: A dictionary with average distances for each class.
    """
    return {cls: np.mean(distances) if distances else 0.0 for cls, distances in closest_distances.items()}

# --- Main Feature Processing Function ---
def process_yolo_gaze_data(gaze_data: pd.DataFrame, yolo_data: pd.DataFrame, target_classes: list) -> pd.DataFrame:
    """
    Process gaze data to calculate features for each object class per frame.
    
    Args:
    - gaze_data (pd.DataFrame): Gaze data with columns: ['VideoFrame', 'PixelX', 'PixelY', 'ArduinoData1'].
    - yolo_data (pd.DataFrame): YOLO detections with columns: ['frame', 'cls', 'x_min', 'y_min', 'x_max', 'y_max'].
    - target_classes (list): List of target object classes to track.
    - distance_threshold (float): Distance threshold for considering an object as "near" the gaze point (default is 100 pixels).
    
    Returns:
    - pd.DataFrame: A DataFrame containing calculated gaze features for the entire file.
    """
    class_counts = Counter()  # For tracking how often each class is looked at
    looked_at_classes = defaultdict(list)  # For tracking frames where each class was looked at
    closest_distances = defaultdict(list)  # For tracking distances to closest objects
    
    # Loop through gaze data
    for gaze_row in gaze_data.itertuples(index=False):
        frame = gaze_row.VideoFrame
        gaze_x, gaze_y = gaze_row.PixelX, gaze_row.PixelY
        
        # Get the closest detection(s) and distance
        closest_bboxes, min_distance = get_closest_detection_and_distance(gaze_x, gaze_y, frame, yolo_data)
        
        # Update class counts and track frames
        for cls, distance in closest_bboxes:
            class_counts[cls] += 1
            looked_at_classes[cls].append(frame)  # Track the frame where the class was looked at
        
        # Track the distance for the average distance calculation
        for cls in closest_bboxes:
            closest_distances[cls].append(min_distance)
    
    # Calculate the most frequently looked-at object class
    most_frequent_class = calculate_most_frequent_class(class_counts)
    
    # Calculate the number of frames per class (duration)
    avg_durations = calculate_avg_duration(looked_at_classes)
    
    # Calculate average distance to the closest object
    avg_distances = calculate_avg_distance(closest_distances)
    
    # Aggregate the data for the entire video (one line per file)
    feature_data = {
        'most_frequent_class': most_frequent_class,
        'avg_duration_near_object': np.mean(list(avg_durations.values())) if avg_durations else 0,
        'avg_distance_to_closest_object': np.mean(list(avg_distances.values())) if avg_distances else 0.0,
    }
    
    return pd.DataFrame([feature_data])
