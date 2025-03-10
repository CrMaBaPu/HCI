import numpy as np
import pandas as pd
from collections import defaultdict

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

# Counts how many fixations and average duration for each object class per frame.
def fixation_count_per_object_class(gaze_data: pd.DataFrame, yolo_data: pd.DataFrame, target_classes: list, threshold: float = 10) -> pd.DataFrame:
    """
    Count how many fixations and average duration for each object class detected by YOLO per frame.
    
    Args:
    - gaze_data (pd.DataFrame): Gaze data with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    - yolo_data (pd.DataFrame): YOLO data with columns: ['frame', 'cls', 'x_min', 'y_min', 'x_max', 'y_max'].
    - target_classes (list): List of target object classes to track fixations for.
    - threshold (int): Number of frames to consider for a gaze fixation. Default is 10.
    
    Returns:
    - pd.DataFrame: A DataFrame containing fixation count and average duration per target class per frame.
    """
    fixation_counts = defaultdict(int)
    fixation_durations = defaultdict(list)
    current_fixation = None
    frame_features = defaultdict(lambda: defaultdict(lambda: {"count": 0, "duration": []}))

    for gaze_row in gaze_data.itertuples(index=False):
        frame = gaze_row.VideoFrame
        gaze_x, gaze_y = gaze_row.PixelX, gaze_row.PixelY

        # Get object classes intersecting with gaze point
        object_classes = get_intersecting_object_classes(gaze_x, gaze_y, frame, yolo_data)

        if current_fixation is None:
            current_fixation = (gaze_x, gaze_y, 1, object_classes, frame)
        elif (gaze_x, gaze_y) == (current_fixation[0], current_fixation[1]) and frame == current_fixation[4]:
            current_fixation = (gaze_x, gaze_y, current_fixation[2] + 1, object_classes, frame)
        else:
            if current_fixation[2] >= threshold:
                for cls in current_fixation[3]:
                    frame_features[current_fixation[4]][cls]["count"] += 1
                    frame_features[current_fixation[4]][cls]["duration"].append(current_fixation[2])
            current_fixation = (gaze_x, gaze_y, 1, object_classes, frame)

    # Check last fixation if it's valid
    if current_fixation and current_fixation[2] >= threshold:
        for cls in current_fixation[3]:
            frame_features[current_fixation[4]][cls]["count"] += 1
            frame_features[current_fixation[4]][cls]["duration"].append(current_fixation[2])

    # Prepare the output DataFrame
    fixation_data = []
    for frame in sorted(frame_features.keys()):
        for cls in target_classes:
            num_fixations = frame_features[frame].get(cls, {"count": 0})["count"]
            avg_duration = np.mean(frame_features[frame].get(cls, {"duration": []})["duration"]) if frame_features[frame].get(cls, {"duration": []})["duration"] else 0.0
            fixation_data.append({
                'frame': frame,
                'class': cls,
                'fixation_count': num_fixations,
                'average_duration': avg_duration
            })

    return pd.DataFrame(fixation_data)


# Calculate how much time the gaze spends inside the bounding box of each object class per frame.
def time_spent_in_each_bbox(gaze_data: pd.DataFrame, yolo_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate how much time the gaze spends in each object class's bounding box per frame.
    
    Args:
    - gaze_data (pd.DataFrame): Gaze data with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    - yolo_data (pd.DataFrame): YOLO detections with columns: ['frame', 'cls', 'x_min', 'y_min', 'x_max', 'y_max'].
    
    Returns:
    - pd.DataFrame: A DataFrame with columns: ['frame', 'class', 'time_spent'].
    """
    time_spent = defaultdict(lambda: defaultdict(int))

    for gaze_row in gaze_data.itertuples(index=False):
        frame = gaze_row.VideoFrame
        gaze_x, gaze_y = gaze_row.PixelX, gaze_row.PixelY
        
        # Get the object classes the gaze point intersects with
        object_classes = get_intersecting_object_classes(gaze_x, gaze_y, frame, yolo_data)
        
        for cls in object_classes:
            time_spent[frame][cls] += 1
    
    # Prepare the output DataFrame
    time_spent_data = []
    for frame, classes in time_spent.items():
        for cls, time in classes.items():
            time_spent_data.append({'frame': frame, 'class': cls, 'time_spent': time})
    
    return pd.DataFrame(time_spent_data)


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


# Function to process gaze data alongside YOLO data per frame
def process_yolo_gaze_data(gaze_data: pd.DataFrame, yolo_data: pd.DataFrame, target_classes: list) -> pd.DataFrame:
    """
    Process gaze data to calculate the relevant features for specific object classes per frame.

    Args:
    - gaze_data (pd.DataFrame): Gaze data with columns: ['VideoFrame', 'PixelX', 'PixelY', 'ArduinoData1'].
    - yolo_data (pd.DataFrame): YOLO detections with columns: ['frame', 'cls', 'x_min', 'y_min', 'x_max', 'y_max'].
    - target_classes (list): List of target object classes to track.

    Returns:
    - pd.DataFrame: A DataFrame containing calculated gaze features for each target class per frame.
    """
    
    # Fixation Count and Duration per Object Class per Frame
    fixation_data = fixation_count_per_object_class(gaze_data, yolo_data, target_classes)
    
    # Time Spent in Each Object's Bounding Box per Frame
    time_spent_in_bbox = time_spent_in_each_bbox(gaze_data, yolo_data)
    
    # Prepare final features
    features = []

    # Merge the time spent and fixation data per frame and class
    for frame in set(fixation_data['frame']):
        frame_features = {
            'frame': frame,
            'label': most_common_arduino
        }

        # Extract time spent and fixation count for each target class
        for cls in target_classes:
            time_spent = time_spent_in_bbox[(time_spent_in_bbox['frame'] == frame) & (time_spent_in_bbox['class'] == cls)]
            fixation_count = fixation_data[(fixation_data['frame'] == frame) & (fixation_data['class'] == cls)]
            
            frame_features[f'time_spent_in_{cls}'] = time_spent['time_spent'].sum() if not time_spent.empty else 0
            frame_features[f'fixation_count_{cls}'] = fixation_count['fixation_count'].sum() if not fixation_count.empty else 0

        features.append(frame_features)

    return pd.DataFrame(features)
