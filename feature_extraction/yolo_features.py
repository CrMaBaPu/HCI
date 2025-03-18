import numpy as np 
import pandas as pd
from collections import defaultdict

# Feature Explanation:
# 
# 1. **total_objects_all**:
#    - **Description**: This is the total number of detections (objects) in a single frame.
#    - **Example**: If frame 1 has 5 bounding boxes (detections), then `total_objects_all = 5`.
#    
# 2. **total_classes_ratio**:
#    - **Description**: The ratio of distinct classes to the total number of detections in a frame. 
#      This shows how many different classes are present relative to the total detections.
#    - **Example**: If frame 1 has 5 detections, and there are 3 distinct classes, then `total_classes_ratio = 3 / 5 = 0.6`.
#    
# 3. **most_central_detection_size**:
#    - **Description**: The size (area) of the bounding box for the most central detection, i.e., the detection closest to the center of the frame.
#    - **Example**: If there are three detections with sizes [100, 150, 120] and the most central detection has a size of 150, then `most_central_detection_size = 150`.
#    
# 4. **max_size_[cls]** (for each target class):
#    - **Description**: The maximum size (area) of bounding boxes for a specific class across all frames in the segment.
#      This measures the largest detection for each class.
#    - **Example**: If for the 'car' class, frame 1 has a bounding box size of 100, and frame 2 has a size of 200, then `max_size_car = max(100, 200) = 200`.
#    
# 5. **num_per_cls_[cls]** (for each target class):
#    - **Description**: The total number of detections for a specific class across the entire video segment (sum of detections across all frames).
#    - **Example**: If 'car' was detected 3 times in frame 1 and 2 times in frame 2, then `num_per_cls_car = 3 + 2 = 5`.
#    
# 6. **max_total_objects_all**:
#    - **Description**: The maximum number of detections in any single frame across the entire segment.
#    - **Example**: If frame 1 has 5 detections and frame 2 has 8 detections, then `max_total_objects_all = 8`.
#    
# 7. **avg_total_classes_ratio**:
#    - **Description**: The average ratio of distinct classes to the total detections for all frames in the segment. 
#      This shows how many distinct classes are present on average relative to the total detections per frame.
#    - **Example**: If in frame 1 the ratio is 0.6 and in frame 2 the ratio is 0.8, then `avg_total_classes_ratio = (0.6 + 0.8) / 2 = 0.7`.
#    
# 8. **max_most_central_detection_size**:
#    - **Description**: The maximum size of the most central detection across all frames in the segment.
#    - **Example**: If the size of the most central detection for frame 1 is 150 and for frame 2 it is 200, then `max_most_central_detection_size = max(150, 200) = 200`.
# 
# These features are aggregated across the entire segment, combining frame-specific measurements to provide a holistic view of the detections and their characteristics.
import numpy as np
import pandas as pd
from collections import defaultdict

# ==========================================================
# Helper functions
# ==========================================================

def calculate_size(bbox: tuple) -> float:
    """
    Calculate the size (area) of a bounding box.
    
    Args:
        bbox (tuple): A tuple containing the frame, class, confidence, 
                      x_min, y_min, x_max, and y_max for a detected object.
    
    Returns:
        float: The area of the bounding box, calculated as width * height.
    """
    _, _, _, x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min
    return width * height


def calculate_distance_to_center(bbox: tuple, frame_width: float, frame_height: float) -> float:
    """
    Calculate the distance from the centroid of the bounding box to the center of the frame.
    
    Args:
        bbox (tuple): A tuple containing the frame, class, confidence, 
                      x_min, y_min, x_max, and y_max for a detected object.
        frame_width (float): The width of the video frame.
        frame_height (float): The height of the video frame.
    
    Returns:
        float: The Euclidean distance from the bounding box's centroid to the center of the frame.
    """
    _, _, _, x_min, y_min, x_max, y_max = bbox
    centroid_x = (x_min + x_max) / 2
    centroid_y = (y_min + y_max) / 2
    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2
    return np.sqrt((centroid_x - frame_center_x) ** 2 + (centroid_y - frame_center_y) ** 2)


# ==========================================================
# Feature calculation functions
# ==========================================================

def calculate_total_objects(bboxes: list) -> int:
    """
    Calculate the total number of detections (objects) in a frame.
    
    Args:
        bboxes (list): A list of bounding boxes for the current frame.
    
    Returns:
        int: The total number of objects detected in the frame.
    """
    return len(bboxes)


def calculate_total_classes_ratio(bboxes: list) -> float:
    """
    Calculate the ratio of distinct classes to the total detections in a frame.
    
    Args:
        bboxes (list): A list of bounding boxes for the current frame.
    
    Returns:
        float: The ratio of distinct classes to total detections.
    """
    total_objects_all = len(bboxes)
    unique_classes = len(set([d[1] for d in bboxes]))  # Get unique class labels in the frame
    return unique_classes / total_objects_all if total_objects_all else 0  # Avoid division by zero


def calculate_most_central_detection(bboxes: list, frame_width: float, frame_height: float) -> float:
    """
    Calculate the size of the most central detection in the frame.
    
    Args:
        bboxes (list): A list of bounding boxes for the current frame.
        frame_width (float): The width of the video frame.
        frame_height (float): The height of the video frame.
    
    Returns:
        float: The size (area) of the most central detection.
    """
    central_sizes = []
    for bbox in bboxes:
        distance_to_center = calculate_distance_to_center(bbox, frame_width, frame_height)
        size = calculate_size(bbox)
        central_sizes.append((distance_to_center, size))  # Store the distance and size for central detection
    
    most_central_detection = min(central_sizes, key=lambda x: x[0], default=(None, 0))
    return most_central_detection[1]  # Get the size of the most central detection


def calculate_max_class_size(bboxes: list, target_classes: list) -> dict:
    """
    Calculate the maximum bounding box size per class across all detections in the frame.
    
    Args:
        bboxes (list): A list of bounding boxes for the current frame.
        target_classes (list): List of target classes to calculate the maximum bounding box size for.
    
    Returns:
        dict: A dictionary with the max size for each target class.
    """
    max_class_size = {}
    for cls in target_classes:
        cls_bboxes = [bbox for bbox in bboxes if bbox[1] == cls]  # Filter bounding boxes for the class
        max_class_size[cls] = max([calculate_size(bbox) for bbox in cls_bboxes], default=0)  # Get the max size
    return max_class_size


def accumulate_class_detections(bboxes: list) -> dict:
    """
    Accumulate the total number of detections for each class across the segment.
    
    Args:
        bboxes (list): A list of bounding boxes for the current frame.
    
    Returns:
        dict: A dictionary with counts of detections per class.
    """
    clscounts = defaultdict(int)  # Dictionary to store counts of detections per class
    for _, cls, _, _, _, _, _ in bboxes:
        clscounts[cls] += 1  # Increment the count for the class
    return clscounts


# ==========================================================
# Process YOLO Data
# ==========================================================

def process_yolo_data(yolo_data: pd.DataFrame, frame_width: float, frame_height: float) -> pd.DataFrame:
    """
    Process YOLO detection data to calculate aggregated features for each frame in a video segment.
    
    Args:
        yolo_data (pd.DataFrame): A DataFrame containing YOLO detection data with columns:
                                  frame, class, confidence, x_min, y_min, x_max, y_max.
        frame_width (float): The width of the video frame.
        frame_height (float): The height of the video frame.
    
    Returns:
        pd.DataFrame: A DataFrame with aggregated features across all frames of the segment.
    """
    frame_features = []
    segment_features = defaultdict(list)

    grouped_by_frame = defaultdict(list)
    for row in yolo_data.itertuples(index=False):
        grouped_by_frame[row.frame].append(row)

    target_classes = ['car', 'bicycle', 'person']

    for frame, bboxes in grouped_by_frame.items():
        # Calculate features for the frame
        total_objects_all = calculate_total_objects(bboxes)
        total_classes_ratio = calculate_total_classes_ratio(bboxes)
        most_central_detection_size = calculate_most_central_detection(bboxes, frame_width, frame_height)
        max_class_size = calculate_max_class_size(bboxes, target_classes)
        class_detections = accumulate_class_detections(bboxes)

        # Prepare the feature dictionary for this frame
        features = {
            'frame': frame,
            'total_objects_all': total_objects_all,
            'total_classes_ratio': total_classes_ratio,
            'most_central_detection_size': most_central_detection_size,
        }

        # Add max size for each class
        features.update(max_class_size)

        # Accumulate the total number of detections per class
        for cls, count in class_detections.items():
            segment_features[f'num_per_cls_{cls}'].append(count)

        # Add the frame's features to the list
        frame_features.append(features)

    # Aggregate segment-level features across all frames
    aggregated_features = {}
    for key, values in segment_features.items():
        aggregated_features[key] = sum(values)

    # Aggregate frame features: Max of total objects, max central detection size, etc.
    aggregated_features['max_total_objects_all'] = max([f['total_objects_all'] for f in frame_features])
    aggregated_features['avg_total_classes_ratio'] = np.mean([f['total_classes_ratio'] for f in frame_features])
    aggregated_features['max_most_central_detection_size'] = max([f['most_central_detection_size'] for f in frame_features])

    # Return the aggregated features in a DataFrame
    return pd.DataFrame([aggregated_features])

