import numpy as np
import pandas as pd

# Standard Deviation in X and Y
def std_x_y(gaze_data: pd.DataFrame) -> tuple:
    """
    Calculate the standard deviation of gaze locations in the X and Y axes.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    
    Returns:
    - tuple: A tuple containing the standard deviations of gaze locations in the X and Y axes (std_x, std_y).
    """
    std_x = np.std(gaze_data['PixelX'])
    std_y = np.std(gaze_data['PixelY'])
    return std_x, std_y

# Velocity: Distance between consecutive points (normalized by frame)
def calculate_velocity(gaze_data: pd.DataFrame) -> np.ndarray:
    """
    Calculate the velocity (distance) between consecutive gaze points.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    
    Returns:
    - np.ndarray: An array of velocities, where each element represents the distance between two consecutive gaze points.
    """
    velocities = []
    previous_gaze = None
    for gaze_row in gaze_data.itertuples(index=False):
        current_gaze = (gaze_row.PixelX, gaze_row.PixelY)
        if previous_gaze is not None:
            distance = np.sqrt((current_gaze[0] - previous_gaze[0])**2 + (current_gaze[1] - previous_gaze[1])**2)
            velocities.append(distance)
        previous_gaze = current_gaze
    return np.array(velocities)

# Acceleration: Change in velocity
def calculate_acceleration(velocities: np.ndarray) -> np.ndarray:
    """
    Calculate the acceleration (change in velocity) between consecutive gaze points.
    
    Args:
    - velocities (np.ndarray): Array of velocities, where each element represents the distance between two consecutive gaze points.
    
    Returns:
    - np.ndarray: An array of accelerations, where each element represents the change in velocity between consecutive gaze points.
    """
    accelerations = np.diff(velocities)  # Change in velocity between frames
    return accelerations

# Saccades: Gaze shifts larger than a given threshold
def calculate_saccades(gaze_data: pd.DataFrame, threshold: float = 50) -> int:
    """
    Count the number of saccades (rapid gaze shifts) where the distance between consecutive gaze points exceeds a threshold.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    - threshold (float): Minimum distance (in pixels) to consider a gaze shift as a saccade. Default is 50 pixels.
    
    Returns:
    - int: The number of saccades (gaze shifts that exceed the threshold distance).
    """
    saccades = 0
    previous_gaze = None
    for gaze_row in gaze_data.itertuples(index=False):
        current_gaze = (gaze_row.PixelX, gaze_row.PixelY)
        if previous_gaze is not None:
            distance = np.sqrt((current_gaze[0] - previous_gaze[0])**2 + (current_gaze[1] - previous_gaze[1])**2)
            if distance >= threshold:
                saccades += 1
        previous_gaze = current_gaze
    return saccades

# Average Fixation Duration
def average_fixation_duration(gaze_data: pd.DataFrame, fixation_threshold: int = 10) -> float:
    """
    Calculate the average duration of fixations, where a fixation is defined as a series of consecutive frames with the same gaze point.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    - fixation_threshold (int): The minimum number of frames required to consider a gaze as a fixation. Default is 10 frames.
    
    Returns:
    - float: The average duration (in frames) of fixations.
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
    
    return np.mean(gaze_durations) if gaze_durations else 0.0

# Speed Across Frames (Average Distance over a range of frames)
def speed_across_frames(gaze_data: pd.DataFrame, frame_range: int = 5) -> np.ndarray:
    speeds = []
    for i in range(len(gaze_data) - frame_range):
        distance = np.sqrt((gaze_data['PixelX'][i+frame_range] - gaze_data['PixelX'][i])**2 +
                           (gaze_data['PixelY'][i+frame_range] - gaze_data['PixelY'][i])**2)
        speeds.append(distance / frame_range)
    return np.array(speeds)

# Function to process gaze data and return the feature DataFrame
def process_gaze_data(gaze_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process gaze data to extract a set of features including standard deviation of gaze, 
    average velocity, average acceleration, fixation duration, and the number of saccades.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    
    Returns:
    - pd.DataFrame: A DataFrame containing the extracted features. The columns include:
        'std_x', 'std_y', 'average_velocity', 'average_acceleration', 
        'average_fixation_duration', and 'saccades'.
    """
    # Calculate individual features
    std_x, std_y = std_x_y(gaze_data)
    velocities = calculate_velocity(gaze_data)
    accelerations = calculate_acceleration(velocities)
    fixation_duration = average_fixation_duration(gaze_data)
    saccades = calculate_saccades(gaze_data)
    speed_frames = speed_across_frames(gaze_data)
    
    # Create the feature DataFrame
    features = {
        'std_x': std_x,
        'std_y': std_y,
        'average_velocity': np.mean(velocities) if len(velocities) > 0 else 0.0,
        'average_acceleration': np.mean(accelerations) if len(accelerations) > 0 else 0.0,
        'average_fixation_duration': fixation_duration,
        'saccades': saccades,
        'average_speed_across_frames': np.mean(speed_frames) if len(speed_frames) > 0 else 0.0,
    }
    
    return pd.DataFrame([features])

