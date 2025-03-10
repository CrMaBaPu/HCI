import numpy as np
import pandas as pd

# Standard Deviation in X and Y (for each frame)
def std_x_y(gaze_data: pd.DataFrame) -> tuple:
    """
    Calculate the standard deviation of gaze locations in the X and Y axes for each frame.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    
    Returns:
    - tuple: A tuple containing the standard deviations of gaze locations in the X and Y axes (std_x, std_y) for each frame.
    """
    std_x = gaze_data.groupby('VideoFrame')['PixelX'].std()
    std_y = gaze_data.groupby('VideoFrame')['PixelY'].std()
    return std_x, std_y

# Velocity: Distance between consecutive points for each frame
def calculate_velocity(gaze_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the velocity (distance) between consecutive gaze points for each frame.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    
    Returns:
    - pd.DataFrame: DataFrame containing the velocity for each gaze point.
    """
    gaze_data['velocity'] = np.sqrt((gaze_data['PixelX'].diff() ** 2) + (gaze_data['PixelY'].diff() ** 2))
    gaze_data['velocity'].fillna(0)
    return gaze_data

# Acceleration: Change in velocity for each frame
def calculate_acceleration(gaze_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the acceleration (change in velocity) between consecutive gaze points for each frame.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY', 'velocity'].
    
    Returns:
    - pd.DataFrame: DataFrame containing the acceleration for each gaze point.
    """
    gaze_data['acceleration'] = gaze_data['velocity'].diff().fillna(0)
    return gaze_data

# Saccades: Gaze shifts larger than a given threshold (for each frame)
def calculate_saccades(gaze_data: pd.DataFrame, threshold: float = 50) -> pd.DataFrame:
    """
    Identify saccades (rapid gaze shifts) where the distance between consecutive gaze points exceeds a threshold.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    - threshold (float): Minimum distance (in pixels) to consider a gaze shift as a saccade.
    
    Returns:
    - pd.DataFrame: DataFrame with a column indicating saccades for each gaze point (1 for saccade, 0 for no saccade).
    """
    gaze_data['saccade'] = np.sqrt((gaze_data['PixelX'].diff() ** 2) + (gaze_data['PixelY'].diff() ** 2)) > threshold
    gaze_data['saccade'] = gaze_data['saccade'].astype(int)
    gaze_data['saccade'].fillna(0)
    return gaze_data

# Fixation Duration: Calculate the duration of fixations (number of consecutive frames with the same gaze point)
def average_fixation_duration(gaze_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the average fixation duration for each frame.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    
    Returns:
    - pd.DataFrame: DataFrame containing the fixation duration for each frame.
    """
    gaze_data['fixation_duration'] = gaze_data.groupby('VideoFrame')['VideoFrame'].transform('size')
    return gaze_data

# Speed Across Frames (Average Distance over a range of frames using sliding window)
def speed_across_frames(gaze_data: pd.DataFrame, frame_range: int = 5) -> pd.DataFrame:
    """
    Calculate the average speed across a sliding window of frames.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    - frame_range (int): The number of frames over which to calculate the average speed.
    
    Returns:
    - pd.DataFrame: DataFrame containing the speed across frames for each frame.
    """
    gaze_data['speed_across_frames'] = np.nan
    for i in range(len(gaze_data) - frame_range):
        start_frame = gaze_data.iloc[i]
        end_frame = gaze_data.iloc[i + frame_range]
        distance = np.sqrt((end_frame['PixelX'] - start_frame['PixelX']) ** 2 + (end_frame['PixelY'] - start_frame['PixelY']) ** 2)
        gaze_data.at[i, 'speed_across_frames'] = distance / frame_range
    gaze_data['speed_across_frames'].fillna(0)
    return gaze_data

# Function to process gaze data and return the feature DataFrame for each frame
def process_gaze_data(gaze_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process gaze data to extract a set of features for each frame including standard deviation of gaze,
    velocity, acceleration, fixation duration, and the number of saccades.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    
    Returns:
    - pd.DataFrame: A DataFrame containing the extracted features for each frame, such as:
        'std_x', 'std_y', 'average_velocity', 'average_acceleration', 'fixation_duration', 'saccades', and 'average_speed_across_frames'.
    """
    # Calculate individual features
    gaze_data = calculate_velocity(gaze_data)
    gaze_data = calculate_acceleration(gaze_data)
    gaze_data = calculate_saccades(gaze_data)
    gaze_data = average_fixation_duration(gaze_data)
    gaze_data = speed_across_frames(gaze_data)
    
    # Calculate standard deviations for each frame
    std_x, std_y = std_x_y(gaze_data)
    gaze_data['std_x'] = gaze_data['VideoFrame'].map(std_x)
    gaze_data['std_y'] = gaze_data['VideoFrame'].map(std_y)

    # Group by frame and aggregate the features
    features = gaze_data.groupby('VideoFrame').agg(
        std_x=('std_x', 'first'),  # First std_x for each frame
        std_y=('std_y', 'first'),  # First std_y for each frame
        average_velocity=('velocity', 'mean'),
        average_acceleration=('acceleration', 'mean'),
        fixation_duration=('fixation_duration', 'mean'),
        saccades=('saccade', 'sum'),
        average_speed_across_frames=('speed_across_frames', 'mean')
    ).reset_index()
    
    return features
