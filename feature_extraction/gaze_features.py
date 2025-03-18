import numpy as np
import pandas as pd

# Feature Explanation: 
# 
# 1. **std_x**:
#    - **Description**: The standard deviation of the gaze locations in the X-axis across the entire dataset. 
#      It shows how much the gaze points vary in the horizontal direction.
#    - **Example**: If the gaze points in the X direction are tightly clustered around a specific value, 
#      then `std_x` will be low. Conversely, if the gaze points are spread out, `std_x` will be high.
# 
# 2. **std_y**:
#    - **Description**: The standard deviation of the gaze locations in the Y-axis across the entire dataset. 
#      It measures the variability of the gaze points in the vertical direction.
#    - **Example**: If the gaze points in the Y direction are tightly clustered around a specific value, 
#      then `std_y` will be low. If the gaze points are spread out vertically, `std_y` will be high.
# 
# 3. **total_velocity**:
#    - **Description**: The sum of all velocities (distance between consecutive gaze points) across the entire dataset. 
#      It provides an indication of how much total movement occurred during the entire gaze tracking period.
#    - **Example**: If the gaze points move significantly from one frame to the next, 
#      the `total_velocity` will be high. For example, if a person quickly shifts their gaze across the screen, 
#      `total_velocity` will be large.
# 
# 4. **max_velocity**:
#    - **Description**: The maximum velocity observed between any two consecutive gaze points. 
#      This tells us the fastest gaze movement in the dataset.
#    - **Example**: If the largest distance between two consecutive gaze points happens to be 30 pixels, 
#      then `max_velocity = 30` (assuming the distance is in pixels).
# 
# 5. **total_acceleration**:
#    - **Description**: The sum of all accelerations (change in velocity between consecutive gaze points) 
#      across the entire dataset. It provides an indication of how rapidly the gaze movements change over time.
#    - **Example**: If the velocity changes significantly between frames, 
#      the `total_acceleration` will be large. For instance, if a person shifts their gaze quickly and then rapidly stops, 
#      the acceleration values will add up to a high `total_acceleration`.
# 
# 6. **max_acceleration**:
#    - **Description**: The maximum acceleration observed between any two consecutive gaze points. 
#      This tells us the largest rate of change in gaze velocity during the tracking.
#    - **Example**: If the change in velocity between two frames is 20 pixels per second, 
#      and no other change is greater than this, then `max_acceleration = 20`.
# 
# 7. **fixation_duration**:
#    - **Description**: The total number of frames considered as part of the same fixation (i.e., the total number of consecutive 
#      frames where the gaze movement is below a threshold). It gives an indication of how long a person focuses on a specific point.
#    - **Example**: If the person spends 100 frames focusing on the same object without large gaze shifts, 
#      then `fixation_duration = 100`.
# 
# 8. **saccades**:
#    - **Description**: The total number of saccades (rapid gaze shifts) across the dataset. 
#      A saccade is counted when the distance between two consecutive gaze points exceeds a certain threshold.
#    - **Example**: If there are 10 instances where the gaze moves more than a threshold of 50 pixels between frames, 
#      then `saccades = 10`.
# 
# 9. **most_common_arduino**:
#    - **Description**: The most common value of the `ArduinoData1` column across the entire dataset. 
#      This tells us the most frequent recorded data value for `ArduinoData1`, which may represent sensor or other device data.
#    - **Example**: If the `ArduinoData1` values are mostly 1, but occasionally 0, then `most_common_arduino = 1`.


# Function to calculate the standard deviation of gaze locations across the whole dataset
def std_x_y(gaze_data: pd.DataFrame) -> tuple:
    """
    Calculate the standard deviation of gaze locations in the X and Y axes across the entire dataset.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    
    Returns:
    - tuple: A tuple containing the standard deviations of gaze locations in the X and Y axes (std_x, std_y) for the entire dataset.
    """
    std_x = np.std(gaze_data['PixelX'])
    std_y = np.std(gaze_data['PixelY'])
    return std_x, std_y


# Function to calculate the velocity (distance) between consecutive gaze points across frames
def calculate_velocity(gaze_data: pd.DataFrame) -> pd.Series:
    """
    Calculate the velocity (distance) between consecutive gaze points across frames.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    
    Returns:
    - pd.Series: A series containing the velocity for each gaze point.
    """
    gaze_data['velocity'] = np.sqrt((gaze_data['PixelX'].diff() ** 2) + (gaze_data['PixelY'].diff() ** 2))
    return gaze_data['velocity'].dropna()


# Function to calculate the acceleration (change in velocity) across frames
def calculate_acceleration(gaze_data: pd.DataFrame) -> pd.Series:
    """
    Calculate the acceleration (change in velocity) between consecutive gaze points across frames.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY', 'velocity'].
    
    Returns:
    - pd.Series: A series containing the acceleration for each gaze point.
    """
    gaze_data['acceleration'] = gaze_data['velocity'].diff().fillna(0)
    return gaze_data['acceleration']


# Function to calculate saccades (rapid gaze shifts) across frames
def calculate_saccades(gaze_data: pd.DataFrame, threshold: float = 50) -> pd.Series:
    """
    Identify saccades (rapid gaze shifts) where the distance between consecutive gaze points exceeds a threshold.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    - threshold (float): Minimum distance (in pixels) to consider a gaze shift as a saccade.
    
    Returns:
    - pd.Series: A series with 1 for saccade and 0 for no saccade.
    """
    saccades = np.sqrt((gaze_data['PixelX'].diff() ** 2) + (gaze_data['PixelY'].diff() ** 2)) > threshold
    return saccades.astype(int).fillna(0)


# Function to calculate fixation duration (sum of frames between saccades)
def fixation_duration(gaze_data: pd.DataFrame, threshold: float = 1.0) -> int:
    """
    Calculate the fixation duration across the entire dataset based on consecutive frames with small gaze shifts.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    - threshold (float): Maximum movement (in pixels) to consider frames part of the same fixation.
    
    Returns:
    - int: The total number of fixation periods across the dataset.
    """
    gaze_data['shift'] = np.sqrt((gaze_data['PixelX'].diff() ** 2) + (gaze_data['PixelY'].diff() ** 2))
    gaze_data['fixation'] = gaze_data['shift'] < threshold
    return gaze_data['fixation'].sum()


# Function to calculate the most common ArduinoData1 value across all frames
def calculate_most_common_arduino(gaze_data: pd.DataFrame) -> int:
    """
    Calculate the most common ArduinoData1 value across the entire dataset.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with a column: 'ArduinoData1'.
    
    Returns:
    - int: The most common ArduinoData1 value across all frames.
    """
    most_common_arduino = gaze_data['ArduinoData1'].mode()[0]
    return most_common_arduino


# Final function to process gaze data and return features across the whole dataset
def process_gaze_data(gaze_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process gaze data across the entire dataset to extract features such as standard deviation, velocity, acceleration,
    fixation duration, and saccades.
    
    Args:
    - gaze_data (pd.DataFrame): DataFrame containing gaze points with columns: ['VideoFrame', 'PixelX', 'PixelY'].
    
    Returns:
    - pd.DataFrame: A DataFrame containing the extracted features for the entire dataset, such as:
        'std_x', 'std_y', 'total_velocity', 'max_velocity', 'total_acceleration', 'max_acceleration', 'fixation_duration', 'saccades'.
    """
    # Calculate the required features
    std_x, std_y = std_x_y(gaze_data)
    velocity = calculate_velocity(gaze_data)
    acceleration = calculate_acceleration(gaze_data)
    saccades = calculate_saccades(gaze_data)
    fixation_duration_value = fixation_duration(gaze_data)
    most_common_arduino_value = calculate_most_common_arduino(gaze_data)

    features = {
        'std_x': std_x,
        'std_y': std_y,
        'total_velocity': velocity.sum(),  # Total velocity (sum of all velocities)
        'max_velocity': velocity.max(),    # Maximum velocity observed
        'total_acceleration': acceleration.sum(),  # Total acceleration (sum of all accelerations)
        'max_acceleration': acceleration.max(),    # Maximum acceleration observed
        'fixation_duration': fixation_duration_value,
        'saccades': saccades.sum(),  # Total number of saccades (count of saccades)
        'most_common_arduino': most_common_arduino_value  # Most common ArduinoData1 value across the dataset
    }

    return pd.DataFrame([features])
