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