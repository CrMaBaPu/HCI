from pathlib import Path
import pandas as pd
import cv2

def process_video(video_path: Path) -> pd.DataFrame:
    """
    Processes video data. Extracting frame data, timestamps, and frame rate.
    
    Args:
    video_path (Path): The path to the video file.
    
    Returns:
    pd.DataFrame: A DataFrame with video frame data (e.g., frame number, timestamp).
    """
    video = cv2.VideoCapture(str(video_path))
    frame_data = []
    
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp = video.get(cv2.CAP_PROP_POS_MSEC) / 1000
        
        frame_data.append([frame_number, timestamp])
    
    video.release()
    
    video_df = pd.DataFrame(frame_data, columns=["FrameNumber", "Timestamp"])
    return video_df, frame_rate


def process_yolo_data(yolo_path: Path) -> pd.DataFrame:
    """
    Process YOLO detection data from CSV file.
    
    Args:
    yolo_path (Path): The path to the YOLO detection data CSV file.
    
    Returns:
    pd.DataFrame: Processed YOLO detection data with columns:
                  ['frame', 'class', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max']
    """
    # Read YOLO data
    yolo_data = pd.read_csv(yolo_path)

    # Check if data is already in the correct format
    if {'frame', 'class', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max'}.issubset(yolo_data.columns):
        return yolo_data[['frame', 'class', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max']]

    # If data is in the old format, convert it
    required_columns = {'frame', 'class', 'confidence', 'x', 'y', 'width', 'height'}
    if required_columns.issubset(yolo_data.columns):
        # Convert (x, y, width, height) to (x_min, y_min, x_max, y_max)
        yolo_data['x_min'] = yolo_data['x'] - (yolo_data['width'] / 2)
        yolo_data['y_min'] = yolo_data['y'] - (yolo_data['height'] / 2)
        yolo_data['x_max'] = yolo_data['x'] + (yolo_data['width'] / 2)
        yolo_data['y_max'] = yolo_data['y'] + (yolo_data['height'] / 2)

        # Select and reorder columns
        processed_data = yolo_data[['frame', 'class', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max']]
        return processed_data




def process_gaze_data(varjo_path: Path) -> pd.DataFrame:
    """
    Process gaze data from the Varjo CSV file and downsample it to match the video frame rate.
    
    Args:
    varjo_path (Path): The path to the Varjo gaze tracking data CSV file.
    video_frame_count (int): The total number of frames in the video.
    
    Returns:
    pd.DataFrame: Processed gaze tracking data with downsampled timestamps to match the video.
    """
    gaze_data = pd.read_csv(varjo_path, sep=";")
    
    # Extract relevant columns and process
    numeric_columns = ['VideoFrame','PixelX', 'PixelY', 'ArduinoData1']
    for col in numeric_columns:
        gaze_data[col] = gaze_data[col].replace({',': '.'}, regex=True)  # Replace comma with period
        gaze_data[col] = pd.to_numeric(gaze_data[col], errors='coerce')  # Convert to numeric, coercing errors to NaN
    
    # Check for any NaN values in the numeric columns and remove those rows
    gaze_data = gaze_data.dropna(subset=numeric_columns)
    
    # Extract relevant columns and process
    gaze_data_processed = gaze_data[['VideoFrame', 'PixelX', 'PixelY', 'ArduinoData1']]
    
    # Group by 'VideoFrame' and calculate the mean for the numeric columns
    gaze_data_mean = gaze_data_processed.groupby('VideoFrame', as_index=False).mean()
    
      # # remove Video frame 0 and adjust Video frame numbering
    # gaze_data = gaze_data[gaze_data['VideoFrame'] > 0].copy()
    # gaze_data['VideoFrame'] -= 1
            
    return gaze_data_mean