import pathlib as Path
import pandas as pd
import math

import pathlib as Path 
import pandas as pd
import math

def create_segments(
    gaze_data: pd.DataFrame,
    video_data: pd.DataFrame,
    yolo_data: pd.DataFrame,
    output_path: Path,
    video_path: Path,
    frame_rate: float,
    segment_length: int = 5,
    window_increment: int = 1
):
    """
    Create segments from the given gaze, video, and YOLO data, then save them directly to the correct folders.

    Args:
    - gaze_data (pd.DataFrame): The processed gaze tracking data.
    - video_data (pd.DataFrame): The processed video data.
    - yolo_data (pd.DataFrame): The processed YOLO object detection data.
    - output_path (Path): The root path where the processed results will be saved.
    - frame_rate (float): The frame rate of the video.
    - segment_length (int): The length of each segment in seconds (default 5 seconds).
    - window_increment (int): The increment of the window in seconds (default 1 second).
    - video_path (Path): The path to the video file.
    
    Returns:
    - None: The segments are saved directly to the file system.
    """
    num_frames = video_data.shape[0]  # Total number of frames in the video
    frame_rate = math.ceil(frame_rate)
    segment_frames = int(segment_length * frame_rate)  # Number of frames in a 5-second segment
    window_step = int(window_increment * frame_rate)  # Number of frames for each 1-second increment

    video_base_name = video_path.stem
    file_folder = output_path / video_base_name
    file_folder.mkdir(parents=True, exist_ok=True)

    # Loop through the video frames to create segments
    for start_frame in range(0, num_frames - segment_frames + 1, window_step):
        end_frame = start_frame + segment_frames - 1

        # Extract YOLO and gaze data for the segment (without forcing them to match 1-to-1)
        yolo_segment = yolo_data[(yolo_data['frame'] >= start_frame) & (yolo_data['frame'] <= end_frame)]
        gaze_segment = gaze_data[(gaze_data['VideoFrame'] >= start_frame) & (gaze_data['VideoFrame'] <= end_frame)]

        # Save YOLO data for the segment
        yolo_filename = f"{video_base_name}_yolo_{start_frame:04d}-{end_frame:04d}.csv"
        gaze_filename = f"{video_base_name}_gaze_{start_frame:04d}-{end_frame:04d}.csv"

        yolo_segment.to_csv(file_folder / yolo_filename, index=False)
        gaze_segment.to_csv(file_folder / gaze_filename, index=False)

    print(f"Segments for {video_base_name} have been saved to {file_folder}")

