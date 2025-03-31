from pathlib import Path
import pandas as pd
import math
from processing_functions import process_video, process_gaze_data, process_yolo_data

frame_rate = 60 #30

def create_segments(
    gaze_data: pd.DataFrame,
    yolo_data: pd.DataFrame,
    output_path: Path,
    gaze_path: Path,
    yolo_path: Path,
    frame_rate: int  
) -> None:
    """
    Creates structured segments from gaze and YOLO data, ensuring uniform segment length.

    Args:
    - gaze_data (pd.DataFrame): Processed gaze tracking data.
    - yolo_data (pd.DataFrame): Processed YOLO object detection data.
    - output_path (Path): Path to save the processed results.
    - gaze_path (Path): Path to the original gaze CSV file.
    - yolo_path (Path): Path to the original YOLO CSV file.
    - frame_rate (int): Frame rate of the data (either 30 or 60).

    Returns:
    - None: Segments are saved to the file system.
    """

    # Set segment length and step size based on frame rate
    if frame_rate == 30:
        segment_length = 150
        step_size = 30
    elif frame_rate == 60:
        segment_length = 300
        step_size = 60
    else:
        raise ValueError(f"Unsupported frame rate: {frame_rate}. Expected 30 or 60.")

    # Determine total number of frames
    num_frames = int(gaze_data["VideoFrame"].iloc[-1])  # Last frame in gaze data
    print(num_frames)
    start_frame = int(gaze_data["VideoFrame"].iloc[0])  # First frame in gaze data
    print(start_frame)
    # Create structured folder
    structured_folder = output_path / yolo_path.parent.parts[-2] / yolo_path.parent.parts[-1]
    structured_folder.mkdir(parents=True, exist_ok=True)

    # Use filename bases
    yolo_filename_base = yolo_path.stem
    gaze_filename_base = gaze_path.stem

    # Create segments
    for start in range(start_frame, num_frames - segment_length + 1, step_size):
        end = start + segment_length - 1

        yolo_segment = yolo_data[(yolo_data['frame'] >= start) & (yolo_data['frame'] <= end)]
        gaze_segment = gaze_data[(gaze_data['VideoFrame'] >= start) & (gaze_data['VideoFrame'] <= end)]

        yolo_filename = f"{yolo_filename_base}_yolo_{start:04d}-{end:04d}.csv"
        gaze_filename = f"{gaze_filename_base}_gaze_{start:04d}-{end:04d}.csv"

        yolo_segment.to_csv(structured_folder / yolo_filename, index=False)
        gaze_segment.to_csv(structured_folder / gaze_filename, index=False)

    print(f"Segments saved to {structured_folder}")



def process_single_file(
    yolo_path: Path,
    gaze_paths: list,
    output_folder: Path
) -> None:
    """
    Processes video, YOLO, and multiple gaze datasets for a single video.

    Args:
    - video_path (Path): Path to the video file.
    - yolo_path (Path): Path to the corresponding YOLO CSV file.
    - gaze_paths (list): List of paths to gaze CSV files.
    - video_folder (Path): Path to the video folder.
    - output_folder (Path): Path to the output folder where results will be saved.

    Returns:
    - None: Processed segments are saved.
    """
    yolo_data = process_yolo_data(yolo_path)

    for gaze_path in gaze_paths:
        gaze_data = process_gaze_data(gaze_path)
        create_segments(gaze_data, yolo_data, output_folder, gaze_path, yolo_path, frame_rate)


def process_data_files(base_path: Path) -> None:
    """
    Processes video, YOLO, and gaze files from the dataset.

    Args:
    - base_path (Path): Base path of the dataset directory.

    Returns:
    - None: Processed data is saved to output folders.
    """
    data_path = base_path / "Data"
    output_folder = data_path / "Processed_data"
    gaze_folder = data_path / "Gaze_data"
    yolo_folder = data_path / "YOLO_data"

    yolo_files = {file.stem: file for file in yolo_folder.rglob("*.csv")}
    gaze_files = {}

    for gaze_file in gaze_folder.rglob("*.csv"):
        base_name = "_".join(gaze_file.stem.split("_")[1:])  # Remove person ID
        if base_name not in gaze_files:
            gaze_files[base_name] = []
        gaze_files[base_name].append(gaze_file)

    for filename in yolo_files.keys() & gaze_files.keys():
        print(f"Processing files for {filename}")
        process_single_file(yolo_files[filename], gaze_files[filename], output_folder)


# Base path of the data directory
base_path = Path("C:/Users/bayer/Documents/HCI")
process_data_files(base_path)
