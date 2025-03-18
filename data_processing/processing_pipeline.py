from pathlib import Path 
import shutil
from helper import unzip_folder, zip_folder
from yolo_object_detection import generate_yolo_csv
from processing import process_video, process_gaze_data, process_yolo_data
from segmentation import create_segments

# ==========================================================
# Constants and variables
# ==========================================================
base_path = Path("C:/Users/bayer/Documents/HCI")

# ==========================================================
# helper function
# ==========================================================

def process_single_file(video_path, yolo_path, gaze_path, video_folder, output_folder):
    # Process the video, YOLO, and gaze data
    video_data, frame_rate = process_video(video_path)
    gaze_data = process_gaze_data(gaze_path)
    yolo_data = process_yolo_data(yolo_path)
    
    # Derive the output folder structure and create it if necessary
    relative_folder = video_path.parent.relative_to(video_folder)
    output_path = output_folder / relative_folder
    output_path.mkdir(parents=True, exist_ok=True)

    # Create segments and save them
    create_segments(gaze_data, video_data, yolo_data, output_path, video_path, frame_rate)
    
# ==========================================================
# function
# ==========================================================
def process_dataset(path):
    """
    Handles the dataset processing workflow.
    
    This function extracts the dataset from a zip file, processes video, gaze, and YOLO data,
    and segments the data before re-zipping the results.
    
    Args:
    - None
    
    Returns:
    - None: The processed data is saved to the specified output directory.
    """
    zip_file_path = path / "Data.zip"
    # Unzip data
    extract_folder = zip_file_path.parent
    unzip_folder(zip_file_path, extract_folder)

    extracted_data_path = extract_folder / "Data"
    output_folder = extracted_data_path / "Data/Processed_results"
    yolo_folder = extracted_data_path / "Data/Object_detection_YOLO"
    video_folder = extracted_data_path / "Data/Input_traffic_videos"
    varjo_folder = extracted_data_path / "Data/Gaze_tracking_Varjo"

    # Generate YOLO CSV files if not already done
    if not any(yolo_folder.glob("*.csv")):
        print("Generating YOLO detection CSV files...")
        generate_yolo_csv(video_folder, yolo_folder)
    
    # Collect all video, YOLO, and gaze files
    video_files = {file.stem: file for file in video_folder.rglob("*.mp4")}
    yolo_files = {file.stem: file for file in yolo_folder.rglob("*.csv")}
    gaze_files = {file.stem: file for file in varjo_folder.rglob("*.csv")}

    print("Video files:", video_files)
    print("YOLO files:", yolo_files)
    print("Gaze files:", gaze_files)

    # Match files based on their base filenames and process them
    for filename in video_files.keys() & yolo_files.keys() & gaze_files.keys():
            print(f"Processing files for {filename}")
            process_single_file(video_files[filename], yolo_files[filename], gaze_files[filename], video_folder, output_folder)
    
    # Re-zip the contents of the extracted folder back to Data.zip
    zip_folder(extract_folder, zip_file_path)
    print("Re-zipped the folder.")
    
    unzipped_folder = extract_folder / "Data"
    if unzipped_folder.exists() and unzipped_folder.is_dir():
        shutil.rmtree(unzipped_folder)
        print(f"Deleted the folder: {unzipped_folder}")


process_dataset(base_path)
