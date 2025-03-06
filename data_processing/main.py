from pathlib import Path
from yolo_object_detection import generate_yolo_csv
from processing import process_video, process_gaze_data, process_yolo_data
from segmentation import create_segments

def main():
    """
    The entry point for processing the dataset.
    
    Collects the matching files (YOLO, Varjo, and video) and processes them.
    After processing, the rsesults are saved in the specified output folder.
    """
    base_path = Path("C:/Users/bayer/Documents/HCI")
    output_folder = base_path / "Data/Processed_results"
    yolo_folder = base_path / "Data/Object_detection_YOLO"
    video_folder = base_path / "Data/Input_traffic_videos"
    varjo_folder = base_path / "Data/Gaze_tracking_Varjo"
    
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
    for filename in video_files:
        if filename in yolo_files and filename in gaze_files and filename in video_files:
            print(f"Processing files for {filename}")
            video_path = video_files[filename]
            yolo_path = yolo_files[filename]
            gaze_path = gaze_files[filename]

            # Process the video, YOLO, and gaze data
            video_data, frame_rate = process_video(video_path)
            gaze_data = process_gaze_data(gaze_path)
            yolo_data = process_yolo_data(yolo_path)
            
            # Derive the output folder structure
            relative_folder = video_path.parent.relative_to(video_folder)
            output_path = output_folder / relative_folder
            output_path.mkdir(parents=True, exist_ok=True)

            # Create segments and save them
            create_segments(gaze_data, video_data, yolo_data, output_path, video_path, frame_rate)


if __name__ == "__main__":
    main()