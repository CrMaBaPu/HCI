from pathlib import Path
import cv2
import csv
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("yolo11x.pt")  # Use the correct model path or version

def generate_yolo_csv(video_folder: Path, yolo_folder: Path):
    """
    Process each video in the given input folder using YOLO model,
    detect objects in the frames, and save the results in CSV format.

    Args:
    video_folder (Path): Path to the folder containing video files.
    yolo_folder (Path): Path to the folder where the resulting CSV files will be saved.

    Returns:
    None
    """
    # Loop through all video files in the video folder
    for video_file in video_folder.rglob("*.mp4"):
        video_path = video_file
        relative_path = video_file.relative_to(video_folder)  # Get the relative path of the video file
        yolo_csv_path = yolo_folder / relative_path.with_suffix(".csv")  # Same folder structure, but with .csv extension
        
        # Ensure the output folder exists in YOLO_data
        yolo_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Process the video file and generate YOLO detection data
        print(f"Processing video file: {video_path.stem}")
        
        video_capture = cv2.VideoCapture(str(video_path))
        
        if not video_capture.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            continue

        detection_metadata = []
        frame_index = 0

        while True:
            ret, frame = video_capture.read()
            if not ret:
                break  # Exit if no frames left

            # Perform YOLO object detection
            results = model(frame)

            # Store detections for the current frame
            frame_data = {"frame": frame_index, "detections": []}
            for result in results[0].boxes:
                class_name = model.names[int(result.cls)]  # Get class name
                confidence = float(result.conf)
                bbox = result.xyxy[0].tolist()  # Convert tensor to list [x_min, y_min, x_max, y_max]

                frame_data["detections"].append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": bbox
                })

            detection_metadata.append(frame_data)
            frame_index += 1

        # Release the video capture object
        video_capture.release()

        # Save detections to CSV
        save_to_csv(yolo_csv_path, detection_metadata)

def save_to_csv(csv_file: Path, metadata: list):
    """
    Saves the detection metadata to a CSV file.

    Args:
    csv_file (Path): The path to the output CSV file.
    metadata (list): List of detection data for each frame.

    Returns:
    None
    """
    # Write to CSV
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write header row
        writer.writerow(['frame', 'class', 'confidence', 'x_min', 'y_min', 'x_max', 'y_max'])

        # Write detection data
        for frame_data in metadata:
            for detection in frame_data['detections']:
                writer.writerow([frame_data['frame'], detection['class'], detection['confidence'], *detection['bbox']])

    print(f"Detection metadata saved to {csv_file}")

# Generate YOLO files (step 1)

generate_yolo_csv(base_path / "Data" / "Video_data", base_path / "Data" / "YOLO_data")
