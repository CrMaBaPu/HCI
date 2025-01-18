# data_processing/yolo_object_detection.py
import os
import cv2
import csv
from ultralytics import YOLO
from pathlib import Path

# Initialize YOLO model
model = YOLO("yolo11x.pt")  # Use correct model path or version

def generate_yolo_csv(input_folder: str, output_folder: str):
    """
    Process each video in the given input folder using YOLO model,
    detect objects in the frames, and save the results in CSV format.

    Args:
    input_folder (str): Path to the folder containing video files.
    output_folder (str): Path to the folder where the resulting CSV files will be saved.

    Returns:
    None
    """
    # Ensure the output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Loop through all video files in the input folder
    for video_file in os.listdir(input_folder):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(input_folder, video_file)
            video_capture = cv2.VideoCapture(video_path)

            # Check if the video was opened successfully
            if not video_capture.isOpened():
                print(f"Error: Cannot open video file {video_path}")
                continue

            detection_metadata = []
            frame_index = 0

            # Process the video frame by frame
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
            save_to_csv(video_file, detection_metadata, output_folder)

def save_to_csv(video_file: str, metadata: list, output_folder: str):
    """
    Saves the detection metadata to a CSV file.

    Args:
    video_file (str): The name of the video file (used to generate output CSV name).
    metadata (list): List of detection data for each frame.
    output_folder (str): The folder where the CSV will be saved.

    Returns:
    None
    """
    csv_file = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}.csv")

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
