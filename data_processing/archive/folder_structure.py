from pathlib import Path
import zipfile
import shutil

def extract_parameters(base_path):
    """
    Extracts person IDs, categories, criticality, and file IDs from the current folder structure,
    including handling zip files. This handles both CSV and MP4 files.
    """
    parameters = {'gaze': {}, 'video': {}}

    for zip_file in Path(base_path).glob("*_data_complete.zip"):  # Process all *_data_complete.zip files
        print(f"Processing ZIP file: {zip_file.stem}")  # Debug: Which ZIP file is being processed
        person_id = zip_file.stem.split("_data_complete")[0]  # Extract person ID from the ZIP filename
        
        # Initialize dictionary for this person if it doesn't exist
        if person_id not in parameters['gaze']:
            parameters['gaze'][person_id] = {}
        if person_id not in parameters['video']:
            parameters['video'][person_id] = {}

        with zipfile.ZipFile(zip_file, 'r') as archive:
            print(f"Opened ZIP file: {zip_file.stem}")  # Debug: ZIP file opened
            for file_path in archive.namelist():
                # Process CSV files for gaze data
                if file_path.endswith(".csv"):
                    print(f"Processing CSV file: {file_path}")  # Debug: Which file is being processed
                    filename = file_path.split('/')[-1]  # Get the actual CSV filename
                    filename_parts = filename.split("_")  # Split filename into parts

                    if len(filename_parts) >= 3:
                        criticality = filename_parts[0]  # Criticality is the first part of filename
                        category = filename_parts[1]    # Category is the second part of filename
                        file_id = filename_parts[-1].replace(".csv", "")  # File ID is the last part before .csv

                        # Initialize category and criticality in the dictionary if not already present
                        if category not in parameters['gaze'][person_id]:
                            parameters['gaze'][person_id][category] = {}
                        if criticality not in parameters['gaze'][person_id][category]:
                            parameters['gaze'][person_id][category][criticality] = {}

                        # Initialize the file ID list (if not already present) and append it to the dictionary
                        if file_id not in parameters['gaze'][person_id][category][criticality]:
                            parameters['gaze'][person_id][category][criticality][file_id] = []

                        # Construct the key and append it to the correct list
                        key = f"{criticality}_{category}_{file_id}.csv"
                        parameters['gaze'][person_id][category][criticality][file_id].append(file_path)  # Use the full path here

                # Process MP4 files for video data
                elif file_path.endswith(".mp4") and not file_path.endswith("output.mp4"):
                    print(f"Processing MP4 file: {file_path}")  # Debug: Which file is being processed
                    filename = file_path.split('/')[-1]  # Get the actual MP4 filename
                    filename_parts = filename.split("_")  # Split filename into parts

                    if len(filename_parts) == 3:
                        criticality = filename_parts[0]  # Criticality is the first part of filename
                        category = filename_parts[1]    # Category is the second part of filename
                        file_id = filename_parts[2].replace(".mp4", "")  # File ID is the last part before .mp4

                        # Initialize category and criticality in the dictionary if not already present
                        if category not in parameters['video'][person_id]:
                            parameters['video'][person_id][category] = {}
                        if criticality not in parameters['video'][person_id][category]:
                            parameters['video'][person_id][category][criticality] = {}

                        # Initialize the file ID list (if not already present) and append it to the dictionary
                        if file_id not in parameters['video'][person_id][category][criticality]:
                            parameters['video'][person_id][category][criticality][file_id] = []

                        # Construct the key and append it to the correct list
                        key = f"{criticality}_{category}_{file_id}.mp4"
                        parameters['video'][person_id][category][criticality][file_id].append(file_path)  # Use the full path here

    return parameters

def create_gaze_structure(base_path, parameters):
    """
    Creates a structured folder system based on extracted parameters for gaze data (CSV),
    now including person_id in the structure.
    """
    gaze_data_path = Path(base_path) / "Gaze_data"  # Ensure Gaze_data is a subfolder of Data
    gaze_data_path.mkdir(parents=True, exist_ok=True)
    print(f"Gaze_data directory created at: {gaze_data_path}")  # Debug: Where the folder is created

    # Iterate through parameters to create new structure for gaze data (CSV)
    for person_id, cat_dict in parameters['gaze'].items():
        for category, crit_dict in cat_dict.items():
            for criticality, files in crit_dict.items():
                folder_path = gaze_data_path / person_id / category / criticality
                folder_path.mkdir(parents=True, exist_ok=True)
                print(f"Created folder: {folder_path}")  # Debug: Folder being created

                # Process and copy the files to the new folder structure
                for file_id, file_paths in files.items():
                    for file_path in file_paths:
                        new_file_name = f"{person_id}_{category}_{criticality}_{file_id}.csv"
                        new_file_path = folder_path / new_file_name

                        zip_file_path = Path(base_path) / f"{person_id}_data_complete.zip"
                        with zipfile.ZipFile(zip_file_path, 'r') as archive:
                            if file_path in archive.namelist():
                                with archive.open(file_path) as file:
                                    with new_file_path.open('wb') as new_file:
                                        shutil.copyfileobj(file, new_file)
                                    print(f"Created: {new_file_path}")

def create_video_structure(base_path, parameters):
    """
    Creates a structured folder system based on extracted parameters for video data (MP4).
    """
    video_data_path = Path(base_path) / "Video_data"  # Ensure Video_data is a subfolder of Data
    video_data_path.mkdir(parents=True, exist_ok=True)
    print(f"Video_data directory created at: {video_data_path}")  # Debug: Where the folder is created

    # Iterate through parameters to create new structure for video data (MP4)
    for person_id, cat_dict in parameters['video'].items():
        for category, crit_dict in cat_dict.items():
            for criticality, files in crit_dict.items():
                folder_path = video_data_path / category / criticality
                folder_path.mkdir(parents=True, exist_ok=True)
                print(f"Created folder: {folder_path}")  # Debug: Folder being created

                # Process and copy the files to the new folder structure
                for file_id, file_paths in files.items():
                    for file_path in file_paths:
                        new_file_name = f"{category}_{criticality}_{file_id}.mp4"
                        new_file_path = folder_path / new_file_name

                        zip_file_path = Path(base_path) / f"{person_id}_data_complete.zip"
                        with zipfile.ZipFile(zip_file_path, 'r') as archive:
                            if file_path in archive.namelist():
                                with archive.open(file_path) as file:
                                    with new_file_path.open('wb') as new_file:
                                        shutil.copyfileobj(file, new_file)
                                    print(f"Created: {new_file_path}")

# Parameters
base_path = "Data"

# Run the functions
parameters = extract_parameters(base_path)
create_gaze_structure(base_path, parameters)
create_video_structure(base_path, parameters)
