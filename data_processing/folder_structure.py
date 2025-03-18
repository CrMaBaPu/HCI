from pathlib import Path
import zipfile
import shutil

def extract_parameters(base_path):
    """
    Extracts person IDs, categories, and file IDs from the current folder structure,
    including handling zip files. Category is now extracted from CSV filenames.
    """
    parameters = {}
    
    for zip_file in Path(base_path).glob("*.zip"):
        person_id = zip_file.stem.split("_data_complete")[0]  # Extract person ID
        parameters[person_id] = {}
        
        with zipfile.ZipFile(zip_file, 'r') as archive:
            for file_path in archive.namelist():
                parts = file_path.split('/')
                if len(parts) >= 3:
                    filename = parts[-1]  # Get the actual CSV filename
                    filename_parts = filename.split("_")
                    
                    if len(filename_parts) >= 3:
                        category = filename_parts[1]  # Extract category from filename (bike, cars)
                        crit = "crit" if "crit" in filename else "uncrit"  # Determine criticality
                        file_id = filename_parts[-1].split(".")[0]  # Extract file ID
                        
                        if category not in parameters[person_id]:
                            parameters[person_id][category] = {"crit": [], "uncrit": []}
                        
                        parameters[person_id][category][crit].append((file_path, filename))
    
    return parameters

def create_file_structure(base_path):
    """
    Creates a structured folder system based on extracted parameters.
    The "Gaze_data" folder is created parallel to the ZIP files.
    """
    parameters = extract_parameters(base_path)
    gaze_data_path = Path(base_path).parent / "Gaze_data"
    
    # Ensure the "Gaze_data" directory exists
    gaze_data_path.mkdir(parents=True, exist_ok=True)
    
    # Iterate through parameters to create new structure
    for person_id, cat_dict in parameters.items():
        for category, crit_dict in cat_dict.items():
            for crit, files in crit_dict.items():
                folder_path = gaze_data_path / person_id / category / crit
                folder_path.mkdir(parents=True, exist_ok=True)
                
                # Process and copy the files to the new folder structure
                for original_file_path, original_filename in files:
                    # Extract file content from the zip file
                    with zipfile.ZipFile(base_path + f'/{person_id}_data_complete.zip', 'r') as archive:
                        with archive.open(original_file_path) as file:
                            # Create new filename
                            file_id = original_filename.split(".")[0].split("_")[-1]
                            new_filename = f"{person_id}_{category}_{crit}_{file_id}.csv"
                            new_file_path = folder_path / new_filename
                            
                            # Write the contents of the original file to the new location
                            with new_file_path.open('wb') as new_file:
                                shutil.copyfileobj(file, new_file)
                            
                            print(f"Created: {new_file_path}")

# Parameters
base_path = "HCI/Data"  # Current data directory (containing zip files)

# Run the function
create_file_structure(base_path)