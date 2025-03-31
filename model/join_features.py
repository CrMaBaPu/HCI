import pandas as pd
from pathlib import Path

# Define the function to extract key columns from filenames
def extract_key_columns(file_name, is_yolo=False):
    # Split the filename by underscores
    parts = file_name.split('_')
    
    # Extract the columns based on their positions in the split filename
    if is_yolo:
        # YOLO files do not have a person_id
        person_id = None
        category = parts[1]
        criticality = parts[2]
        file_id = parts[3]
        segment = parts[5]
    else:
        # For gaze and combi files, person_id is the first part
        person_id = parts[0]
        category = parts[1]
        criticality = parts[2]
        file_id = parts[3]
        segment = parts[5]
    
    return person_id, category, criticality, file_id, segment

# Get the path of the processed data folder
processed_results_folder = Path('C:/Users/bayer/Documents/HCI/Data/Processed_data')  

# Get all feature files
processed_files = {file.stem: file for file in processed_results_folder.rglob("*features.csv")}

# Get YOLO, Gaze, and Combi feature files
yolo_files = {name: file for name, file in processed_files.items() if "yolo" in name.lower()}
gaze_files = {name: file for name, file in processed_files.items() if "gaze" in name.lower()}
combi_files = {name: file for name, file in processed_files.items() if "combi" in name.lower()}

# Initialize an empty list to store the final rows of the dataset
final_data = []

# Process each combination of Gaze, Combi, and YOLO files
for gaze_name, gaze_file in gaze_files.items():
    # Extract the key columns from the Gaze filename
    person_id, category, criticality, file_id, segment = extract_key_columns(gaze_name)
    
    # Process the Gaze features
    gaze_data = pd.read_csv(gaze_file)
    gaze_features = gaze_data.values.flatten().tolist()
    gaze_columns = gaze_data.columns.tolist()

    # Get corresponding Combi file based on the gaze file name
    combi_name = gaze_name.replace("gaze", "combi")
    combi_features = []
    combi_columns = []
    if combi_name in combi_files:
        combi_data = pd.read_csv(combi_files[combi_name])
        combi_features = combi_data.values.flatten().tolist()
        combi_columns = combi_data.columns.tolist()

    # Get the YOLO file corresponding to this Gaze file
    yolo_name = "_".join(gaze_name.split('_')[1:]).replace("gaze", "yolo")
    yolo_features = []
    yolo_columns = []
    if yolo_name in yolo_files:
        yolo_data = pd.read_csv(yolo_files[yolo_name])
        yolo_features = yolo_data.values.flatten().tolist()
        yolo_columns = yolo_data.columns.tolist()

    # Combine all features (YOLO, Gaze, and Combi)
    row = [person_id, category, criticality, file_id, segment] + yolo_features + gaze_features + combi_features
    final_data.append(row)

# Create the final dataset as a DataFrame
columns = ['person_id', 'category', 'criticality', 'file_id', 'segment'] + yolo_columns + gaze_columns + combi_columns

# Create the DataFrame
final_dataset = pd.DataFrame(final_data, columns=columns)

# Save the final dataset to a CSV file in the current directory (using pathlib)
current_directory = Path(__file__).parent
final_dataset.to_csv(current_directory / 'features_dataset.csv', index=False)

# Print a message to confirm the location of the saved file
print("Final dataset saved to:", current_directory / 'features_dataset.csv')
