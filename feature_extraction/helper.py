from pathlib import Path
import zipfile
import shutil
# Function to unzip a folder
def unzip_folder(zip_path: Path, extract_to: Path) -> None:
    """
    Extract the contents of a zip file into a specified directory.

    Args:
    - zip_path (Path): Path to the zip file to be extracted.
    - extract_to (Path): Path to the directory where the contents will be extracted.

    Returns:
    - None: Extracts files into the specified directory.
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Function to zip a folder back
def zip_folder(folder_path: Path, zip_path: Path) -> None:
    """
    Create a zip file from the contents of a specified folder (without including the folder itself),
    and store it as 'Data.zip' in the parent directory.

    Args:
    - folder_path (Path): Path to the folder whose contents will be compressed into a zip file.
    - zip_path (Path): Path where the zip file will be saved (e.g., 'HCI/Data.zip').

    Returns:
    - None: Creates a zip file at the specified path.
    """
    # Ensure we are zipping the contents of the folder, not the folder itself
    folder_path_str = folder_path.as_posix()

    # Create the zip file with the contents of the folder (but not the folder itself)
    shutil.make_archive(zip_path.with_suffix('').as_posix(), 'zip', folder_path_str, '.')
