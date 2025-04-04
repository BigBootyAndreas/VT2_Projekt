import os
from user_dir_detection import dir  # Assuming `dir` is the base directory

# Function to list folders in a given directory
def list_folders(directory):
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return None

    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    # Prioritize IMU Data before Acoustic Data
    sorted_folders = sorted(folders, key=lambda x: (x != "IMU Data", x))

    if not sorted_folders:
        print("No folders found.")
        return None

    print("Available folders:")
    for idx, folder in enumerate(sorted_folders):
        print(f"{idx + 1}. {folder}")

    while True:
        try:
            choice = int(input("Enter the number corresponding to the folder: ")) - 1
            if 0 <= choice < len(sorted_folders):
                return os.path.join(directory, sorted_folders[choice])
            else:
                print("Invalid selection. Please choose a valid folder number.")
        except ValueError:
            print("Invalid input. Please enter a number.")


# Function to list and select a subfolder
def list_subfolders(folder_path):
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    
    if not subfolders:
        print("No subfolders found.")
        return None
    
    print("Available subfolders:")
    for idx, subfolder in enumerate(subfolders):
        print(f"{idx + 1}. {subfolder}")
    
    while True:
        try:
            choice = int(input("Enter the number corresponding to the subfolder: ")) - 1
            if 0 <= choice < len(subfolders):
                return os.path.join(folder_path, subfolders[choice])
            else:
                print("Invalid selection. Please choose a valid subfolder number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Function to list and select files from a folder
def list_and_select_files(folder_path, file_extensions=(".csv", ".npz")):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return None
    
    files = [f for f in os.listdir(folder_path) if f.endswith(file_extensions)]
    
    if not files:
        print(f"No {file_extensions} files found in the folder.")
        return None
    
    print("Files available:")
    for idx, file in enumerate(files):
        print(f"{idx + 1}. {file}")
    
    while True:
        try:
            choice = int(input("Enter the number corresponding to the file: ")) - 1
            if 0 <= choice < len(files):
                return os.path.join(folder_path, files[choice])
            else:
                print("Invalid selection. Please choose a valid file number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Main execution
if __name__ == "__main__":
    base_folder = dir  # Base directory from user_dir_detection
    selected_folder = list_folders(base_folder)
    if selected_folder:
        selected_subfolder = list_subfolders(selected_folder)
        if selected_subfolder:
            selected_file = list_and_select_files(selected_subfolder)
            if selected_file:
                print(f"Selected file: {selected_file}")
