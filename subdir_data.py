import os
from user_dir_detection import dir, dir2  # Import both directories

# Function to list folders in a given directory
def list_folders(directory):
    if not os.path.exists(directory):
        #print(f"Directory not found: {directory}")
        return []

    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    
    # Return all folders without prioritizing one over the other
    return [(directory, folder) for folder in folders]

# Function to list and select a folder from multiple directories
def select_folder():
    # Combine folders from dir and dir2
    all_folders = list_folders(dir) + list_folders(dir2)

    if not all_folders:
        print("No folders found in either directory.")
        return None

    print("Available folders:")
    for idx, (base_path, folder_name) in enumerate(all_folders):
        print(f"{idx + 1}. {folder_name}")  # Print only folder names

    # Loop until a valid folder is selected
    while True:
        try:
            choice = int(input("Enter the number corresponding to the folder: ")) - 1
            if 0 <= choice < len(all_folders):
                base_path, folder_name = all_folders[choice]
                return os.path.join(base_path, folder_name)
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
    selected_folder = select_folder()
    if selected_folder:
        selected_subfolder = list_subfolders(selected_folder)
        if selected_subfolder:
            selected_file = list_and_select_files(selected_subfolder)
            if selected_file:
                print(f"Selected file: {selected_file}")
