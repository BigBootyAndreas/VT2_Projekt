import os 
from user_dir_detection import dir
from npz_cleaner import npz_rm
from csv_compiler import csv_compiler


# Function to find the subdirectory under the base directory
def find_subdirectory(dir, folder_name):
    subdirectory_path = os.path.join(dir, folder_name)
    if os.path.exists(subdirectory_path):
        return subdirectory_path
    else:
        print(f"Subdirectory '{folder_name}' not found in the base directory.")
        return None


# Function to list and select files from a folder
def list_and_select_files(folder_path, file_extension=".csv"):

    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return None

    # List all files with the specified extension
    files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]

    if not files:
        print(f"No {file_extension} files found in the folder.")
        return None

    print(f"Files available in {os.path.basename(folder_path)}:")
    for idx, file in enumerate(files):
        print(f"{idx + 1}. {file}")

    # Let the user select a file
    while True:
        try:
            choice = int(input("Enter the number corresponding to the file: ")) - 1
            if 0 <= choice < len(files):
                selected_file = files[choice]
                return os.path.join(folder_path, selected_file)
            else:
                print("Invalid selection. Please choose a valid file number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

