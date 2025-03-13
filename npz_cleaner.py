import os
import time

def npz_rm(selected_file):
    folder_path = os.path.dirname(selected_file)  # Get the folder path

    # List all .npz files in the folder
    files = os.listdir(folder_path)
    npz_files = [file for file in files if file.endswith(".npz")]

    if not npz_files:
        print("No .npz files found to delete.")
        return

    for filename in npz_files:
        full_path = os.path.join(folder_path, filename)
        
        try:
            os.remove(full_path)
            print(f"Deleted {filename}")
            time.sleep(0.1)
        except Exception as e:
            print(f"Error deleting {filename}: {e}")
