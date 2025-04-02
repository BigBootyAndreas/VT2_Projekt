import os
import pandas as pd
from user_dir_detection import *
from subdir_data import list_and_select_files, list_folders, list_subfolders
from File_reader import *
from IMU_data import imu_processing
from Acoustic_data import acoustic_processing

def main():
    if dir:
        print(f"Welcome to TCM system {username}")
    else:
        print("Invalid user, currently having problem defining user name.")

    # Ask the user to choose between IMU and Acoustic folders
    print("Choose a folder to open:")
    print("1. IMU Data")
    print("2. Acoustic Data")
    folder_choice = input("Enter your choice (1 or 2): ").strip()

    if folder_choice == '1':
        folder_name = 'IMU Data'
    elif folder_choice == '2':
        folder_name = 'Acoustic Data'
    else:
        print("Invalid choice. Exiting.")
        return

    # Find the subdirectory
    selected_folder = list_folders(dir)
    if selected_folder:
        subdirectory_path = list_subfolders(selected_folder)
    else:
        subdirectory_path = None

    # If not found in the first directory, try the second directory
    if not subdirectory_path:
        selected_folder2 = list_folders(dir2)
        if selected_folder2:
            subdirectory_path2 = list_subfolders(selected_folder2)
        else:
            subdirectory_path2 = None
    else:
        subdirectory_path2 = None

    # If neither directory contains the subfolder, exit
    if not subdirectory_path and not subdirectory_path2:
        print(f"Subdirectory '{folder_name}' not found in the base directories.")
        exit()

    # Choose the valid path
    selected_path = subdirectory_path if subdirectory_path else subdirectory_path2
    
    # List and select files from the chosen folder
    selected_file = list_and_select_files(selected_path)

    # Print the selected file
    if selected_file:
        print(f"You selected: {selected_file}")

        df = read_csv_file(selected_file, folder_choice)
        
        if df is not None:  # Ensure df is valid before processing
            if folder_choice == '1':
                # Process IMU data
                imu_processing(df)
                
            elif folder_choice == '2':
                # Process Acoustic data
                stft_result, sr = acoustic_processing(df)
                
                # Ask if they want advanced analysis
                do_advanced = input("Would you like to perform advanced spectral analysis? (y/n): ")
                if do_advanced.lower() == 'y':
                    from Acoustic_data import advanced_acoustic_analysis
                    advanced_acoustic_analysis(df, stft_result, sr)
        else:
            print("Error: Dataframe is empty or could not be loaded.")
    else:
        print("No file was selected.")

if __name__ == "__main__":
    main()
