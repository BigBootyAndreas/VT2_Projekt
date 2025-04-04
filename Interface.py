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

    # List and select the main folder (IMU Data or Acoustic Data)
    selected_folder = list_folders(dir)
    
    if not selected_folder:
        print("No valid folder selected. Exiting.")
        return

    # Find the subdirectory within the selected folder
    subdirectory_path = list_subfolders(selected_folder)
    
    if not subdirectory_path: 
        selected_folder2 = list_folders(dir2)
        if selected_folder2:
            subdirectory_path2 = list_subfolders(selected_folder2)
        else:
            subdirectory_path2 = None
    else:
        subdirectory_path2 = None

    if not subdirectory_path and not subdirectory_path2:
        print(f"Subdirectory not found in the base directories.")
        exit()

    selected_path = subdirectory_path if subdirectory_path else subdirectory_path2
    
    # List and select files from the chosen folder
    selected_file = list_and_select_files(selected_path)

    if selected_file:
        print(f"You selected: {selected_file}")
        
        # Determine folder type (IMU or Acoustic) based on selection
        folder_name = os.path.basename(selected_folder)
        folder_choice = '1' if 'IMU' in folder_name else '2'
        
        df = read_csv_file(selected_file, folder_choice)
        
        if df is not None:
            if folder_choice == '1':
                imu_processing(df)
            elif folder_choice == '2':
                stft_result, sr = acoustic_processing(df)
                
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
