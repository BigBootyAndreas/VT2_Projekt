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
        return

    # Try to get folders from dir or dir2
    folder_list = list_folders(dir) or list_folders(dir2)
    if not folder_list:
        print("No valid folder selected. Exiting.")
        return

    # Auto-select the first folder found
    base_path, folder_name = folder_list[0]
    selected_folder = os.path.join(base_path, folder_name)

    print(f"Selected folder: {selected_folder}")

    # Try to get subfolders
    subdirectory_path = list_subfolders(selected_folder)

    # If no subfolders found, try switching to dir2
    if not subdirectory_path:
        folder_list2 = list_folders(dir2)
        if folder_list2:
            base_path2, folder_name2 = folder_list2[0]
            selected_folder2 = os.path.join(base_path2, folder_name2)
            subdirectory_path2 = list_subfolders(selected_folder2)
        else:
            subdirectory_path2 = None
    else:
        subdirectory_path2 = None

    # Final path to continue with
    selected_path = subdirectory_path if subdirectory_path else subdirectory_path2

    if not selected_path:
        print("Subdirectory not found in the base directories.")
        return

    # List and select files from the chosen subfolder
    selected_file = list_and_select_files(selected_path)

    if selected_file:
        print(f"You selected: {selected_file}")

        # Determine folder type (IMU or Acoustic) based on name
        folder_name_check = os.path.basename(selected_folder)
        folder_choice = '1' if 'IMU' in folder_name_check else '2'

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
