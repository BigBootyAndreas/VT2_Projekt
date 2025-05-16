import os 
import pandas as pd
import glob
from user_dir_detection import *
from subdir_data import list_and_select_files, list_folders, list_subfolders
from File_reader import *
from IMU_data import imu_processing
from Acoustic_data import acoustic_processing

# Import the tool wear predictor components
from tool_wear_predictor import ToolWearPredictor, train_model_from_recordings

def main():
    if dir:
        print(f"Welcome to TCM system {username}")
    else:
        print("Invalid user, currently having problem defining user name.")
        return

    # Try to get folders from dir or dir2
    folder_list = list_folders(dir) + list_folders(dir2)  # Combine both lists
    if not folder_list:
        print("No valid folder selected. Exiting.")
        return

    print("Available Folders:")
    for idx, (base_path, folder_name) in enumerate(folder_list):
        print(f"{idx + 1}. {folder_name}")


    while True:
        try:
            choice = int(input("Enter the number corresponding to the folder: ")) - 1
            if 0 <= choice < len(folder_list):
                base_path, folder_name = folder_list[choice]
                selected_folder = os.path.join(base_path, folder_name)
                print(f"Selected folder: {selected_folder}")
                break                # Determine folder type (IMU, Acoustic, or Drill) based on name
                folder_name_check = os.path.basename(selected_folder)
                
                if 'IMU' in folder_name_check:
                    folder_choice = '1'  # IMU data
                elif 'Acoustic' in folder_name_check:
                    folder_choice = '2'  # Acoustic data
                elif 'Drill' in folder_name_check:
                    folder_choice = '3'  # Drill data
                else:
                    print("Unknown folder type. Exiting.")
                    return
            else:
                print("Invalid selection. Please choose a valid folder number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

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
            # Display menu with ML options
            print("\nOptions:")
            print("1. Process data (Original functionality)")
            print("2. Train ML model for tool wear prediction")
            print("3. Predict tool wear with existing model")
            
            choice = input("Enter your choice (1/2/3): ")
            
            if choice == '1':
                # Original functionality
                if folder_choice == '1':  # IMU data
                    print("Choose the type of plot:")
                    print("1. Raw Data")
                    print("2. Power Spectral Density (PSD)")
                    while True:
                        try:
                            plot_choice = int(input("Enter your choice (1 or 2): "))
                            if plot_choice == 1:
                                imu_processing(df, plot_type="raw")  # Plot raw data
                                break
                            elif plot_choice == 2:
                                imu_processing(df, plot_type="psd")  # Plot PSD
                                break
                            else:
                                print("Invalid choice. Please enter 1 or 2.")
                        except ValueError:
                            print("Invalid input. Please enter a number.")
                elif folder_choice == '2':  # Acoustic data
                    stft_result, sr = acoustic_processing(df)

                    do_advanced = input("Would you like to perform advanced spectral analysis? (y/n): ")
                    if do_advanced.lower() == 'y':
                        from Acoustic_data import advanced_acoustic_analysis
                        advanced_acoustic_analysis(df, stft_result, sr)
            
            elif choice == '2':
                # Train ML model
                print("Training a new tool wear prediction model...")
                data_dir = os.path.dirname(selected_file)
                
                # Create models directory if it doesn't exist
                models_dir = "models"
                os.makedirs(models_dir, exist_ok=True)
                
                # Tool selection for different lifetime values
                print("\nSelect tool type:")
                print("1. Reamer (200 min lifetime)")
                print("2. E-mill (100 min lifetime)")
                
                tool_type = ""
                while tool_type not in ["1", "2"]:
                    tool_type = input("Enter your choice (1/2): ")
                    if tool_type not in ["1", "2"]:
                        print("Invalid selection. Please choose 1 or 2.")
                
                # Set default lifetime based on tool type
                if tool_type == "1":
                    default_lifetime = 12000  # 200 minutes in seconds
                    tool_name = "reamer"
                else:
                    default_lifetime = 6000   # 100 minutes in seconds
                    tool_name = "emill"
                
                # Ask for confirmation or custom lifetime value
                tool_lifetime = 0
                while tool_lifetime <= 0:
                    try:
                        user_input = input(f"Enter tool lifetime in seconds (default {default_lifetime} for {tool_name}), or press Enter for default: ")
                        if user_input.strip() == "":
                            tool_lifetime = default_lifetime
                        else:
                            tool_lifetime = float(user_input)
                        if tool_lifetime <= 0:
                            print("Tool lifetime must be positive.")
                    except ValueError:
                        print("Please enter a valid number.")
                
                # Create tool-specific models directory
                tool_models_dir = os.path.join(models_dir, tool_name)
                os.makedirs(tool_models_dir, exist_ok=True)
                
                print(f"Training in progress for {tool_name} (this may take a while)...")
                predictor = train_model_from_recordings(
                    data_dir,
                    output_dir=tool_models_dir,
                    tool_lifetime=tool_lifetime
                )
                
                if predictor and predictor.model_loaded:
                    print(f"Model trained successfully for {tool_name}!")
                else:
                    print("Error training model. See above for details.")
            
            elif choice == '3':
                # Predict tool wear
                print("Tool wear prediction...")
                
                # Check if models directory exists
                models_dir = "models"
                if not os.path.exists(models_dir):
                    print("No 'models' directory found. Please train a model first.")
                    return
                
                # First, check for tool-specific subdirectories
                tool_dirs = [d for d in os.listdir(models_dir) 
                           if os.path.isdir(os.path.join(models_dir, d)) 
                           and d in ['reamer', 'emill']]
                
                if tool_dirs:
                    print("\nSelect tool type:")
                    for i, tool in enumerate(tool_dirs):
                        print(f"{i+1}. {tool.capitalize()}")
                    
                    tool_idx = -1
                    while tool_idx < 0 or tool_idx >= len(tool_dirs):
                        try:
                            tool_idx = int(input("Select tool number: ")) - 1
                            if tool_idx < 0 or tool_idx >= len(tool_dirs):
                                print("Invalid selection. Please choose a valid tool number.")
                        except ValueError:
                            print("Please enter a valid number.")
                    
                    selected_tool = tool_dirs[tool_idx]
                    models_dir = os.path.join(models_dir, selected_tool)
                
                # Get available models in the selected directory
                models = glob.glob(os.path.join(models_dir, "*_model.pkl"))
                
                if not models:
                    print(f"No trained models found in {models_dir}. Train a model first.")
                    return
                
                print("\nAvailable models:")
                for i, model_path in enumerate(models):
                    model_name = os.path.basename(model_path).replace("_model.pkl", "").replace("_", " ").title()
                    print(f"{i+1}. {model_name}")
                
                model_idx = -1
                while model_idx < 0 or model_idx >= len(models):
                    try:
                        model_idx = int(input("Select model number: ")) - 1
                        if model_idx < 0 or model_idx >= len(models):
                            print("Invalid selection. Please choose a valid model number.")
                    except ValueError:
                        print("Please enter a valid number.")
                
                model_path = models[model_idx]
                base_name = model_path.replace("_model.pkl", "")
                scaler_path = f"{base_name}_scaler.pkl"
                features_path = f"{base_name}_features.pkl"
                
                predictor = ToolWearPredictor(model_path, scaler_path, features_path)
                
                if not predictor.model_loaded:
                    print("Error loading model")
                    return
                
                # Get paired file
                other_file = None
                if folder_choice == '1':  # IMU file selected, need acoustic file
                    # Try to find a matching acoustic file in the same directory
                    data_dir = os.path.dirname(selected_file)
                    acoustic_files = glob.glob(os.path.join(data_dir, "*acoustic*.csv"))
                    
                    if not acoustic_files:
                        acoustic_files = glob.glob(os.path.join(data_dir, "*.*"))
                        print("\nPotential acoustic files in the same directory:")
                        for i, f in enumerate(acoustic_files):
                            print(f"{i+1}. {os.path.basename(f)}")
                        
                        file_idx = -1
                        while file_idx < 0 or file_idx >= len(acoustic_files):
                            try:
                                file_idx = int(input("Select acoustic file number (or 0 to enter a different path): ")) - 1
                                if file_idx == -1:  # User chose 0
                                    acoustic_path = input("Enter path to acoustic data file: ")
                                    if os.path.exists(acoustic_path):
                                        break
                                    else:
                                        print("File not found")
                                        return
                                elif file_idx < 0 or file_idx >= len(acoustic_files):
                                    print("Invalid selection")
                            except ValueError:
                                print("Please enter a valid number")
                        
                        if file_idx >= 0:
                            acoustic_path = acoustic_files[file_idx]
                    else:
                        # If only one acoustic file, use it automatically
                        if len(acoustic_files) == 1:
                            acoustic_path = acoustic_files[0]
                            print(f"Using acoustic file: {os.path.basename(acoustic_path)}")
                        else:
                            print("\nSelect acoustic file:")
                            for i, f in enumerate(acoustic_files):
                                print(f"{i+1}. {os.path.basename(f)}")
                            
                            file_idx = -1
                            while file_idx < 0 or file_idx >= len(acoustic_files):
                                try:
                                    file_idx = int(input("Select file number: ")) - 1
                                    if file_idx < 0 or file_idx >= len(acoustic_files):
                                        print("Invalid selection")
                                except ValueError:
                                    print("Please enter a valid number")
                            
                            acoustic_path = acoustic_files[file_idx]
                    
                    print(f"Analyzing tool wear using:\n- IMU: {os.path.basename(selected_file)}\n- Acoustic: {os.path.basename(acoustic_path)}")
                    wear, time, replace = predictor.analyze_recording(acoustic_path, selected_file)
                
                else:  # Acoustic file selected, need IMU file
                    # Try to find a matching IMU file in the same directory
                    data_dir = os.path.dirname(selected_file)
                    imu_files = glob.glob(os.path.join(data_dir, "*IMU*.csv"))
                    
                    if not imu_files:
                        imu_files = glob.glob(os.path.join(data_dir, "*.*"))
                        print("\nPotential IMU files in the same directory:")
                        for i, f in enumerate(imu_files):
                            print(f"{i+1}. {os.path.basename(f)}")
                        
                        file_idx = -1
                        while file_idx < 0 or file_idx >= len(imu_files):
                            try:
                                file_idx = int(input("Select IMU file number (or 0 to enter a different path): ")) - 1
                                if file_idx == -1:  # User chose 0
                                    imu_path = input("Enter path to IMU data file: ")
                                    if os.path.exists(imu_path):
                                        break
                                    else:
                                        print("File not found")
                                        return
                                elif file_idx < 0 or file_idx >= len(imu_files):
                                    print("Invalid selection")
                            except ValueError:
                                print("Please enter a valid number")
                        
                        if file_idx >= 0:
                            imu_path = imu_files[file_idx]
                    else:
                        # If only one IMU file, use it automatically
                        if len(imu_files) == 1:
                            imu_path = imu_files[0]
                            print(f"Using IMU file: {os.path.basename(imu_path)}")
                        else:
                            print("\nSelect IMU file:")
                            for i, f in enumerate(imu_files):
                                print(f"{i+1}. {os.path.basename(f)}")
                            
                            file_idx = -1
                            while file_idx < 0 or file_idx >= len(imu_files):
                                try:
                                    file_idx = int(input("Select file number: ")) - 1
                                    if file_idx < 0 or file_idx >= len(imu_files):
                                        print("Invalid selection")
                                except ValueError:
                                    print("Please enter a valid number")
                            
                            imu_path = imu_files[file_idx]
                    
                    print(f"Analyzing tool wear using:\n- Acoustic: {os.path.basename(selected_file)}\n- IMU: {os.path.basename(imu_path)}")
                    wear, time, replace = predictor.analyze_recording(selected_file, imu_path)
                
                if wear is not None:
                    print("\nPrediction Results:")
                    print(f"- Predicted tool wear: {wear:.1%}")
                    print(f"- Estimated remaining time: {predictor._format_time(time)}")
                    print(f"- Tool needs replacement: {'YES' if replace else 'No'}")
                    
                    # Ask if user wants to see prediction history plot
                    if predictor.predictions and len(predictor.predictions) > 1:
                        view_history = input("\nView prediction history plot? (y/n): ")
                        if view_history.lower() == 'y':
                            predictor.plot_wear_history()
                else:
                    print("Prediction failed")
            
            else:
                print("Invalid choice")
                
        else:
            print("Error: Dataframe is empty or could not be loaded.")
    else:
        print("No file was selected.")

if __name__ == "__main__":
    main()