import pandas as pd

def read_csv_file(file_path, folder_choice):
    try:
        if folder_choice == '1':
            # For IMU Data, read specific rows and columns
            df = pd.read_csv(file_path, usecols=[1,2,3,4], skiprows=1)
        elif folder_choice == '2':
            # For Acoustic Data, read different rows and columns
            df = pd.read_csv(file_path, usecols=[1,2])
        else:
            print("Invalid folder choice.")
            return

        # Print the filtered rows
        print(df)

    except Exception as e:
        print(f"Error reading file: {e}")