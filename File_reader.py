import pandas as pd

def read_csv_file(file_path, folder_choice):
    try:
        if folder_choice == '1':
            # For IMU Data skip the first row
            df = pd.read_csv(file_path)
        elif folder_choice == '2':
            # For Acoustic Data, read different columns
            df = pd.read_csv(file_path,skiprows=[0], usecols=[1,2], names=["Time","Amplitude"])
        else:
            print("Invalid folder choice.")
            return None  # Return None if the folder choice is invalid

        # Print the dataframe to check the content
        print("DataFrame loaded successfully")
        #print(df.head()) 
    
        return df  
    
    except Exception as e:
        print(f"Error reading file: {e}")
        return None  # Return None if there's an error
