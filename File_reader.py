import pandas as pd

def read_csv_file(file_path):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path,usecols=[1,2],skiprows=1)
    
        # Print the filtered rows
        print(df)

    except Exception as e:
        print(f"Error reading file: {e}")
