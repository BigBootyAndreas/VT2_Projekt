import os
import glob
import csv

def read_csv_file(directory):
    #Finds and reads the first csv file in the directory.
    csv_files = glob.glob(os.path.join(directory, '*.csv'))

    if not csv_files:
        print("No csv files found in the directory.")
        return
    
    file_path = csv_files[0]  # Select the first text file found

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            print(f"\nReading file: {os.path.basename(file_path)}\n")
            print(file.read())  # Print file content
    except Exception as e:
        print(f"Error reading file: {e}")
