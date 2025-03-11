import os
import csv

def read_csv_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            print(f"\nReading file: {os.path.basename(file_path)}\n")
            reader = csv.reader(file)
            for row in reader:
                print(row)  # Print each row of the CSV file
    except Exception as e:
        print(f"Error reading file: {e}")