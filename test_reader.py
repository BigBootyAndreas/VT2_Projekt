import os
import glob

def read_text_file(directory):
    """Finds and reads the first .txt file in the directory."""
    txt_files = glob.glob(os.path.join(directory, '*.txt'))  # Get all text files

    if not txt_files:
        print("No text files found in the directory.")
        return
    
    file_path = txt_files[0]  # Select the first text file found

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            print(f"\nReading file: {os.path.basename(file_path)}\n")
            print(file.read())  # Print file content
    except Exception as e:
        print(f"Error reading file: {e}")
