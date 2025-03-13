import os
import numpy as np

def csv_compiler(selected_file):
    folder_path = os.path.dirname(selected_file)  # Get the folder path

    files = os.listdir(folder_path)
    npz_files = [file for file in files if file.endswith(".npz")]

    if not npz_files:
        print("No .npz files found in the directory.")
        return

    for filename in npz_files:
        full_path = os.path.join(folder_path, filename)
        
        try:
            loaded = np.load(full_path)
            print(f"Contents of {filename}: {loaded.files}")  # Check keys inside npz

            if "a" not in loaded:
                print(f"Warning: 'a' key not found in {filename}. Skipping.")
                continue

            data = loaded["a"]

            # Convert to 2D if needed
            if data.ndim > 2:
                print(f"Unexpected shape {data.shape} in {filename}. Flattening data.")
                data = data.reshape(-1, data.shape[-1])

            csv_filename = filename.replace(".npz", ".csv")
            csv_filepath = os.path.join(folder_path, csv_filename)

            # Save with UTF-8 encoding to avoid decoding errors
            np.savetxt(csv_filepath, data, delimiter=",", fmt="%s", encoding="utf-8-sig",
                       header="Timestamp, Time (s), Amplitude", comments="")

            print(f"Converted {filename} to {csv_filename}")
        except Exception as e:
            print(f"Error converting {filename}: {e}")
