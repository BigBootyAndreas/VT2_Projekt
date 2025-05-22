import os
import csv

# Folders
base_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(base_dir, '20khz')
output_folder = os.path.join(base_dir, '400hz')

# Create output folder
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):

        dc_source = os.path.join(input_folder, filename)
        dc_destination = os.path.join(output_folder, f'{filename}_400hz.csv')

        # Open the input file and read data
        with open(dc_source, mode="r", newline="") as infile:
            reader = csv.reader(infile)
            header = next(reader)  # Read the header

            # Read and downsample: keep every 3rd row
            downsampled_rows = [row for index, row in enumerate(reader) if index % 50 == 0]

        # Write the downsampled data to a new file
        with open(dc_destination, mode="w", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)  # Write header
            writer.writerows(downsampled_rows)  # Write downsampled data

        print(f"Downsampled data written to: {dc_destination}")
