import csv
import random
import names
import time

#Text colors
CYELLOW = '\033[93m'
CGREEN = '\033[92m'
CBLUE = '\033[94m'
CRED = '\033[91m'
CBLINK = '\033[5m'
CEND = '\033[0m'

def record_imu(duration, filename):

    print(CYELLOW + "IMU initialized" + CEND)

    # Define the CSV file name
    csv_filename = "segments/imu/" + filename + ".csv"

    # Define a list of sample cities
    cities = ["New York", "Los Angeles", "Chicago", "Houston", "Miami", "San Francisco", "Dallas", "Seattle"]

    # Open the file in append mode
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row if the file is empty
        file.seek(0)
        if file.tell() == 0:
            writer.writerow(["Name", "Age", "City"])
        
        start_time = time.time()
        while time.time() - start_time < duration:
            name = names.get_full_name()
            age = random.randint(18, 80)
            city = random.choice(cities)
            writer.writerow([name, age, city])
            time.sleep(0.1)  # Adjust the sleep time to control writing speed

    print(f"IMU segment: '{csv_filename}' recorded and saved")