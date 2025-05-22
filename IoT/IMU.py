import csv
import time
import smbus2
import time

#Text colors
CYELLOW = '\033[93m'
CGREEN = '\033[92m'
CBLUE = '\033[94m'
CRED = '\033[91m'
CBLINK = '\033[5m'
CEND = '\033[0m'

bus = smbus2.SMBus(1)  # I2C Bus 1
device_address = 0x1D  # MMA8451 address

# set ACTIVE mode
bus.write_byte_data(device_address, 0x2A, 0b00011101)  # CTRL_REG1: Active mode

# Set MMA8451 to Standby mode (required before changing settings)
bus.write_byte_data(device_address, 0x2A, 0x00)

bus.write_byte_data(device_address, 0x0E, 0x00) #2g setting 

# Activate the sensor
bus.write_byte_data(device_address, 0x2A, 0x09)  #0x09 sets sample rate at 400 Hz

def read_acceleration():
    # Read 6 bytes of data (OUT_X_MSB -> OUT_Z_LSB)
    data = bus.read_i2c_block_data(device_address, 0x01, 6)

    # Convert the data (12-bit values)
    x = ((data[0] << 8) | data[1]) >> 4
    y = ((data[2] << 8) | data[3]) >> 4
    z = ((data[4] << 8) | data[5]) >> 4

    # Convert 12-bit signed values
    if x > 2047: x -= 4096
    if y > 2047: y -= 4096
    if z > 2047: z -= 4096

    # Scale
    sensitivity = 1024.0  # 2g range: 1024 LSB/g
    x_g = x / sensitivity
    y_g = y / sensitivity
    z_g = z / sensitivity

    return x_g, y_g, z_g

def is_data_ready():

    # Read the STATUS register (0x00)
    status = bus.read_byte_data(device_address, 0x00)

    # Check the data-ready bit (bit 0) to see if new data is available
    return status & 0x01 != 0

def record_imu(duration, filename):

    id = 0
    
    print(CYELLOW + "IMU initialized" + CEND)

    # Define the CSV file name
    csv_filename = "segments/imu/" + filename + ".csv"

    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "epoch", "timestamp", "X (g)", "Y (g)", "Z (g)"])  # CSV Header
    
        start_time = time.time()

        while time.time() - start_time < duration:

            # Wait for data to be ready
            while not is_data_ready():
                time.sleep(0.0005)  # Sleep for 0.5ms before checking again

            #Epoch
            epoch = int(time.time() * 1000)

            # Read new acceleration data
            x, y, z = read_acceleration()

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")  # Get current time
        
            # Write data to CSV file
            writer.writerow([id, epoch, timestamp, x, y, z])
            file.flush()  # Ensure data is written to disk

            # Update ID
            id += 1

            time.sleep(1/400)  # Ensures the sample rate of 400 Hz, with no duplicates

        print(f"IMU segment {filename} saved")
