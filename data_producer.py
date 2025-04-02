import socket
import pandas as pd
import json
import time

HOST = '127.0.0.1'
PORT = 65432

imu_df = pd.read_csv("IMU_data.csv", names=["Index", "Accel_X", "Accel_Y", "Accel_Z", "Timestamp"])
acoustic_df = pd.read_csv("acoustic_data.csv", delimiter="\t", names=["RawData"])

# Split acoustic data
acoustic_df[["Timestamp", "Frequency", "Amplitude"]] = acoustic_df["RawData"].str.split(',', expand=True)
acoustic_df.drop(columns=["RawData", "Timestamp"], inplace=True)
acoustic_df["Frequency"] = acoustic_df["Frequency"].astype(float)
acoustic_df["Amplitude"] = acoustic_df["Amplitude"].astype(float)

sampling_imu = 800
imu_interval = 1 / sampling_imu

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print("Server is running, waiting for connection...")

    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        imu_index = 0
        acoustic_index = 0

        while imu_index < len(imu_df) and acoustic_index < len(acoustic_df):
            data_packet = {
                'imu': imu_df.iloc[imu_index].to_dict(),
                'acoustic': acoustic_df.iloc[acoustic_index].to_dict()
            }
            conn.sendall(json.dumps(data_packet).encode('utf-8') + b'\n')

            imu_index += 1
            acoustic_index += 1

            time.sleep(imu_interval)
