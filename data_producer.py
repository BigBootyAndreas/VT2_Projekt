import socket
import pandas as pd
import json
import time
import threading

# Load data
imu_df = pd.read_csv("IMU_data.csv", names=["Index", "Accel_X", "Accel_Y", "Accel_Z", "Timestamp"])
acoustic_df = pd.read_csv("acoustic_data.csv", delimiter="\t", names=["RawData"])
acoustic_df[["Timestamp", "Frequency", "Amplitude"]] = acoustic_df["RawData"].str.split(',', expand=True)
acoustic_df.drop(columns=["RawData", "Timestamp"], inplace=True)
acoustic_df["Frequency"] = acoustic_df["Frequency"].astype(float)
acoustic_df["Amplitude"] = acoustic_df["Amplitude"].astype(float)

# Sampling rates
imu_hz = 800
acoustic_hz = 16000
imu_interval = 1 / imu_hz
acoustic_batch_interval = 0.01  # 10ms batches
acoustic_batch_size = int(acoustic_hz * acoustic_batch_interval)  # 160 samples

HOST = "0.0.0.0"
PORT = 65432

def imu_sender(conn):
    index = 0
    next_time = time.perf_counter()
    while index < len(imu_df):
        now = time.perf_counter()
        if now >= next_time:
            row = imu_df.iloc[index]
            packet = {
                "type": "imu",
                "Accel_X": row["Accel_X"],
                "Accel_Y": row["Accel_Y"],
                "Accel_Z": row["Accel_Z"]
            }
            conn.sendall((json.dumps(packet) + "\n").encode())
            index += 1
            next_time += imu_interval
        else:
            time.sleep(0.0005)

def acoustic_sender(conn):
    index = 0
    next_time = time.perf_counter()
    while index < len(acoustic_df):
        now = time.perf_counter()
        if now >= next_time:
            batch = acoustic_df.iloc[index:index+acoustic_batch_size]
            packet = {
                "type": "acoustic_batch",
                "Amplitude": batch["Amplitude"].tolist()
            }
            conn.sendall((json.dumps(packet) + "\n").encode())
            index += acoustic_batch_size
            next_time += acoustic_batch_interval
        else:
            time.sleep(0.0005)

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            threading.Thread(target=imu_sender, args=(conn,), daemon=True).start()
            acoustic_sender(conn)

if __name__ == "__main__":
    main()
