import threading
from IMU import record_imu
from MIC import record_audio

def main():

    COUNTER_FILE = "counter.txt"

    with open(COUNTER_FILE, "r") as f:
        counter, data_set = map(int, f.read().split())

    # Check if 12 runs is reached and update the dataset if true
    if counter % 12 == 0:
        data_set += 1
        print(f"Run {counter}: Data{data_set}")
    else:
        print(f"Run {counter}: Data{data_set}")

    samplerate = 16000
    channels = 1
    duration = 10
    filename = f"sample_{counter}"

    # Create threads for concurrent recording
    audio_thread = threading.Thread(target=record_audio, args=(duration, samplerate, channels, filename))
    imu_thread = threading.Thread(target=record_imu, args=(duration, filename))

    # Start both threads
    audio_thread.start()
    imu_thread.start()

    # Wait for both threads to complete
    audio_thread.join()
    imu_thread.join()

    counter += 1
    with open(COUNTER_FILE, "w") as f:
        f.write(f"{counter} {data_set}")

if __name__ == "__main__":
    main()
