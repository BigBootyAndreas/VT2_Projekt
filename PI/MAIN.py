import threading
from IMU import record_imu
from MIC import record_audio
from UPLOAD import upload_to_gcs

def main():

    tool = "reamer"
    samplerate = 16000

    record(samplerate, tool)


def record(samplerate, tool):

    COUNTER_FILE = "counter.txt"

    with open(COUNTER_FILE, "r") as f:
        counter, data_set = map(int, f.read().split())

    # Check if 12 runs is reached and update the dataset if true
    if counter % 12 == 0:
        data_set += 1
        print(f"Run {counter}: Data{data_set}")
    else:
        print(f"Run {counter}: Data{data_set}")
    
    channels = 1 #Mic channels (Only one is needed)
    filename = f"sample_{counter}" #Filename of the both the NPZ and CSV files

    #Tool duration selection
    if tool == "reamer":
        duration = 90
    else:
        duration = 80

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

    #initialize the upload function.


if __name__ == "__main__":
    main()
