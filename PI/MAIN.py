import threading
from IMU import record_imu
from MIC import record_audio
from UPLOAD import upload_to_gcs

#Text colors
CYELLOW = '\033[93m'
CGREEN = '\033[92m'
CBLUE = '\033[94m'
CRED = '\033[91m'
CORANGE = '\033[0;33m'
CEND = '\033[0m'

def main():

    tool = "reamer"
    samplerate = 16000

    record(samplerate, tool)


def record(samplerate, tool):

    COUNTER_FILE = "counter.txt"

    with open(COUNTER_FILE, "r") as f:
        tool_counter_1, tool_counter_2 = map(int, f.read().split())
    
    channels = 1 #Mic channels (Only one is needed)

    #Tool duration and sample naming selection
    if tool == "reamer":
        duration = 9
        print(CRED + f"Tool: " + CEND + f"{tool} " + CRED + f"Sample: " + CEND + f"{tool_counter_1}")
        filename = f"{tool}_sample_{tool_counter_1}"
        tool_counter_1 += 1
    else:
        duration = 8
        print(CRED + f"Tool: " + CEND + f"{tool} " + CRED + f"Sample: " + CEND + f"{tool_counter_2}")
        filename = f"{tool}_sample_{tool_counter_2}"
        tool_counter_2 += 1

    # Create threads for concurrent recording
    audio_thread = threading.Thread(target=record_audio, args=(duration, samplerate, channels, filename))
    imu_thread = threading.Thread(target=record_imu, args=(duration, filename))

    # Start both threads
    audio_thread.start()
    imu_thread.start()

    # Wait for both threads to complete
    audio_thread.join()
    imu_thread.join()

    with open(COUNTER_FILE, "w") as f:
        f.write(f"{tool_counter_1} {tool_counter_2}")


    #initialize the upload function and provide bucket name and credentials.
    print(CORANGE + f"Upload initialied" + CEND)
    BUCKET_NAME = "aau_vt1"
    CREDENTIALS_FILE = "euphoric-anchor-439706-k3-d5147822d8ac.json"

    #AUDIO
    print("Uploading NPZ files:")
    upload_to_gcs(BUCKET_NAME, f"segments/{tool}/" + filename + r".npz", filename + r".npz", CREDENTIALS_FILE, filename )
    
    #IMU
    print("Uploading CSV files:")
    upload_to_gcs(BUCKET_NAME, f"segments/{tool}/" + filename + r".csv", filename + r".csv", CREDENTIALS_FILE, filename )


if __name__ == "__main__":
    main()