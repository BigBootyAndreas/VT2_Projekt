import threading
from IMU import record_imu
from MIC import record_audio
from UPLOAD import upload_to_gcs
from decompressor import decompress
from encrypter import encrypt
import os

# Text colors used console
CYELLOW = '\033[93m'
CGREEN = '\033[92m'
CBLUE = '\033[94m'
CRED = '\033[91m'
CORANGE = '\033[0;33m'
CEND = '\033[0m'


def record(samplerate, tool):

    COUNTER_FILE = "counter.txt"

    with open(COUNTER_FILE, "r") as f:
        tool_counter_1, tool_counter_2 = map(int, f.read().split())
    
    channels = 1 # Microphone channels

    # Tool duration and sample naming selection for file handling
    if tool == "reamer":
        duration = 60
        print(CRED + f"Tool: " + CEND + f"{tool} " + CRED + f"Sample: " + CEND + f"{tool_counter_1}")
        filename = f"{tool}_sample_{tool_counter_1}"
        tool_counter_1 += 1
    else:
        duration = 55
        print(CRED + f"Tool: " + CEND + f"{tool} " + CRED + f"Sample: " + CEND + f"{tool_counter_2}")
        filename = f"{tool}_sample_{tool_counter_2}"
        tool_counter_2 += 1

    # Threading for concurrent recording
    audio_thread = threading.Thread(target=record_audio, args=(duration, samplerate, channels, filename))
    imu_thread = threading.Thread(target=record_imu, args=(duration, filename))

    # Thread start
    audio_thread.start()
    imu_thread.start()

    # Wait for threads to finish
    audio_thread.join()
    imu_thread.join()

    with open(COUNTER_FILE, "w") as f:
        f.write(f"{tool_counter_1} {tool_counter_2}")

    #Decompress NPZ files to CSV files
    decompress()

    #Delete the converted NPZ files
    folder_path = "segments/audio"

    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter out only the .npz files
    npz_files = [file for file in files if file.endswith(".npz")]

    for file_name in npz_files:

        full_path = os.path.join(folder_path, file_name)
        
        # Delete npz file
        os.remove(full_path)
        print(f"Deleted {file_name}")

    # Encryption key used to en- and cecrypt the CSV files (Placeholder is used for GitHub)
    key = b'SUPER SECRET ENCRYPTION KEY'

    encrypt(f"segments/audio/{filename}.csv", f"segments/audio/{filename}_encrypted.csv", key)
    encrypt(f"segments/imu/{filename}.csv", f"segments/imu/{filename}_encrypted.csv", key)

    print("Succesfully encrypted all files")

    # initialize the upload function with bucket name and credentials.
    print(CORANGE + f"Upload initialied" + CEND)
    BUCKET_NAME = "aau-vt2"
    CREDENTIALS_FILE = "aau-vt2-secret-credentials.json" # JSON credentials file is not included in the commit.

    # Upload acoustic data
    print(f"Uploading audio file: {filename}_encrypted")
    upload_to_gcs(BUCKET_NAME, f"segments/audio/{filename}_encrypted.csv", f"audio/{tool}/{filename}_encrypted.csv", CREDENTIALS_FILE, filename)
    
    # Upload IMU data
    print(f"Uploading imu file: {filename}_encrypted")
    upload_to_gcs(BUCKET_NAME, f"segments/imu/{filename}_encrypted.csv", f"imu/{tool}/{filename}_encrypted.csv", CREDENTIALS_FILE, filename)
    
    print("Process finished")