import numpy as np
import sounddevice as sd
import datetime
from google.cloud import storage

#Text colors
CYELLOW = '\033[93m'
CGREEN = '\033[92m'
CBLUE = '\033[94m'
CRED = '\033[91m'
CBLINK = '\033[5m'
CEND = '\033[0m'

def record_audio(duration, samplerate, channels, filename):

    segment_start_time = datetime.datetime.now()
    
    print(CBLUE + "Recording started" + CEND)

    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype="float32")

    sd.wait() # Wait until recording is finished

    print(CBLUE + f"Recording finished. Size of recording: {recording.shape}" + CEND)

    time_values = np.linspace(0, duration, len(recording))

    # Save segment data to file or any other storage method
    print(CGREEN + f"Saving segment: {filename}" + CEND)

    # Add time values to array
    data = np.column_stack((time_values, recording))

    # Create timestamps
    timestamps = [(segment_start_time +
    datetime.timedelta(seconds=time_values[i])).strftime("%Y-%m-%d %H:%M:%S") for
    i in range(len(recording))]

    # Add timestamp column to the data
    data_with_timestamp = np.column_stack((timestamps, data))
    np.savez_compressed("segments/audio/"+filename, a=data_with_timestamp)
    print(CGREEN + f"Segment {filename} saved" + CEND)