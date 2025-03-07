import librosa
import os
import numpy as np
import librosa.display
import matplotlib.pyplot as plt


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Load the audio file
audio_path = "your_audio_file.wav"  # Change this to your file
y, sr = librosa.load(audio_path, sr=None)  # Load with original sampling rate

# Compute STFT
stft_result = librosa.stft(y)

# Convert to magnitude (log scale for better visualization)
stft_magnitude = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)

# Plot the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(stft_magnitude, sr=sr, x_axis="time", y_axis="log")
plt.colorbar(label="Amplitude (dB)")
plt.title("Spectrogram (STFT)")
plt.xlabel("Time")
plt.ylabel("Frequency (log scale)")
plt.show()

