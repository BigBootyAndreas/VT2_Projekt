import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from File_reader import *


def acoustic_processing(df, show_plot=False):
    # Extract time and amplitude from the dataframe
    time = df["Time"].values
    amplitude = df["Amplitude"].values
    
    # Calculate the sampling rate from the time data
    # This assumes the time column is in seconds and is uniformly sampled
    time_diffs = np.diff(time)
    avg_time_diff = np.mean(time_diffs)
    sr = int(1 / avg_time_diff)
    
    print(f"Detected sampling rate: {sr} Hz")
    
    # Compute STFT
    n_fft = 2048  # FFT window size
    hop_length = 512  # Number of samples between successive frames
    
    # Perform STFT
    stft_result = librosa.stft(amplitude, n_fft=n_fft, hop_length=hop_length)
    
    # Convert to magnitude decibels
    stft_magnitude = librosa.amplitude_to_db(np.abs(stft_result), ref=np.max)
    
    # Create figure with multiple plots for different analyses
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot 1: Time domain signal
    axs[0].plot(time, amplitude)
    axs[0].set_title("Acoustic Signal (Time Domain)")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)
    
    # Plot 2: Spectrogram
    img = librosa.display.specshow(
        stft_magnitude, 
        sr=sr, 
        hop_length=hop_length,
        x_axis="time", 
        y_axis="linear", 
        ax=axs[1]
    )
    fig.colorbar(img, ax=axs[1], format="%+2.0f dB", label="Amplitude (dB)")
    axs[1].set_title("Spectrogram (STFT)")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Frequency")
    
    # Add some additional analysis options
    def on_key(event):
        """Handle keyboard events for additional visualization options"""
        if event.key == 'm':
            # 'm' key: Show mel spectrogram instead
            axs[1].clear()
            mel_spec = librosa.feature.melspectrogram(y=amplitude, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            img = librosa.display.specshow(
                mel_spec_db, 
                sr=sr, 
                x_axis="time", 
                y_axis="mel", 
                ax=axs[1]
            )
            fig.colorbar(img, ax=axs[1], format="%+2.0f dB", label="Amplitude (dB)")
            axs[1].set_title("Mel Spectrogram")
            fig.canvas.draw()
        elif event.key == 's':
            # 's' key: Show original STFT spectrogram
            axs[1].clear()
            img = librosa.display.specshow(
                stft_magnitude, 
                sr=sr, 
                hop_length=hop_length,
                x_axis="time", 
                y_axis="linear", 
                ax=axs[1]
            )
            fig.colorbar(img, ax=axs[1], format="%+2.0f dB", label="Amplitude (dB)")
            axs[1].set_title("Spectrogram (STFT)")
            fig.canvas.draw()
    
    # Connect the key event handler
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.tight_layout()
    print("Press 'm' to switch to Mel spectrogram or 's' to return to STFT spectrogram")
    if show_plot:
        plt.tight_layout()
        print("Press 'm' to switch to Mel spectrogram or 's' to return to STFT spectrogram")
        plt.show()
    else:
        plt.close(fig)  # Close the figure to free memory
    
    return stft_result, sr

# Optional: Function for advanced analysis that can be added later
def advanced_acoustic_analysis(df, stft_result=None, sr=None):
   
    if stft_result is None or sr is None:
        stft_result, sr = acoustic_processing(df)
    
    # Extract amplitude from the dataframe
    amplitude = df.iloc[:, 1].values
    
    # Create figure for advanced analysis
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot 1: Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=amplitude, sr=sr)[0]
    # Convert frame indices to time domain
    frames = range(len(spectral_centroids))
    t = librosa.frames_to_time(frames)
    
    axs[0].plot(t, spectral_centroids)
    axs[0].set_title("Spectral Centroid")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Frequency (Hz)")
    axs[0].grid(True)
    
    # Plot 2: Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=amplitude, sr=sr)[0]
    axs[1].plot(t, spectral_bandwidth)
    axs[1].set_title("Spectral Bandwidth")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Frequency (Hz)")
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return spectral_centroids, spectral_bandwidth