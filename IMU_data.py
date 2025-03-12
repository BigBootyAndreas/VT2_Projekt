import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

# Hereunder the STFT for IMU is coded
def imu_processing(df):
    # Extracting X, Y, Z accelerations and the epoch time from the dataframe
    x_accel = df.iloc[:, 1].values  
    y_accel = df.iloc[:, 2].values  
    z_accel = df.iloc[:, 3].values  
    time = df.iloc[:, 4].values     

    # Converting the epoch time to relative time 
    time = time - time[0]

    # Define the sampling rate 
    sr = 100  
   
    # Function to compute and plot the Power Spectral Density (PSD) for IMU data
    def plot_psd(accel, sr, axis_name, ax):
        #Short-Time Fourier Transform (STFT) parameters
        n_fft = 8192       # Number of points in FFT, higher means better frequency resolution
        hop_length = 2048  # Number of points between FFTs (controls time resolution)
        win_length = 8192  # Window length for each FFT computation, usually equal to n_fft
        
        # Performing STFT
        stft_imu = librosa.stft(accel, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        # Calculate the magnitude and power from magnitude
        magnitude = np.abs(stft_imu)
        power = np.mean(magnitude**2, axis=1)  # Averaging power over time (across columns)

        #Frequency bins corresponding to the FFT results
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Normalize power by frequency bin width to get the Power Spectral Density (PSD)
        freq_resolution = sr / len(freq_bins)  # Frequency resolution
        psd = power / freq_resolution  # PSD (Power Spectral Density)

        ax.plot(freq_bins, psd, label=f'{axis_name}-axis', linewidth=2)
        ax.set_xlabel('Frequency (Hz)')  # X-axis label (Frequency in Hz)
        ax.set_ylabel('Power Spectral Density (G²/Hz)')  # Y-axis label (PSD in G²/Hz)
        ax.set_title(f'PSD for {axis_name} Acceleration')  # Title for each subplot
        ax.set_xscale('log')  # Logarithmic scale for X-axis (frequency)
        ax.set_yscale('log')  # Logarithmic scale for Y-axis (power)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)  # Grid with both major and minor lines
        ax.legend()  # Legend for each axis label

    # Create a figure with 3 subplots (one for each axis: X, Y, Z)
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))  # 3 rows, 1 column for subplots

    # Compute and plot PSD for X, Y, and Z axes in their respective subplots
    plot_psd(x_accel, sr, 'X', axs[0])  # Plot X-axis PSD in the first subplot
    plot_psd(y_accel, sr, 'Y', axs[1])  # Plot Y-axis PSD in the second subplot
    plot_psd(z_accel, sr, 'Z', axs[2])  # Plot Z-axis PSD in the third subplot

    # Adjust layout with custom vertical spacing between subplots
    plt.subplots_adjust(hspace=0.345)  # Set the vertical spacing between subplots

    # Show the plots
    plt.show()  # Display the plots