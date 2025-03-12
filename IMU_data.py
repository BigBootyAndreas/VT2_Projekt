import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

# Hereunder the STFT for IMU is coded
def imu_processing(df):
    # Below the data for X, Y, Z and epoch time are isolated.
    x_accel = df.iloc[:, 1].values  
    y_accel = df.iloc[:, 2].values  
    z_accel = df.iloc[:, 3].values  
    time = df.iloc[:, 4].values     

    # Below, the time is converted from epoch
    time = time - time[0]  

    # The sampling rate is defined 
    sr = 100  # Ensure this is correct for your data
   
    # Function to compute and plot the PSD for IMU data
    def plot_psd(accel, sr, axis_name):
        # Define STFT parameters
        #n_fft= is the number of points used for each FFT, hop_length is the number of points between each FFT (resulotion)
        # win_length is the window size of each FFT
        #Remember higher n_fft = better frequency resolution, lower hop_length= better resulotion
        n_fft = 8192
        hop_length = 2048
        win_length = 8192

        # Compute STFT using librosa
        stft_imu = librosa.stft(accel, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        # Calculate magnitude and power
        magnitude = np.abs(stft_imu)
        power = np.mean(magnitude**2, axis=1)  # Calculate the average power over time

        # Get frequency values
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Normalize power by frequency bin width
        freq_resolution = sr / len(freq_bins)
        psd = power / freq_resolution

        # Plot Acceleration vs. Frequency
        plt.figure(figsize=(8, 5))
        plt.plot(freq_bins, psd, label=f'{axis_name}-axis', linewidth=2)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (G²/Hz)')
        plt.title(f'PSD for {axis_name} Acceleration')
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.show()

    # Compute and plot PSD for X, Y, and Z axes
    plot_psd(x_accel, sr, 'X')
    plot_psd(y_accel, sr, 'Y')
    plot_psd(z_accel, sr, 'Z')

