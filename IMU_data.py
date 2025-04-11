import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

# STFT-based IMU processing
def imu_processing(df):
    print("Detected columns:", df.columns)

    # Use actual column names from the CSV
    x_accel = df["X (g)"].astype(float).values
    y_accel = df["Y (g)"].astype(float).values
    z_accel = df["Z (g)"].astype(float).values
    time = df["epoch"].astype(float).values

    # Convert epoch to relative time
    time = time - time[0]

    # Sampling rate
    sr = 100

    # Plot PSD function
    def plot_psd(accel, sr, axis_name, ax):
        n_fft = 8192
        hop_length = 2048
        win_length = 8192

        stft_imu = librosa.stft(accel, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        magnitude = np.abs(stft_imu)
        power = np.mean(magnitude**2, axis=1)

        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        freq_resolution = sr / len(freq_bins)
        psd = power / freq_resolution

        ax.plot(freq_bins, psd, label=f'{axis_name}-axis', linewidth=2)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power Spectral Density (G²/Hz)')
        ax.set_title(f'PSD for {axis_name} Acceleration')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.legend()

        # Limit frequency range to above 10 Hz (this will cut the plot at 10 Hz)
        ax.set_xlim(left=1)  # Set the lower limit of the x-axis to 10 Hz

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    plot_psd(x_accel, sr, 'X', axs[0])
    plot_psd(y_accel, sr, 'Y', axs[1])
    plot_psd(z_accel, sr, 'Z', axs[2])

    plt.subplots_adjust(hspace=0.345)
    plt.show()
