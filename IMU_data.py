import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

# STFT-based IMU processing
def imu_processing(df, plot_type="psd"):
    print("Detected columns:", df.columns)

    # Use actual column names from the CSV
    id = df["id"].astype(int).values
    x_accel = df["X (g)"].astype(float).values
    y_accel = df["Y (g)"].astype(float).values
    z_accel = df["Z (g)"].astype(float).values
    time = df["epoch"].astype(float).values

    # Convert epoch to relative time
    time = time - time[0]

    # Sampling rate
    sr = 100

    if plot_type == "raw":
        # Determine the global y-axis range across all axes
        y_min = min(x_accel.min(), y_accel.min(), z_accel.min())
        y_max = max(x_accel.max(), y_accel.max(), z_accel.max())
        y_range = y_max - y_min

        # Adjust the range to add a 10% margin
        y_min_adjusted = round(y_min - 0.1 * y_range, 2)
        y_max_adjusted = round(y_max + 0.1 * y_range, 2)

        # Plot X-axis data
        plt.figure(figsize=(10, 4))
        plt.plot(time, x_accel, label="X-axis", color="r")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (g)")
        plt.title("Raw IMU Data - X-axis")
        plt.ylim(y_min_adjusted, y_max_adjusted)
        plt.grid(True)
        plt.show()

        # Plot Y-axis data
        plt.figure(figsize=(10, 4))
        plt.plot(time, y_accel, label="Y-axis", color="g")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (g)")
        plt.title("Raw IMU Data - Y-axis")
        plt.ylim(y_min_adjusted, y_max_adjusted)
        plt.grid(True)
        plt.show()

        # Plot Z-axis data
        plt.figure(figsize=(10, 4))
        plt.plot(time, z_accel, label="Z-axis", color="b")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (g)")
        plt.title("Raw IMU Data - Z-axis")
        plt.ylim(y_min_adjusted, y_max_adjusted)
        plt.grid(True)
        plt.show()

    elif plot_type == "psd":
        # Plot PSD function
        def plot_psd(accel, sr, axis_name):
            n_fft = 8192
            hop_length = 2048
            win_length = 8192

            stft_imu = librosa.stft(accel, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            magnitude = np.abs(stft_imu)
            power = np.mean(magnitude**2, axis=1)

            freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            freq_resolution = sr / len(freq_bins)
            psd = power / freq_resolution

            plt.figure(figsize=(10, 4))
            plt.plot(freq_bins, psd, label=f'{axis_name}-axis', linewidth=2)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density (G²/Hz)')
            plt.title(f'PSD for {axis_name} Acceleration')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.legend()
            plt.xlim(left=1)
            plt.show()

        # Plot PSD for each axis
        plot_psd(x_accel, sr, 'X')
        plot_psd(y_accel, sr, 'Y')
        plot_psd(z_accel, sr, 'Z')