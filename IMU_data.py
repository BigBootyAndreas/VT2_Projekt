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
    time = (time - time[0])/1000  # Convert to seconds

    # Sampling rate
    sr = 400

    if plot_type == "raw":
        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        # Plot X-axis data
        axs[0].plot(time, x_accel, label="X-axis", color="r")
        axs[0].set_ylabel("Acceleration (g)")
        axs[0].set_title("Raw IMU Data - X-axis")
        axs[0].grid(True)

        # Plot Y-axis data
        axs[1].plot(time, y_accel, label="Y-axis", color="g")
        axs[1].set_ylabel("Acceleration (g)")
        axs[1].set_title("Raw IMU Data - Y-axis")
        axs[1].grid(True)

        # Plot Z-axis data
        axs[2].plot(time, z_accel, label="Z-axis", color="b")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Acceleration (g)")
        axs[2].set_title("Raw IMU Data - Z-axis")
        axs[2].grid(True)

        # Adjust layout
        plt.tight_layout()
        plt.show()

    elif plot_type == "psd":
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

            # Limit frequency range to above 10 Hz
            ax.set_xlim(left=1)

        # Plotting
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        plot_psd(x_accel, sr, 'X', axs[0])
        plot_psd(y_accel, sr, 'Y', axs[1])
        plot_psd(z_accel, sr, 'Z', axs[2])

        plt.subplots_adjust(hspace=0.345)
        plt.show()