import customtkinter as ctk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
import pandas as pd

# Load Data
imu_df = pd.read_csv("IMU_data.csv", names=["Index", "Accel_X", "Accel_Y", "Accel_Z", "Timestamp"])
acoustic_df = pd.read_csv("acoustic_data.csv", delimiter="\t", names=["RawData"])

# Split the RawData column into separate fields
acoustic_df[["Timestamp", "Frequency", "Amplitude"]] = acoustic_df["RawData"].str.split(',', expand=True)
acoustic_df.drop(columns=["RawData"], inplace=True)

# Convert Timestamp column to datetime
acoustic_df["Timestamp"] = pd.to_datetime(acoustic_df["Timestamp"], errors='coerce')

# Drop any rows where timestamp conversion failed
acoustic_df = acoustic_df.dropna().reset_index(drop=True)

# Convert timestamps to numeric (seconds since start)
acoustic_df["Timestamp"] = (acoustic_df["Timestamp"] - acoustic_df["Timestamp"].iloc[0]).dt.total_seconds()
imu_df["Timestamp"] = imu_df["Timestamp"].astype(float) / 1000  # Convert to seconds

# Convert Frequency and Amplitude to float
acoustic_df["Frequency"] = acoustic_df["Frequency"].astype(float)
acoustic_df["Amplitude"] = acoustic_df["Amplitude"].astype(float)

# Sampling Rates
imu_sampling_interval = 1.25 / 1000  # 1.25ms per sample (800Hz)
acoustic_sampling_interval = 0.0625 / 1000  # 0.0625ms per sample (16kHz)

# CustomTkinter Theme Setup
ctk.set_appearance_mode("dark")  # Dark Mode
ctk.set_default_color_theme("blue")

# Main App Window
app = ctk.CTk()
app.title("CNC Tool Wear Monitor")
app.geometry("1200x800")
app.grid_columnconfigure((0, 1), weight=1, uniform="column")
app.grid_rowconfigure((0, 1), weight=1, uniform="row")

# Tool Information (Top Left)
tool_info_frame = ctk.CTkFrame(app, corner_radius=20, fg_color="#1a1a1a")
tool_info_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
tool_label = ctk.CTkLabel(tool_info_frame, text="Tool: Reamer 20", font=("Arial", 22, "bold"))
tool_label.pack(pady=5)
life_label = ctk.CTkLabel(tool_info_frame, text="Spec. Tool Life: 200 CT", font=("Arial", 18))
life_label.pack(pady=5)
job_label = ctk.CTkLabel(tool_info_frame, text="Current Job: Rimming", font=("Arial", 18))
job_label.pack(pady=5)
time_label = ctk.CTkLabel(tool_info_frame, text="Remaining Time: 20 min", font=("Arial", 18))
time_label.pack(pady=5)
status_label = ctk.CTkLabel(tool_info_frame, text="Running", fg_color="green", width=120, height=120, corner_radius=60, font=("Arial", 20, "bold"))
status_label.pack(pady=15)

# Estimated Tool Life (Top Right)
tool_life_frame = ctk.CTkFrame(app, corner_radius=20, fg_color="#1a1a1a")
tool_life_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
countdown_label = ctk.CTkLabel(tool_life_frame, text="Remaining: 20:43", text_color="red", font=("Arial", 22, "bold"))
countdown_label.pack(pady=10)
life_progress = ctk.CTkProgressBar(tool_life_frame, width=300, height=20)
life_progress.pack(pady=10, fill="x")
life_progress.set(0.25)  # Example progress (25% tool life used)
progress_labels = tk.Frame(tool_life_frame)
progress_labels.pack()
tk.Label(progress_labels, text="0%", font=("Arial", 14)).pack(side="left", padx=10)
tk.Label(progress_labels, text="100%", font=("Arial", 14)).pack(side="right", padx=10)

# Live Data Graphs (Bottom Left & Bottom Right)
def create_plot_frame(title, row, column):
    frame = ctk.CTkFrame(app, corner_radius=20, fg_color="#1a1a1a")
    frame.grid(row=row, column=column, sticky="nsew", padx=10, pady=10)
    fig, ax = plt.subplots()
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Amplitude", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    return ax, fig, canvas

# Acoustic & IMU Graphs
ax_acoustic, fig_acoustic, canvas_acoustic = create_plot_frame("Acoustic Data", 1, 0)
ax_imu, fig_imu, canvas_imu = create_plot_frame("IMU Data", 1, 1)

# Initialize Index Counters
imu_index = 0
acoustic_index = 0

# Real-Time Data Update
def update_graphs():
    global imu_index, acoustic_index
    while imu_index < len(imu_df) and acoustic_index < len(acoustic_df):
        ax_acoustic.clear()
        ax_acoustic.plot(acoustic_df.iloc[max(0, acoustic_index-100):acoustic_index]["Amplitude"], 'b', label="Acoustic Signal")
        ax_acoustic.legend()

        ax_imu.clear()
        for axis in ["Accel_X", "Accel_Y", "Accel_Z"]:
            ax_imu.plot(imu_df.iloc[max(0, imu_index-100):imu_index][axis], label=axis)
        ax_imu.legend()
        
        canvas_acoustic.draw()
        canvas_imu.draw()
        
        imu_index += 1
        acoustic_index += 1

        time.sleep(imu_sampling_interval)  # Controls the update rate

# Start Real-Time Graph Update Thread
threading.Thread(target=update_graphs, daemon=True).start()

# Run App
app.mainloop()
