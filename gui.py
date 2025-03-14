import customtkinter as ctk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
import pandas as pd

class Variables:
    """
    Class for initialising, setting and getting Variables.
    When instanced in global space, then every other scope can access the Variables.
    """

    def __init__(self):
        
        # contains name of tool in use
        self.current_tool = "Reamer 20"
        
        # contains name of current process
        self.current_job = "Milling"
        
        # contains tool life specified by supplier
        self.spec_toollife = "200 CT"

        self.job_time_remaining = 1200

        # contains estimated tool life in seconds, initially same as specified
        self.est_toollife = 12345

        # bool value containing machine running status
        self.machine_running = 0

vars = Variables()

def seconds_to_hms(seconds):
    if seconds >= 3600:
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
    else:
        minutes, seconds = divmod(seconds, 60)
        return f"{int(minutes):02}:{int(seconds):02}"

def check_machine_status(status):
    if status == 1:
        return "Running"
    else:
        return "Idle"

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
app.grid_rowconfigure(0, weight=1, uniform="row")
app.grid_rowconfigure(1, weight=2, uniform="row")

# Tool Information (Top Left)
tool_info_frame = ctk.CTkFrame(app, corner_radius=20, fg_color="#1a1a1a", )
tool_info_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
tool_label = ctk.CTkLabel(tool_info_frame, text=f"Tool: {vars.current_tool}", font=("Arial", 22, "bold"))
tool_label.pack(pady=5)
life_label = ctk.CTkLabel(tool_info_frame, text=f"Spec. Tool Life: {vars.spec_toollife}", font=("Arial", 18))
life_label.pack(pady=5)
job_label = ctk.CTkLabel(tool_info_frame, text=f"Current Job: {vars.current_job}", font=("Arial", 18))
job_label.pack(pady=5)
time_label = ctk.CTkLabel(tool_info_frame, text=f"Job Time Remaining: {seconds_to_hms(vars.job_time_remaining)}", font=("Arial", 18))
time_label.pack(pady=5)
status_label = ctk.CTkLabel(tool_info_frame, text=f"{check_machine_status(vars.machine_running)}", fg_color="green", width=140, height=140, corner_radius=70, font=("Arial", 22, "bold"))
status_label.pack(pady=20)

# Estimated Tool Life (Top Right)
tool_life_frame = ctk.CTkFrame(app, corner_radius=20, fg_color="#1a1a1a", )
tool_life_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
countdown_label = ctk.CTkLabel(tool_life_frame, text=f"Est. Toollife Remaining: \n {seconds_to_hms(vars.est_toollife)}", text_color="red", font=("Arial", 24, "bold"))
countdown_label.pack(pady=10)
life_progress = ctk.CTkProgressBar(tool_life_frame, width=350, height=25)
life_progress.pack(pady=15, fill="x")
life_progress.set(0.80)  # Example progress (80% tool life used)
progress_labels = tk.Frame(tool_life_frame)
progress_labels.pack()
tk.Label(progress_labels, text="0%", font=("Arial", 16)).pack(side="left", padx=10)
tk.Label(progress_labels, text="100%", font=("Arial", 16)).pack(side="right", padx=10)

# Live Data Graphs (Bottom Left & Bottom Right)
def create_plot_frame(title, row, column, num_subplots=1):
    frame = ctk.CTkFrame(app, corner_radius=20, fg_color="#1a1a1a", )
    frame.grid(row=row, column=column, sticky="nsew", padx=10, pady=10)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(5, 3))
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    if num_subplots == 1:
        axes = [axes]
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.6)
    return axes, fig, canvas

# Acoustic & IMU Graphs
ax_acoustic, fig_acoustic, canvas_acoustic = create_plot_frame("Acoustic Data", 1, 0, 1)
(ax_imu_x, ax_imu_y, ax_imu_z), fig_imu, canvas_imu = create_plot_frame("IMU Data", 1, 1, 3)

# Initialize Index Counters
imu_index = 0
acoustic_index = 0

# Real-Time Data Update
def update_graphs():
    global imu_index, acoustic_index
    start_time = time.time()
    while imu_index < len(imu_df) and acoustic_index < len(acoustic_df):
        ax_acoustic[0].clear()
        ax_acoustic[0].set_xlabel("Index")
        ax_acoustic[0].set_ylabel("Amplitude")
        ax_acoustic[0].tick_params(axis='both', labelsize=8)
        
        # Ensure proper range for x-axis
        data_range = range(max(0, acoustic_index-100), acoustic_index)
        amplitude_values = acoustic_df["Amplitude"].iloc[max(0, acoustic_index-100):acoustic_index]
        
        # Plot Mean Line
        ax_acoustic[0].axhline(y=amplitude_values.mean(), color='r', linestyle='--', label='Mean Amplitude')
        
        # Plot Signal
        ax_acoustic[0].plot(data_range, amplitude_values, 'b', label="Acoustic Signal")
        
        # Fix x-axis limits
        if len(data_range) > 0:
            ax_acoustic[0].set_xlim(min(data_range), max(data_range))
        
        ax_acoustic[0].legend(loc="upper right", fontsize="6")

        ax_imu_x.clear()
        ax_imu_x.set_xticklabels([])
        ax_imu_x.set_ylabel("Accel X")
        ax_imu_x.tick_params(axis='both', labelsize=8)
        
        # Ensure proper range for x-axis
        data_range_x = range(max(0, imu_index-100), imu_index)
        accel_x_values = imu_df["Accel_X"].iloc[max(0, imu_index-100):imu_index]
        
        # Plot Mean Line
        ax_imu_x.axhline(y=accel_x_values.mean(), color='r', linestyle='--', label='Mean X')
        
        # Plot Signal
        ax_imu_x.plot(data_range_x, accel_x_values, 'r', label="Accel X")
        
        # Fix x-axis limits
        if len(data_range_x) > 0:
            ax_imu_x.set_xlim(min(data_range_x), max(data_range_x))

        ax_imu_x.legend(loc="upper right", fontsize="6")

        ax_imu_y.clear()
        ax_imu_y.set_xticklabels([])
        ax_imu_y.set_ylabel("Accel Y")
        ax_imu_y.tick_params(axis='both', labelsize=8)
        
        # Ensure proper range for x-axis
        data_range_y = range(max(0, imu_index-100), imu_index)
        accel_y_values = imu_df["Accel_Y"].iloc[max(0, imu_index-100):imu_index]
        
        # Plot Mean Line
        ax_imu_y.axhline(y=accel_y_values.mean(), color='r', linestyle='--', label='Mean Y')
        
        # Plot Signal
        ax_imu_y.plot(data_range_y, accel_y_values, 'g', label="Accel Y")
        
        # Fix x-axis limits
        if len(data_range_y) > 0:
            ax_imu_y.set_xlim(min(data_range_y), max(data_range_y))
        
        ax_imu_y.legend(loc="upper right", fontsize="6")

        ax_imu_z.clear()
        ax_imu_z.set_xlabel("Index")
        ax_imu_z.set_ylabel("Accel")
        ax_imu_z.tick_params(axis='both', labelsize=8)
        
        # Ensure proper range for x-axis
        data_range_z = range(max(0, imu_index-100), imu_index)
        accel_z_values = imu_df["Accel_Z"].iloc[max(0, imu_index-100):imu_index]
        
        # Plot Mean Line
        ax_imu_z.axhline(y=accel_z_values.mean(), color='r', linestyle='--', label='Mean Z')
        
        # Plot Signal
        ax_imu_z.plot(data_range_z, accel_z_values, 'b', label="Accel Z")
        
        # Fix x-axis limits
        if len(data_range_z) > 0:
            ax_imu_z.set_xlim(min(data_range_z), max(data_range_z))
        
        ax_imu_z.legend(loc="upper right", fontsize="6")
        
        canvas_acoustic.draw()
        canvas_imu.draw()
        
        imu_index += 1
        acoustic_index += 1

        time.sleep(imu_sampling_interval)  # Controls the update rate

# Start Real-Time Graph Update Thread
threading.Thread(target=update_graphs, daemon=True).start()

# Run App
app.mainloop()
