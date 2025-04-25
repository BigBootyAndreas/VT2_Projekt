import customtkinter as ctk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import threading
import time
from collections import deque
import random  # Used for simulating sensor input

# Constants
imu_sampling_interval = 1.25 / 1000  # 800Hz
acoustic_sampling_interval = 0.0625 / 1000  # 16kHz

# Data Buffers
imu_buffer = {
    "Accel_X": deque(maxlen=500),
    "Accel_Y": deque(maxlen=500),
    "Accel_Z": deque(maxlen=500),
    "Timestamp": deque(maxlen=500)
}

acoustic_buffer = {
    "Amplitude": deque(maxlen=500),
    "Timestamp": deque(maxlen=500)
}

# Variables Class
class Variables:
    def __init__(self):
        self.current_tool = "Reamer 20"
        self.current_job = "Milling"
        self.spec_toollife = "200 CT"
        self.job_time_remaining = 1200
        self.est_toollife = 12345
        self.machine_running = 1
        self.button_color = ""

vars = Variables()

# Helpers
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
        vars.button_color = "green"
        return "Running"
    else:
        vars.button_color = "red"
        return "Idle"

# GUI Setup
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
app = ctk.CTk()
app.title("CNC Tool Wear Monitor")
app.geometry("1200x800")
app.grid_columnconfigure((0, 1), weight=1, uniform="column")
app.grid_rowconfigure(0, weight=1, uniform="row")
app.grid_rowconfigure(1, weight=2, uniform="row")

# Tool Info Panel
tool_info_frame = ctk.CTkFrame(app, corner_radius=20, fg_color="#1a1a1a")
tool_info_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
ctk.CTkLabel(tool_info_frame, text=f"Tool: {vars.current_tool}", font=("Arial", 22, "bold")).pack(pady=5)
ctk.CTkLabel(tool_info_frame, text=f"Spec. Tool Life: {vars.spec_toollife}", font=("Arial", 18)).pack(pady=5)
ctk.CTkLabel(tool_info_frame, text=f"Current Job: {vars.current_job}", font=("Arial", 18)).pack(pady=5)
ctk.CTkLabel(tool_info_frame, text=f"Job Time Remaining: {seconds_to_hms(vars.job_time_remaining)}", font=("Arial", 18)).pack(pady=5)
status_label = ctk.CTkLabel(tool_info_frame, text=f"{check_machine_status(vars.machine_running)}", fg_color=f"{vars.button_color}", width=140, height=140, corner_radius=70, font=("Arial", 22, "bold"))
status_label.pack(pady=20)

# Tool Life Panel
tool_life_frame = ctk.CTkFrame(app, corner_radius=20, fg_color="#1a1a1a")
tool_life_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
ctk.CTkLabel(tool_life_frame, text=f"Est. Toollife Remaining: \n {seconds_to_hms(vars.est_toollife)}", text_color="red", font=("Arial", 24, "bold")).pack(pady=10)
life_progress = ctk.CTkProgressBar(tool_life_frame, width=350, height=25)
life_progress.pack(pady=15, fill="x")
life_progress.set(0.80)
progress_labels = tk.Frame(tool_life_frame)
# progress_labels.pack()
# tk.Label(progress_labels, text="0%", font=("Arial", 16)).pack(side="left", padx=10)
# tk.Label(progress_labels, text="100%", font=("Arial", 16)).pack(side="right", padx=10)

# Plot Creation
def create_plot_frame(title, row, column, num_subplots=1):
    frame = ctk.CTkFrame(app, corner_radius=20, fg_color="#1a1a1a")
    frame.grid(row=row, column=column, sticky="nsew", padx=10, pady=10)
    fig, axes = plt.subplots(num_subplots, 1, figsize=(5, 3))
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    if num_subplots == 1:
        axes = [axes]
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.6)
    return axes, fig, canvas

# Create Acoustic and IMU Graphs
ax_acoustic, fig_acoustic, canvas_acoustic = create_plot_frame("Acoustic Data", 1, 0, 1)
(ax_imu_x, ax_imu_y, ax_imu_z), fig_imu, canvas_imu = create_plot_frame("Vibration Data", 1, 1, 3)

# Update Graphs in Real-Time
def update_acoustic():
    while True:
        # Acoustic Graph
        ax_acoustic[0].clear()
        if len(acoustic_buffer["Timestamp"]) > 0:
            ax_acoustic[0].plot(acoustic_buffer["Timestamp"], acoustic_buffer["Amplitude"], 'b', label="Amplitude")
            ax_acoustic[0].axhline(y=np.mean(acoustic_buffer["Amplitude"]), color='r', linestyle='--', label="Mean")
            ax_acoustic[0].set_xlabel("Time")
            ax_acoustic[0].set_ylabel("Amplitude")
            ax_acoustic[0].legend(loc="upper right", fontsize="6")

        canvas_acoustic.draw()

        time.sleep(acoustic_sampling_interval)

def update_imu():
    while True:
        # IMU X
        ax_imu_x.clear()
        if len(imu_buffer["Timestamp"]) > 0:
            ax_imu_x.plot(imu_buffer["Timestamp"], imu_buffer["Accel_X"], 'r', label="Accel X")
            ax_imu_x.axhline(y=np.mean(imu_buffer["Accel_X"]), color='r', linestyle='--', label="Mean X")
            ax_imu_x.set_ylabel("Accel X")
            ax_imu_x.legend(loc="upper right", fontsize="6")

        # IMU Y
        ax_imu_y.clear()
        if len(imu_buffer["Timestamp"]) > 0:
            ax_imu_y.plot(imu_buffer["Timestamp"], imu_buffer["Accel_Y"], 'g', label="Accel Y")
            ax_imu_y.axhline(y=np.mean(imu_buffer["Accel_Y"]), color='r', linestyle='--', label="Mean Y")
            ax_imu_y.set_ylabel("Accel Y")
            ax_imu_y.legend(loc="upper right", fontsize="6")

        # IMU Z
        ax_imu_z.clear()
        if len(imu_buffer["Timestamp"]) > 0:
            ax_imu_z.plot(imu_buffer["Timestamp"], imu_buffer["Accel_Z"], 'b', label="Accel Z")
            ax_imu_z.axhline(y=np.mean(imu_buffer["Accel_Z"]), color='r', linestyle='--', label="Mean Z")
            ax_imu_z.set_xlabel("Time")
            ax_imu_z.set_ylabel("Accel Z")
            ax_imu_z.legend(loc="upper right", fontsize="6")

        canvas_imu.draw()

        time.sleep(imu_sampling_interval)

# Start update thread
threading.Thread(target=update_acoustic, daemon=True).start()
threading.Thread(target=update_imu, daemon=True).start()

# Launch GUI
app.mainloop()
