import socket
import json
import customtkinter as ctk
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
from collections import deque

# Config
HOST = "127.0.0.1"  # Use friend's IP if remote
PORT = 65432
imu_hz = 800
acoustic_hz = 16000
imu_window = 2 * imu_hz
acoustic_window = 2 * acoustic_hz
update_interval = 0.01  # 10ms

# Buffers
imu_data = {
    "x": deque(maxlen=imu_window),
    "y": deque(maxlen=imu_window),
    "z": deque(maxlen=imu_window),
}
acoustic_data = deque(maxlen=acoustic_window)

# UI Setup
ctk.set_appearance_mode("dark")
app = ctk.CTk()
app.title("Real-Time Data Monitor")
app.geometry("1200x600")
app.grid_columnconfigure((0, 1), weight=1)
app.grid_rowconfigure(0, weight=1)

def create_plot(title, rows, column, n_subplots):
    frame = ctk.CTkFrame(app)
    frame.grid(row=rows, column=column, sticky="nsew", padx=10, pady=10)
    fig, axes = plt.subplots(n_subplots, 1, figsize=(5, 4))
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    if n_subplots == 1:
        axes = [axes]
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.6)
    return axes, canvas

ax_imu, canvas_imu = create_plot("IMU", 0, 0, 3)
ax_acoustic, canvas_acoustic = create_plot("Acoustic", 0, 1, 1)

last_update_time = time.perf_counter()

def update_plots():
    global last_update_time
    now = time.perf_counter()
    if now - last_update_time < update_interval:
        return
    last_update_time = now

    # Acoustic plot
    ax_acoustic[0].clear()
    ax_acoustic[0].plot(acoustic_data, 'b', linewidth=0.5)
    ax_acoustic[0].set_title("Acoustic Amplitude")
    ax_acoustic[0].set_ylim(-1, 1)

    # IMU plots
    colors = ['r', 'g', 'b']
    labels = ['Accel_X', 'Accel_Y', 'Accel_Z']
    for i, (k, ax) in enumerate(zip(imu_data.keys(), ax_imu)):
        ax.clear()
        ax.plot(imu_data[k], colors[i])
        ax.set_title(labels[i])
        ax.set_ylim(-20, 20)

    canvas_acoustic.draw_idle()
    canvas_imu.draw_idle()

def receive_data():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        buffer = ""
        while True:
            chunk = s.recv(4096).decode()
            if not chunk:
                break
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                try:
                    data = json.loads(line)
                    if data["type"] == "imu":
                        imu_data["x"].append(float(data["Accel_X"]))
                        imu_data["y"].append(float(data["Accel_Y"]))
                        imu_data["z"].append(float(data["Accel_Z"]))
                    elif data["type"] == "acoustic_batch":
                        acoustic_data.extend([float(a) for a in data["Amplitude"]])
                    app.after(0, update_plots)
                except json.JSONDecodeError:
                    continue

threading.Thread(target=receive_data, daemon=True).start()
app.mainloop()
