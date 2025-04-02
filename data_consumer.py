import customtkinter as ctk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import socket
import json
import threading
import queue
import time

HOST = '127.0.0.1'
PORT = 65432

data_queue = queue.Queue()

# GUI Setup
app = ctk.CTk()
app.title("CNC Tool Wear Monitor")
app.geometry("1280x720")
app.grid_rowconfigure(0, weight=1)
app.grid_columnconfigure((0, 1), weight=1)

# Graph Setup
def create_plot_frame(num_subplots=1):
    frame = ctk.CTkFrame(app, corner_radius=20, fg_color="#1a1a1a")
    frame.grid(row=0, column=create_plot_frame.col, sticky="nsew", padx=10, pady=10)
    app.grid_columnconfigure(create_plot_frame.col, weight=1)
    create_plot_frame.col += 1

    fig, axes = plt.subplots(num_subplots, 1, figsize=(6, 4))
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    if num_subplots == 1:
        axes = [axes]
    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.6)
    return axes, fig, canvas

create_plot_frame.col = 0

ax_acoustic, fig_acoustic, canvas_acoustic = create_plot_frame(1)
(ax_imu_x, ax_imu_y, ax_imu_z), fig_imu, canvas_imu = create_plot_frame(3)

# Network Thread
def data_listener():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        buffer = ''
        while True:
            buffer += s.recv(4096).decode('utf-8')
            while '\n' in buffer:
                packet, buffer = buffer.split('\n', 1)
                try:
                    data_packet = json.loads(packet)
                    data_queue.put(data_packet)
                except json.JSONDecodeError:
                    continue

def graph_updater():
    imu_data = {'x': [], 'y': [], 'z': []}
    acoustic_data = []

    while True:
        try:
            data_packet = data_queue.get(timeout=1)
            imu = data_packet['imu']
            acoustic = data_packet['acoustic']

            imu_data['x'].append(float(imu['Accel_X']))
            imu_data['y'].append(float(imu['Accel_Y']))
            imu_data['z'].append(float(imu['Accel_Z']))
            acoustic_data.append(float(acoustic['Amplitude']))

            for key in imu_data:
                imu_data[key] = imu_data[key][-100:]
            acoustic_data[:] = acoustic_data[-100:]

            def update():
                ax_acoustic[0].clear()
                ax_acoustic[0].plot(acoustic_data, 'b')
                ax_acoustic[0].axhline(y=np.mean(acoustic_data), color='r', linestyle='--')
                canvas_acoustic.draw_idle()

                for ax, data, color in zip([ax_imu_x, ax_imu_y, ax_imu_z], ['x', 'y', 'z'], ['r', 'g', 'b']):
                    ax.clear()
                    ax.plot(imu_data[data], color)
                    ax.axhline(y=np.mean(imu_data[data]), color='r', linestyle='--')
                canvas_imu.draw_idle()

            app.after(0, update)
            time.sleep(0.1)  # limit updates to ~10 FPS

        except queue.Empty:
            continue

threading.Thread(target=data_listener, daemon=True).start()
threading.Thread(target=graph_updater, daemon=True).start()

app.mainloop()
