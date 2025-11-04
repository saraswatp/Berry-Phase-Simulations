import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

# ----------------------------
# Compute Berry phase
# ----------------------------
def berry_phase(theta):
    omega = 2 * np.pi * (1 - np.cos(theta))  # solid angle
    gamma = -0.5 * omega
    return omega, gamma

# ----------------------------
# Draw Bloch sphere with field vector and path
# ----------------------------
def plot_bloch(ax, theta, phi):
    ax.clear()
    # Draw sphere
    u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.3, linewidth=0)

    # Draw axes
    ax.quiver(0, 0, 0, 1.2, 0, 0, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 1.2, 0, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, 1.2, color='b', arrow_length_ratio=0.1)

    # Draw field vector
    x_vec = np.sin(theta)*np.cos(phi)
    y_vec = np.sin(theta)*np.sin(phi)
    z_vec = np.cos(theta)
    ax.quiver(0, 0, 0, x_vec, y_vec, z_vec, color='black', linewidth=2)

    # Draw circular path at fixed theta
    phi_path = np.linspace(0, 2*np.pi, 200)
    x_path = np.sin(theta)*np.cos(phi_path)
    y_path = np.sin(theta)*np.sin(phi_path)
    z_path = np.cos(theta)*np.ones_like(phi_path)
    ax.plot(x_path, y_path, z_path, 'k--', alpha=0.6)

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1,1,1])
    ax.set_title("Bloch Sphere - Berry Phase Path")

# ----------------------------
# Update function (called during animation)
# ----------------------------
def update_plot(phi):
    theta = np.radians(float(theta_var.get()))
    omega, gamma = berry_phase(theta)
    plot_bloch(ax, theta, phi)
    canvas.draw()
    label_phase.config(text=f"Berry Phase: {np.degrees(gamma):.2f}° ({gamma:.4f} rad)")
    label_omega.config(text=f"Solid Angle Ω: {omega:.4f} sr")

# ----------------------------
# Animation function
# ----------------------------
def animate():
    global anim_running
    if anim_running:
        phi = animate.counter
        update_plot(phi)
        animate.counter += 0.05
        root.after(50, animate)

animate.counter = 0

def start_animation():
    global anim_running
    anim_running = True
    animate()

def stop_animation():
    global anim_running
    anim_running = False

# ----------------------------
# Tkinter GUI Setup
# ----------------------------
root = tk.Tk()
root.title("Berry Phase Visual Simulator")

# Frame for controls
control_frame = ttk.Frame(root)
control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

# Angle control
theta_var = tk.DoubleVar(value=45)
ttk.Label(control_frame, text="Angle θ (degrees)").pack()
theta_slider = ttk.Scale(control_frame, from_=0, to=180, variable=theta_var, orient='horizontal')
theta_slider.pack(pady=5)

# Start / Stop buttons
ttk.Button(control_frame, text="▶ Start Rotation", command=start_animation).pack(pady=5)
ttk.Button(control_frame, text="⏸ Stop Rotation", command=stop_animation).pack(pady=5)

# Info Labels
label_phase = ttk.Label(control_frame, text="Berry Phase: ")
label_phase.pack(pady=5)
label_omega = ttk.Label(control_frame, text="Solid Angle Ω: ")
label_omega.pack(pady=5)

# Formula display
formula = ttk.Label(
    control_frame,
    text="γ_Berry = -½ Ω = -π (1 - cos θ)",
    font=("Helvetica", 10, "italic")
)
formula.pack(pady=10)

# Matplotlib Figure
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
plot_bloch(ax, np.radians(theta_var.get()), 0)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

anim_running = False
root.mainloop()