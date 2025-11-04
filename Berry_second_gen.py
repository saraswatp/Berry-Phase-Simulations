import tkinter as tk
from tkinter import ttk
import numpy as np
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D

# ---------- Physics constants & helpers ----------
hbar = 1.0545718e-34   # J*s
gamma_e = 1.760859644e11  # rad s^-1 T^-1 (electron gyromagnetic ratio) - illustrative
# NOTE: units here are mixed for pedagogy; we use "B0" in Tesla and time in seconds.

def berry_phase(theta):
    """Return (Omega, gamma_Berry) for cone angle theta (radians)."""
    Omega = 2 * np.pi * (1 - np.cos(theta))
    gammaB = -0.5 * Omega
    return Omega, gammaB

# ---------- Spin evolution model ----------
def step_spin(S, B_hat, omega_L, tau, dt):
    """
    Evolve spin vector S (3-vector) for small time dt using:
      dS/dt = omega_L * (S x B_hat) + (B_hat - S) / tau
    - omega_L: Larmor frequency (rad/s)
    - tau: alignment timescale (s). Large tau -> slow alignment; small tau -> fast alignment.
    """
    cross = np.cross(S, B_hat)
    dS = omega_L * cross + (B_hat - S) / tau
    S_new = S + dS * dt
    # normalize for numerical stability
    norm = np.linalg.norm(S_new)
    if norm == 0:
        return S
    return S_new / norm

# ---------- GUI + plotting ----------
root = tk.Tk()
root.title("Berry Phase: Spin Precession & Phase Separation Demo")

# Left control panel
ctrl = ttk.Frame(root, padding=8)
ctrl.pack(side=tk.LEFT, fill=tk.Y)

# Controls
ttk.Label(ctrl, text="Cone angle θ (deg)").pack(anchor='w', pady=(2,0))
theta_var = tk.DoubleVar(value=45.0)
theta_scale = ttk.Scale(ctrl, from_=0.0, to=180.0, variable=theta_var, orient='horizontal')
theta_scale.pack(fill='x', pady=2)

ttk.Label(ctrl, text="Rotation freq ω_rot (Hz)").pack(anchor='w', pady=(6,0))
omega_rot_var = tk.DoubleVar(value=0.5)
omega_rot_scale = ttk.Scale(ctrl, from_=0.01, to=5.0, variable=omega_rot_var, orient='horizontal')
omega_rot_scale.pack(fill='x', pady=2)

ttk.Label(ctrl, text="B-field magnitude B0 (mT)").pack(anchor='w', pady=(6,0))
B0_var = tk.DoubleVar(value=10.0)  # milliTesla default
B0_scale = ttk.Scale(ctrl, from_=1.0, to=100.0, variable=B0_var, orient='horizontal')
B0_scale.pack(fill='x', pady=2)

ttk.Label(ctrl, text="Alignment timescale τ (ms)").pack(anchor='w', pady=(6,0))
tau_var = tk.DoubleVar(value=1.0)
tau_scale = ttk.Scale(ctrl, from_=0.01, to=20.0, variable=tau_var, orient='horizontal')
tau_scale.pack(fill='x', pady=2)

ttk.Label(ctrl, text="Time step dt (ms)").pack(anchor='w', pady=(6,0))
dt_var = tk.DoubleVar(value=10.0)
dt_scale = ttk.Scale(ctrl, from_=0.1, to=50.0, variable=dt_var, orient='horizontal')
dt_scale.pack(fill='x', pady=2)

# Run/Stop
running = False
def start():
    global running
    running = True
    animate()

def stop():
    global running
    running = False

ttk.Button(ctrl, text="▶ Start", command=start).pack(fill='x', pady=8)
ttk.Button(ctrl, text="⏸ Stop", command=stop).pack(fill='x', pady=2)

# Numeric readouts
phase_frame = ttk.Frame(ctrl)
phase_frame.pack(fill='x', pady=(10,0))
label_gb = ttk.Label(phase_frame, text="Berry phase: 0.00 rad")
label_gb.pack(anchor='w')
label_gb_deg = ttk.Label(phase_frame, text="Berry phase: 0.00°")
label_gb_deg.pack(anchor='w')
label_gdyn = ttk.Label(phase_frame, text="Dynamical phase: 0.000 rad")
label_gdyn.pack(anchor='w')
label_gtot = ttk.Label(phase_frame, text="Total phase: 0.000 rad")
label_gtot.pack(anchor='w')
label_lag = ttk.Label(phase_frame, text="Lag angle (deg): 0.0")
label_lag.pack(anchor='w')

ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=8)
ttk.Label(ctrl, text="Notes: (ω_rot << ω_L gives adiabatic following)").pack(anchor='w')

# Right: plotting area (2 subplots: 3D sphere + phase vs time)
fig = plt.Figure(figsize=(8,5), dpi=100)
ax3d = fig.add_subplot(121, projection='3d')
ax_phase = fig.add_subplot(122)
ax_phase.set_xlabel("time (s)")
ax_phase.set_ylabel("phase (rad)")
ax_phase.grid(True)
line_dyn, = ax_phase.plot([], [], label='dynamical phase')
line_tot, = ax_phase.plot([], [], label='total phase')
ax_phase.axhline(0, color='0.6', linestyle='--')
ax_phase.legend()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Initialize state
S = np.array([0.0, 0.0, 1.0])  # spin direction initially along z
time = 0.0
dyn_phase = 0.0
time_history = []
dyn_hist = []
tot_hist = []

def draw_bloch(S_vec, B_hat, theta_rad, phi):
    ax3d.clear()
    # sphere
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax3d.plot_surface(x, y, z, color='lightblue', alpha=0.25, linewidth=0)
    # axes
    ax3d.quiver(0,0,0,1.1,0,0, color='r', arrow_length_ratio=0.08)
    ax3d.quiver(0,0,0,0,1.1,0, color='g', arrow_length_ratio=0.08)
    ax3d.quiver(0,0,0,0,0,1.1, color='b', arrow_length_ratio=0.08)
    # B-field vector (black) based on theta and phi
    Bx = math.sin(theta_rad)*math.cos(phi)
    By = math.sin(theta_rad)*math.sin(phi)
    Bz = math.cos(theta_rad)
    ax3d.quiver(0,0,0, Bx, By, Bz, color='k', linewidth=2, arrow_length_ratio=0.08)
    ax3d.text(Bx*1.1, By*1.1, Bz*1.1, 'B', color='k')
    # spin vector S (magenta)
    ax3d.quiver(0,0,0, S_vec[0], S_vec[1], S_vec[2], color='magenta', linewidth=2, arrow_length_ratio=0.08)
    ax3d.text(S_vec[0]*1.05, S_vec[1]*1.05, S_vec[2]*1.05, 'S', color='magenta')
    # cone path
    phi_path = np.linspace(0, 2*np.pi, 200)
    x_path = math.sin(theta_rad)*np.cos(phi_path)
    y_path = math.sin(theta_rad)*np.sin(phi_path)
    z_path = math.cos(theta_rad)*np.ones_like(phi_path)
    ax3d.plot(x_path, y_path, z_path, 'k--', alpha=0.5)
    # draw theta arc from north pole to B
    # arc points (great circle slice) in plane phi=0 between z-axis and B vector
    arc_phi = np.linspace(0, theta_rad, 30)
    arc_x = np.sin(arc_phi)
    arc_y = np.zeros_like(arc_x)
    arc_z = np.cos(arc_phi)
    ax3d.plot(arc_x, arc_y, arc_z, color='orange', linewidth=2)
    ax3d.text(0.6, 0, 0.6, f"θ={np.degrees(theta_rad):.1f}°", color='orange')
    ax3d.set_xlim([-1.1, 1.1]); ax3d.set_ylim([-1.1,1.1]); ax3d.set_zlim([-1.1,1.1])
    ax3d.set_box_aspect([1,1,1])
    ax3d.set_title("Bloch sphere (B black, S magenta)")

def animate():
    global time, S, dyn_phase
    if not running:
        return
    # read controls
    theta_deg = float(theta_var.get())
    theta_rad = math.radians(theta_deg)
    omega_rot = 2 * math.pi * float(omega_rot_var.get())  # rad/s
    B0_mT = float(B0_var.get())
    B0 = B0_mT * 1e-3  # convert mT to T
    tau_ms = float(tau_var.get())
    tau = tau_ms * 1e-3
    dt_ms = float(dt_var.get())
    dt = dt_ms * 1e-3

    # instantaneous phi of rotating B
    phi = omega_rot * time
    # B_hat vector components
    B_hat = np.array([math.sin(theta_rad)*math.cos(phi), math.sin(theta_rad)*math.sin(phi), math.cos(theta_rad)])
    # Larmor freq (rad/s) ~ gamma_e * B
    omega_L = gamma_e * B0
    # evolve spin vector with our minimal model
    S = step_spin(S, B_hat, omega_L, tau, dt)

    # compute dynamical phase increment: E = - (ħ/2) * (gamma_e * B0)  (eigenenergy magnitude)
    # so d(gamma_dyn)/dt = -E/ħ = +(1/2) * gamma_e * B0   (note sign)
    # we integrate numerically:
    dg_dt = 0.5 * gamma_e * B0  # rad/s
    dyn_phase += dg_dt * dt

    # Berry is geometric (compute from theta only)
    Omega, gammaB = berry_phase(theta_rad)
    total_phase = dyn_phase + gammaB

    # update histories and plots
    time += dt
    time_history.append(time)
    dyn_hist.append(dyn_phase)
    tot_hist.append(total_phase)

    # update visualization
    draw_bloch(S, B_hat, theta_rad, phi)
    # update phase plot (simple sliding window)
    ax_phase.clear()
    ax_phase.plot(time_history, dyn_hist, label='dynamical phase')
    ax_phase.plot(time_history, [gammaB]*len(time_history), '--', label='Berry (const)')
    ax_phase.plot(time_history, tot_hist, label='total phase')
    ax_phase.legend(loc='upper left', fontsize='small')
    ax_phase.set_xlabel("time (s)"); ax_phase.set_ylabel("phase (rad)")
    ax_phase.grid(True)

    # numeric readouts
    label_gb.config(text=f"Berry phase: {gammaB:.6f} rad")
    label_gb_deg.config(text=f"Berry phase: {np.degrees(gammaB):.3f}°")
    label_gdyn.config(text=f"Dynamical phase: {dyn_phase:.6f} rad")
    label_gtot.config(text=f"Total phase: {total_phase:.6f} rad")
    # lag angle between S and B:
    coslag = np.clip(np.dot(S, B_hat) / (np.linalg.norm(S)*np.linalg.norm(B_hat)), -1.0, 1.0)
    lag_deg = np.degrees(np.arccos(coslag))
    label_lag.config(text=f"Lag angle (deg): {lag_deg:.3f}")

    canvas.draw()
    # schedule next frame
    if running:
        root.after(int(max(1, dt_ms)), animate)

# start/stop handlers use running flag
def start():
    global running
    running = True
    animate()

def stop():
    global running
    running = False

# Bind buttons already created earlier: reassign to use these local functions
# (If the earlier buttons are present, they call these functions; else user can start)
running = False

# Pre-draw
draw_bloch(S, np.array([0,0,1.0]), math.radians(float(theta_var.get())), 0.0)
canvas.draw()

root.mainloop()