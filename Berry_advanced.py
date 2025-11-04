import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import math
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
import csv
import os
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---------- Constants ----------
hbar = 1.0545718e-34
gamma_e = 1.760859644e11  # rad/s/T
e_charge = 1.602176634e-19
m_e = 9.10938356e-31

# ---------- Utilities ----------
def safe_eval_expr(expr, t):
    allowed = {
        't': float(t),
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'exp': math.exp, 'sqrt': math.sqrt, 'pi': math.pi,
        'np': np, 'math': math
    }
    try:
        return float(eval(expr, {"__builtins__": {}}, allowed))
    except Exception:
        return 0.0

def clamp_arr(a, lo, hi):
    a = np.array(a, dtype=float)
    return np.minimum(np.maximum(a, lo), hi)

# ---------- Waveform generators ----------
def waveform_value(kind, t, amp=1.0, freq=1.0, offset=0.0, duty=0.5, t0=0.0, tau=1.0, custom_expr="0"):
    omega = 2*np.pi*freq
    if kind == "Sinusoid":
        return offset + amp * math.sin(omega * (t - t0))
    elif kind == "Linear":
        if freq <= 0:
            return offset + amp * (t - t0)
        period = 1.0 / freq
        frac = ((t - t0) % period) / period
        return offset + amp * (2*frac - 1)
    elif kind == "Square":
        if freq <= 0:
            return offset
        period = 1.0 / freq
        frac = ((t - t0) % period) / period
        return offset + amp * (1.0 if frac < duty else -1.0)
    elif kind == "ExpDecay":
        dt = max(0.0, t - t0)
        return offset + amp * math.exp(-dt / max(1e-12, tau))
    elif kind == "Custom":
        return offset + amp * safe_eval_expr(custom_expr, t)
    else:
        return offset

# ---------- Physics helpers ----------
def berry_phase(theta):
    Omega = 2 * np.pi * (1 - np.cos(theta))
    gammaB = -0.5 * Omega
    return Omega, gammaB

def compute_omega(KX, KY, m):
    KX = np.asarray(KX, dtype=np.float64)
    KY = np.asarray(KY, dtype=np.float64)
    m = float(m)
    s = KX**2 + KY**2 + m*m
    s = np.where(s <= 0, 1e-30, s)
    denom = s**1.5
    denom = np.where(~np.isfinite(denom), 1e30, denom)
    Omega = -m / (2.0 * denom)
    Omega = np.clip(Omega, -1e6, 1e6)
    return Omega

def solid_angle_from_unit_vectors(rs):
    rs = np.array(rs, dtype=float)
    if rs.shape[0] < 2:
        return 0.0
    total = 0.0
    for i in range(len(rs)-1):
        a = rs[i]
        b = rs[i+1]
        cross = np.linalg.norm(np.cross(a, b))
        dot = np.dot(a, b)
        denom = 1.0 + dot
        total += 2.0 * math.atan2(max(0.0, cross), max(1e-16, denom))
    return float(total)

# ---------- RK4 integrators ----------
def rk4_step_vec(f, y, t, dt):
    k1 = f(y, t)
    k2 = f(y + 0.5*dt*k1, t + 0.5*dt)
    k3 = f(y + 0.5*dt*k2, t + 0.5*dt)
    k4 = f(y + dt*k3, t + dt)
    return y + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# ---------- GUI & Application ----------
root = tk.Tk()
root.title("Advanced Berry Phase Simulator (RK4 + Dynamic Curvature)")

notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

# ---------------- Tab A: Advanced Control / Bloch Dynamics ----------------
tab_adv = ttk.Frame(notebook)
notebook.add(tab_adv, text="Advanced Control / Bloch Dynamics")

# --- Scrollable control panel ---
ctrl_canvas = tk.Canvas(tab_adv, width=280)
ctrl_scroll = ttk.Scrollbar(tab_adv, orient="vertical", command=ctrl_canvas.yview)
ctrl_inner = ttk.Frame(ctrl_canvas, padding=5)

ctrl_inner.bind(
    "<Configure>",
    lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox("all"))
)

ctrl_canvas.create_window((0,0), window=ctrl_inner, anchor='nw')
ctrl_canvas.configure(yscrollcommand=ctrl_scroll.set)

ctrl_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
ctrl_scroll.pack(side=tk.LEFT, fill=tk.Y)

plot_frame = ttk.Frame(tab_adv)
plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# ---------- Controls ----------
wave_kinds = ["Sinusoid", "Linear", "Square", "ExpDecay", "Custom"]

ttk.Label(ctrl_inner, text="B(t) waveform").pack(anchor='w', pady=(4,0))
B_kind_var = tk.StringVar(value="Sinusoid")
ttk.OptionMenu(ctrl_inner, B_kind_var, B_kind_var.get(), *wave_kinds).pack(fill='x', pady=2)

ttk.Label(ctrl_inner, text="B amplitude (mT)").pack(anchor='w')
B_amp_var = tk.DoubleVar(value=10.0)
ttk.Entry(ctrl_inner, textvariable=B_amp_var).pack(fill='x', pady=2)

ttk.Label(ctrl_inner, text="B frequency (Hz)").pack(anchor='w')
B_freq_var = tk.DoubleVar(value=0.5)
ttk.Entry(ctrl_inner, textvariable=B_freq_var).pack(fill='x', pady=2)

ttk.Label(ctrl_inner, text="B offset (mT)").pack(anchor='w')
B_off_var = tk.DoubleVar(value=10.0)
ttk.Entry(ctrl_inner, textvariable=B_off_var).pack(fill='x', pady=2)

ttk.Label(ctrl_inner, text="m(t) waveform").pack(anchor='w', pady=(8,0))
m_kind_var = tk.StringVar(value="Sinusoid")
ttk.OptionMenu(ctrl_inner, m_kind_var, m_kind_var.get(), *wave_kinds).pack(fill='x', pady=2)

ttk.Label(ctrl_inner, text="m amplitude").pack(anchor='w')
m_amp_var = tk.DoubleVar(value=0.89)
ttk.Entry(ctrl_inner, textvariable=m_amp_var).pack(fill='x', pady=2)

ttk.Label(ctrl_inner, text="m frequency (Hz)").pack(anchor='w')
m_freq_var = tk.DoubleVar(value=0.1)
ttk.Entry(ctrl_inner, textvariable=m_freq_var).pack(fill='x', pady=2)

ttk.Label(ctrl_inner, text="m offset").pack(anchor='w')
m_off_var = tk.DoubleVar(value=0.5)
ttk.Entry(ctrl_inner, textvariable=m_off_var).pack(fill='x', pady=2)

ttk.Label(ctrl_inner, text="Custom expr (for Custom kind); use 't' as time").pack(anchor='w', pady=(8,0))
custom_B_expr = tk.StringVar(value="sin(2*pi*0.1*t)")
custom_m_expr = tk.StringVar(value="sin(2*pi*0.1*t)")
ttk.Entry(ctrl_inner, textvariable=custom_B_expr).pack(fill='x', pady=2)
ttk.Entry(ctrl_inner, textvariable=custom_m_expr).pack(fill='x', pady=2)

ttk.Label(ctrl_inner, text="Spin magnitude (visual scaling)").pack(anchor='w', pady=(8,0))
spin_var = tk.StringVar(value="1")
ttk.OptionMenu(ctrl_inner, spin_var, spin_var.get(), "1/2", "1", "3/2").pack(fill='x', pady=2)

ttk.Label(ctrl_inner, text="Relaxation τ (ms)").pack(anchor='w', pady=(8,0))
tau_var_adv = tk.DoubleVar(value=10.0)
ttk.Entry(ctrl_inner, textvariable=tau_var_adv).pack(fill='x', pady=2)

ttk.Label(ctrl_inner, text="RK4 dt (ms)").pack(anchor='w', pady=(8,0))
rk4_dt_var = tk.DoubleVar(value=8.0)
ttk.Entry(ctrl_inner, textvariable=rk4_dt_var).pack(fill='x', pady=2)

running_adv = False
def start_adv():
    global running_adv
    running_adv = True
def stop_adv():
    global running_adv
    running_adv = False

ttk.Button(ctrl_inner, text="▶ Start", command=start_adv).pack(fill='x', pady=4)
ttk.Button(ctrl_inner, text="⏸ Stop", command=stop_adv).pack(fill='x', pady=2)

record_adv_var = tk.BooleanVar(value=False)
ttk.Checkbutton(ctrl_inner, text="Record advanced traces", variable=record_adv_var).pack(anchor='w', pady=(4,0))

ttk.Button(ctrl_inner, text="💾 Export CSV", command=lambda: export_adv()).pack(fill='x', pady=4)

# ---------- Export data storage ----------
adv_records = []

def export_adv():
    if len(adv_records) == 0:
        messagebox.showwarning("No Data", "No recorded traces to export.")
        return

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    fname = f"advanced_traces_{timestamp}.csv"
    
    inputs = {
        'B_kind': B_kind_var.get(),
        'B_amp_mT': B_amp_var.get(),
        'B_freq_Hz': B_freq_var.get(),
        'B_offset_mT': B_off_var.get(),
        'm_kind': m_kind_var.get(),
        'm_amp': m_amp_var.get(),
        'm_freq_Hz': m_freq_var.get(),
        'm_offset': m_off_var.get(),
        'custom_B_expr': custom_B_expr.get(),
        'custom_m_expr': custom_m_expr.get(),
        'spin_choice': spin_var.get(),
        'tau_ms': tau_var_adv.get(),
        'rk4_dt_ms': rk4_dt_var.get()
    }

    with open(fname, "w", newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["# Input Parameters"])
        for k,v in inputs.items():
            w.writerow([k, v])
        w.writerow([])
        header = ["t_s", "B_mT", "m", "Sx", "Sy", "Sz", "kx", "ky", "x", "y"]
        w.writerow(header)
        for r in adv_records:
            w.writerow([float(x) for x in r])
    
    messagebox.showinfo("Saved", f"Saved advanced traces to {os.path.abspath(fname)}")

# ---------- Remaining program (plots, dynamics) unchanged ----------
# You can copy everything else from your original program below this point, including:
# - Bloch sphere plotting
# - Hybrid k/r-space plotting
# - RK4 update loop
# - spin_deriv(), hybrid_deriv()
# - update_advanced() loop
# - initialization of hyb_* variables
# - root.after() scheduling
# - root.geometry() and mainloop()


# ---------- Plot area: left = Bloch sphere, center = curvature map, right = real/k-space ----------
fig = plt.Figure(figsize=(12,5))
ax_bloch = fig.add_subplot(131, projection='3d')
ax_curv = fig.add_subplot(132)
ax_hybrid = fig.add_subplot(133)

# adjust spacing to avoid overlap
fig.subplots_adjust(left=0.06, right=0.95, top=0.94, bottom=0.06, wspace=0.22)

canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Setup static Bloch sphere
u,v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
X_sphere = np.cos(u)*np.sin(v)
Y_sphere = np.sin(u)*np.sin(v)
Z_sphere = np.cos(v)

ax_bloch.plot_surface(X_sphere, Y_sphere, Z_sphere, color='lightgray', alpha=0.25, linewidth=0)
ax_bloch.set_box_aspect([1,1,1])
ax_bloch.set_xlim([-1,1]); ax_bloch.set_ylim([-1,1]); ax_bloch.set_zlim([-1,1])
ax_bloch.set_title("Bloch sphere (spin vector)")

# K-grid for curvature map
_kmax = 2.0; _n = 160
kx = np.linspace(-_kmax, _kmax, _n)
ky = np.linspace(-_kmax, _kmax, _n)
KX, KY = np.meshgrid(kx, ky)

# initial curvature image (fixed bug: use .get() methods properly)
m_init = float(m_off_var.get()) + float(m_amp_var.get())*0.0
Omega0 = compute_omega(KX, KY, float(m_off_var.get()))
im_curv = ax_curv.imshow(Omega0, extent=[kx.min(),kx.max(),ky.min(),ky.max()], origin='lower', cmap='RdBu_r')
# improved colorbar placement using make_axes_locatable
divider = make_axes_locatable(ax_curv)
cax = divider.append_axes("right", size="5%", pad=0.1)
cb_curv = fig.colorbar(im_curv, cax=cax)
cb_curv.set_label("Ω(kx, ky)")
ax_curv.set_title("Berry curvature Ω(kx,ky)")
ax_curv.set_xlabel("kx"); ax_curv.set_ylabel("ky")

ax_hybrid.set_title("Real-space (left) and k-space (right)")
ax_hybrid.set_xlabel("x / kx")
ax_hybrid.set_ylabel("y / ky")
ax_hybrid.set_aspect('auto')

# ---------- Dynamic state ----------
S_vec = np.array([0.0, 0.0, 1.0], dtype=float)  # Bloch spin vector (visual)
hyb_k = np.array([0.1, 0.0], dtype=float)
hyb_r = np.array([0.0, 0.0], dtype=float)
t_adv = 0.0

# plotting handles & traces
bloquiver = None
bfield_quiver = None
bloctrace = []
B_hat_trace = []
hyb_trace_real = []
hyb_trace_k = []
hyb_real_line = None
hyb_k_line = None

# display text handles
berry_phase_text = None
chern_text = None

# ---------- Hybrid parameters ----------
hyb_kmax = _kmax
hyb_n = _n

# ---------- Advanced integrator & dynamics ----------
def spin_deriv(S, t, B_hat, omega_L, tau_s):
    """dS/dt for Bloch-like dynamics with simple relaxation towards B_hat direction."""
    cross = np.cross(S, B_hat)
    dS = omega_L * cross + (B_hat - S) / tau_s
    return dS

def hybrid_deriv(state, t, params):
    """Return derivatives [dkx/dt, dky/dt, dx/dt, dy/dt]."""
    kx_val, ky_val, x, y = state
    Bz = params['Bz']
    E_field = params['E']
    m_eff = params['m_eff']
    use_berry = params['use_berry']
    m_val = params['m']
    m_eff_safe = m_eff if m_eff > 1e-30 else 1e-30
    v_group = (hbar * np.array([kx_val, ky_val])) / m_eff_safe
    v_cross_B = np.array([-v_group[1]*Bz, v_group[0]*Bz])
    force_k = -e_charge * (E_field + v_cross_B)
    dk_dt = force_k / hbar
    Omega_k = compute_omega(np.array([[kx_val]]), np.array([[ky_val]]), m_val)[0,0]
    if not np.isfinite(Omega_k):
        Omega_k = 0.0
    if use_berry:
        v_anom = np.array([-Omega_k * dk_dt[1], Omega_k * dk_dt[0]])
    else:
        v_anom = np.array([0.0, 0.0])
    v_total = v_group + v_anom
    dk_dt = np.nan_to_num(dk_dt); dk_dt = np.clip(dk_dt, -1e12, 1e12)
    v_total = np.nan_to_num(v_total); v_total = np.clip(v_total, -1e8, 1e8)
    return np.array([dk_dt[0], dk_dt[1], v_total[0], v_total[1]])

# ---------- Main update loop for advanced tab ----------
UPDATE_MS = 50  # GUI update interval (ms)
def update_advanced():
    global t_adv, S_vec, bloquiver, bfield_quiver, bloctrace, B_hat_trace
    global hyb_k, hyb_r, hyb_trace_real, hyb_trace_k, berry_phase_text, chern_text

    dt_ms = rk4_dt_var.get()
    dt = max(1e-3, float(dt_ms))*1e-3  # seconds

    if running_adv:
        # compute B(t) and m(t) per waveform choices
        B_mT = waveform_value(B_kind_var.get(), t_adv,
                              amp=float(B_amp_var.get()), freq=float(B_freq_var.get()),
                              offset=float(B_off_var.get()), custom_expr=custom_B_expr.get())
        B_T = float(B_mT) * 1e-3

        # get cone angle (use theta_var if exists else default)
        try:
            theta_deg = float(theta_var.get())
        except Exception:
            theta_deg = 45.0
        theta_rad = math.radians(theta_deg)

        # rotate B around z using angular position from B frequency and global time
        phi = 2*np.pi*float(B_freq_var.get())*t_adv
        B_hat = np.array([math.sin(theta_rad)*math.cos(phi),
                          math.sin(theta_rad)*math.sin(phi),
                          math.cos(theta_rad)])
        omega_L = gamma_e * B_T

        # compute m(t)
        m_val = waveform_value(m_kind_var.get(), t_adv,
                               amp=float(m_amp_var.get()), freq=float(m_freq_var.get()),
                               offset=float(m_off_var.get()), custom_expr=custom_m_expr.get())

        # spin magnitude scaling
        s_choice = spin_var.get()
        spin_scale = 1.0
        if s_choice == "1/2":
            spin_scale = 0.5
        elif s_choice == "1":
            spin_scale = 1.0
        elif s_choice == "3/2":
            spin_scale = 1.5

        tau_s = max(1e-6, float(tau_var_adv.get())*1e-3)

        # RK4 step for S_vec; then normalize to chosen visual magnitude
        def fS(y, tloc):
            return spin_deriv(y, tloc, B_hat, omega_L, tau_s)

        S_vec = rk4_step_vec(fS, S_vec, t_adv, dt)
        normS = np.linalg.norm(S_vec)
        if normS > 0:
            S_vec = (S_vec / normS) * float(spin_scale)

        # Hybrid RK4 step
        params = {
            'Bz': float(hyb_Bz_var.get()) if 'hyb_Bz_var' in globals() else 0.5,
            'E': np.array([float(hyb_Ex_var.get()) if 'hyb_Ex_var' in globals() else 0.0,
                           float(hyb_Ey_var.get()) if 'hyb_Ey_var' in globals() else 0.0]),
            'm': float(m_val),
            'use_berry': True,
            'm_eff': float(hyb_meff_var.get())*m_e if 'hyb_meff_var' in globals() else m_e
        }
        state = np.array([hyb_k[0], hyb_k[1], hyb_r[0], hyb_r[1]])
        def fstate(y, tloc):
            return hybrid_deriv(y, tloc, params)
        state_new = rk4_step_vec(fstate, state, t_adv, dt)
        hyb_k = clamp_arr(state_new[0:2], -10*hyb_kmax, 10*hyb_kmax)
        hyb_r = clamp_arr(state_new[2:4], -1e3, 1e3)

        # update curvature map dynamically with new m_val (compute at current resolution; okay for moderate sizes)
        try:
            Omega_map = compute_omega(KX, KY, m_val)
            im_curv.set_data(Omega_map)
            vmax = np.nanmax(np.abs(Omega_map))
            if np.isfinite(vmax) and vmax > 0:
                im_curv.set_clim(-vmax, vmax)
        except Exception:
            Omega_map = compute_omega(KX, KY, m_val)

        # plotting updates: Bloch sphere arrows & trace
        # append traces
        bloctrace.append(S_vec.copy())
        if len(bloctrace) > 800:
            bloctrace = bloctrace[-800:]
        B_hat_trace.append(B_hat.copy())
        if len(B_hat_trace) > 2000:
            B_hat_trace = B_hat_trace[-2000:]

        # compute Berry phase (solid angle) from B_hat_trace
        Omega_now = solid_angle_from_unit_vectors(B_hat_trace)
        gamma_now = -0.5 * Omega_now  # radians

        # redraw Bloch sphere, B̂ arrow, S arrow, and traces
        ax_bloch.cla()
        ax_bloch.plot_surface(X_sphere, Y_sphere, Z_sphere, color='lightgray', alpha=0.25, linewidth=0)
        ax_bloch.set_box_aspect([1,1,1]); ax_bloch.set_xlim([-1,1]); ax_bloch.set_ylim([-1,1]); ax_bloch.set_zlim([-1,1])
        # cone path for B
        phi_line = np.linspace(0, 2*np.pi, 200)
        cx = np.sin(theta_rad) * np.cos(phi_line)
        cy = np.sin(theta_rad) * np.sin(phi_line)
        cz = np.cos(theta_rad) * np.ones_like(phi_line)
        ax_bloch.plot(cx, cy, cz, color='gray', lw=0.8, alpha=0.6)
        # arrows
        try:
            if bfield_quiver is not None:
                bfield_quiver.remove()
        except Exception:
            pass
        try:
            if bloquiver is not None:
                bloquiver.remove()
        except Exception:
            pass
        # draw B_hat (black) and spin S_vec (magenta)
        bfield_quiver = ax_bloch.quiver(0,0,0, B_hat[0], B_hat[1], B_hat[2], color='k', lw=2, length=1.0, normalize=True)
        bloquiver = ax_bloch.quiver(0,0,0, S_vec[0], S_vec[1], S_vec[2], color='magenta', lw=2, length=1.0, normalize=True)
        if len(bloctrace) > 1:
            tb = np.array(bloctrace)
            ax_bloch.plot(tb[:,0], tb[:,1], tb[:,2], color='magenta', lw=1.2)
        # legend
        ax_bloch.plot([], [], color='k', label='B̂ (field)')
        ax_bloch.plot([], [], color='magenta', label='S (spin)')
        ax_bloch.legend(loc='lower left', fontsize=8)
        # Berry phase text
        try:
            if berry_phase_text is not None:
                berry_phase_text.remove()
        except Exception:
            pass
        berry_phase_text = ax_bloch.text2D(0.02, 0.94, f"Berry phase = {gamma_now:.3f} rad ({math.degrees(gamma_now):.2f}°)",
                                           transform=ax_bloch.transAxes, fontsize=9, color='darkred', bbox=dict(facecolor='white', alpha=0.7))

        ax_bloch.set_title(f"Bloch sphere (t={t_adv:.2f}s) | m={m_val:.3f}")

        # hybrid real and k paths
        hyb_trace_real.append(hyb_r.copy())
        hyb_trace_k.append(hyb_k.copy())
        if len(hyb_trace_real) > 2000:
            hyb_trace_real = hyb_trace_real[-2000:]
            hyb_trace_k = hyb_trace_k[-2000:]

        ax_hybrid.cla()
        real_arr = np.array(hyb_trace_real) if len(hyb_trace_real)>0 else np.zeros((1,2))
        k_arr = np.array(hyb_trace_k) if len(hyb_trace_k)>0 else np.zeros((1,2))
        # plot real-space
        ax_hybrid.plot(real_arr[:,0], real_arr[:,1], '-k', lw=1.0, label='real (x,y)')
        ax_hybrid.scatter(real_arr[-1,0], real_arr[-1,1], c='r', s=30)
        # plot k-space dashed
        ax_hybrid.plot(k_arr[:,0], k_arr[:,1], '--', lw=1.0, label='k-space (kx,ky)')
        ax_hybrid.scatter(k_arr[-1,0], k_arr[-1,1], c='b', s=30)
        ax_hybrid.set_title("Real-space (solid) and k-space (dashed)")
        ax_hybrid.legend(fontsize=7)

        # compute & display Chern estimate on curvature panel
        try:
            dk = (kx[1]-kx[0])*(ky[1]-ky[0])
            chern_est = np.sum(Omega_map) * dk / (2*math.pi)
        except Exception:
            chern_est = 0.0
        try:
            if chern_text is not None:
                chern_text.remove()
        except Exception:
            pass
        chern_text = ax_curv.text(0.02, 0.96, f"Chern ≈ {chern_est:.4f}", transform=ax_curv.transAxes,
                                  fontsize=9, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # record if enabled
        if record_adv_var.get():
            adv_records.append([t_adv, B_mT, m_val, S_vec[0], S_vec[1], S_vec[2], hyb_k[0], hyb_k[1], hyb_r[0], hyb_r[1]])

        # advance time by dt (seconds)
        t_adv += dt

    # redraw canvas (even if not running allow interactive changes)
    try:
        canvas.draw_idle()
    except Exception:
        canvas.draw()

    root.after(UPDATE_MS, update_advanced)

# initialize update loop
root.after(200, update_advanced)

# ----------------- Keep previous tabs controls available by importing global names used in hybrid (if present) -----------------
try:
    hyb_Bz_var
except NameError:
    hyb_Bz_var = tk.DoubleVar(value=0.5)
    hyb_Ex_var = tk.DoubleVar(value=0.0)
    hyb_Ey_var = tk.DoubleVar(value=0.0)
    hyb_meff_var = tk.DoubleVar(value=1.0)

# Final window sizing & start
root.geometry("1300x700")
root.mainloop()
