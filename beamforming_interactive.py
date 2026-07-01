# -*- coding: utf-8 -*-
"""
beamforming_interactive.py — Interactive 2D radiation pattern of a phased-array
                              radar (Uniform Linear Array), with live sliders.

Move the sliders / radio buttons to change, in real time :
  - N        : number of antenna elements,
  - d        : inter-element spacing (in wavelengths lambda),
  - steering : main-lobe pointing angle (deg),
  - floor    : plot dynamic-range floor (dB),
  - taper    : amplitude weighting (windowing).

NOTE : interactive widgets need a GUI backend (TkAgg / QtAgg), NOT the inline
one. Run it as a plain script :  python beamforming_interactive.py
In VS Code / Spyder / Jupyter, first switch backend, e.g.  %matplotlib qt
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button


# --------------------------------------------------------------------------- #
# 1. PHYSICS CORE : weights + array factor + beam metrics
# --------------------------------------------------------------------------- #

def make_weights(n, taper):
    """Amplitude of the N elements for the chosen window (sum normalized to 1)."""
    taper = taper.lower()
    if taper in ("uniform", "rect", "none"):
        w = np.ones(n)
    elif taper == "hamming":
        w = np.hamming(n)
    elif taper in ("hann", "hanning"):
        w = np.hanning(n)
    elif taper == "blackman":
        w = np.blackman(n)
    elif taper in ("cheb", "chebyshev"):
        try:
            from scipy.signal.windows import chebwin
            w = chebwin(n, at=30.0)
        except ImportError:
            w = np.hamming(n)
    else:
        w = np.ones(n)
    s = w.sum()
    return w / s if s else w


def array_factor(theta_rad, n, d_lambda, beta_rad, weights):
    """Complex array factor of a ULA along x, evaluated at angles theta (from broadside)."""
    n_idx = np.arange(n)
    psi = 2.0 * np.pi * d_lambda * np.sin(theta_rad) + beta_rad
    return np.exp(1j * np.outer(psi, n_idx)) @ weights


def steering_phase(d_lambda, steer_deg):
    """Progressive phase beta (rad) that points the main lobe to steer_deg."""
    return -2.0 * np.pi * d_lambda * np.sin(np.deg2rad(steer_deg))


def _hpbw(theta_deg, db, i_peak, level=-3.0):
    left = right = np.nan
    for i in range(i_peak, 0, -1):
        if db[i] >= level >= db[i - 1]:
            left = np.interp(level, [db[i - 1], db[i]], [theta_deg[i - 1], theta_deg[i]])
            break
    for i in range(i_peak, len(db) - 1):
        if db[i] >= level >= db[i + 1]:
            right = np.interp(level, [db[i + 1], db[i]], [theta_deg[i + 1], theta_deg[i]])
            break
    return right - left if np.isfinite(left) and np.isfinite(right) else np.nan


def _sidelobe_level(db, i_peak):
    iL = i_peak
    while iL > 0 and db[iL - 1] <= db[iL]:
        iL -= 1
    iR = i_peak
    while iR < len(db) - 1 and db[iR + 1] <= db[iR]:
        iR += 1
    mask = np.ones(len(db), bool)
    mask[iL:iR + 1] = False
    return db[mask].max() if mask.any() else np.nan


# --------------------------------------------------------------------------- #
# 2. INITIAL PARAMETERS
# --------------------------------------------------------------------------- #

NAME   = "MPPAWR"
N0     = 16          # number of elements
D0     = 0.5         # spacing (lambda)
STEER0 = 20.0        # steering angle (deg)
FLOOR0 = -40.0       # dB floor
TAPER0 = "uniform"

N_THETA = 2001
theta_deg = np.linspace(-90.0, 90.0, N_THETA)
theta_rad = np.deg2rad(theta_deg)


def compute_db(n, d, steer, floor, taper):
    """Return the normalized pattern in dB plus (main_dir, hpbw, sll)."""
    n = int(n)
    weights = make_weights(n, taper)
    beta = steering_phase(d, steer)
    af = array_factor(theta_rad, n, d, beta, weights)
    mag = np.abs(af)
    mag = mag / mag.max()
    raw_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    i_peak = int(np.argmax(mag))
    main_dir = theta_deg[i_peak]
    hpbw = _hpbw(theta_deg, raw_db, i_peak)
    sll = _sidelobe_level(raw_db, i_peak)
    return np.clip(raw_db, floor, 0.0), main_dir, hpbw, sll


# --------------------------------------------------------------------------- #
# 3. FIGURE + WIDGETS
# --------------------------------------------------------------------------- #

fig = plt.figure(figsize=(13, 7))
fig.subplots_adjust(left=0.28, bottom=0.30, right=0.97, top=0.90, wspace=0.25)

ax_pol = fig.add_subplot(1, 2, 1, projection="polar")
ax_cart = fig.add_subplot(1, 2, 2)


def draw(n, d, steer, floor, taper):
    """Clear both axes and redraw the pattern for the current parameters."""
    db, main_dir, hpbw, sll = compute_db(n, d, steer, floor, taper)
    n = int(n)

    # (a) polar diagram
    ax_pol.clear()
    ax_pol.plot(theta_rad, db, color="C0", lw=1.6)
    ax_pol.fill(theta_rad, db, color="C0", alpha=0.12)
    ax_pol.set_theta_zero_location("N")
    ax_pol.set_theta_direction(-1)
    ax_pol.set_thetamin(-90)
    ax_pol.set_thetamax(90)
    ax_pol.set_rlim(floor, 0)
    ax_pol.set_rlabel_position(135)
    ax_pol.axvline(np.deg2rad(main_dir), color="C3", lw=1.2, ls="--")
    ax_pol.set_title("Radiation pattern (dB)", pad=18)

    # (b) cartesian dB cut
    ax_cart.clear()
    ax_cart.plot(theta_deg, db, color="C0", lw=1.6)
    ax_cart.axhline(-3.0, color="C2", lw=0.8, ls=":", label="-3 dB")
    ax_cart.axvline(main_dir, color="C3", lw=1.0, ls="--")
    if np.isfinite(sll):
        ax_cart.axhline(sll, color="C1", lw=0.8, ls=":", label=f"SLL {sll:.1f} dB")
    ax_cart.set_xlim(-90, 90)
    ax_cart.set_ylim(floor, 2)
    ax_cart.set_xlabel("angle from broadside (deg)")
    ax_cart.set_ylabel("normalized gain (dB)")
    ax_cart.set_xticks(np.arange(-90, 91, 30))
    ax_cart.grid(True, alpha=0.3)
    ax_cart.set_title("dB cut")
    hp = f"{hpbw:.1f}" if np.isfinite(hpbw) else "n/a"
    ax_cart.text(0.02, 0.04,
                 f"N = {n}   d = {d:.2f}$\\lambda$   taper = {taper}\n"
                 f"steering = {steer:+.1f}°   HPBW = {hp}°   SLL = {sll:.1f} dB",
                 transform=ax_cart.transAxes, fontsize=8, va="bottom",
                 bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    ax_cart.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Beamforming — {NAME}", fontsize=13)
    fig.canvas.draw_idle()


# ---- slider axes ----
ax_n     = fig.add_axes([0.35, 0.18, 0.55, 0.03])
ax_d     = fig.add_axes([0.35, 0.13, 0.55, 0.03])
ax_steer = fig.add_axes([0.35, 0.08, 0.55, 0.03])
ax_floor = fig.add_axes([0.35, 0.03, 0.55, 0.03])

s_n     = Slider(ax_n,     "N elements", 1, 64, valinit=N0, valstep=1)
s_d     = Slider(ax_d,     "d (λ)",      0.1, 1.5, valinit=D0, valstep=0.01)
s_steer = Slider(ax_steer, "steering (°)", -70, 70, valinit=STEER0, valstep=1)
s_floor = Slider(ax_floor, "floor (dB)", -70, -10, valinit=FLOOR0, valstep=1)

# ---- taper radio buttons ----
ax_taper = fig.add_axes([0.03, 0.30, 0.16, 0.22])
ax_taper.set_title("taper", fontsize=9)
r_taper = RadioButtons(ax_taper, ("uniform", "hamming", "hann", "blackman", "chebyshev"))

# ---- reset button ----
ax_reset = fig.add_axes([0.03, 0.05, 0.10, 0.05])
b_reset = Button(ax_reset, "Reset")


def update(_=None):
    draw(s_n.val, s_d.val, s_steer.val, s_floor.val, r_taper.value_selected)


def reset(_=None):
    s_n.reset(); s_d.reset(); s_steer.reset(); s_floor.reset()
    r_taper.set_active(0)        # -> "uniform" ; triggers update via its callback


for s in (s_n, s_d, s_steer, s_floor):
    s.on_changed(update)
r_taper.on_clicked(update)
b_reset.on_clicked(reset)

draw(N0, D0, STEER0, FLOOR0, TAPER0)     # first render

if __name__ == "__main__":
    plt.show()
