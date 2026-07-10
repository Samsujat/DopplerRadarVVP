"""
================================================================================
 Generation of SYNTHETIC dual-Doppler radar data (two radars) for the
 dual-Doppler analysis of Ray, Ziegler, Bumgarner & Serafin (1980)
================================================================================

Provides:
  - true_wind(x, y, z) : analytic "truth" wind field (u, v, w) for validation
  - generate(radar1_pos, radar2_pos) : radial velocities (vr1, vr2) sampled by
    the two radars on an MP-PAWR-like polar geometry, with Gaussian noise

Truth wind field:
  - mean flow veering with altitude
  - non-divergent Gaussian vortex (mesocyclone) centered in the northern lobe
  - divergence profile D(z): convergence at low levels, divergence aloft
    -> w(z) consistent with anelastic continuity (Eq. 4 of the paper)
"""

import numpy as np

RNG = np.random.default_rng(0)

# ---- Physical parameters ----
VT = 0.0          # hydrometeor terminal fall speed, W = w + Vt (Eq. 2)
H_RHO = 9.5       # density scale height (km) : rho ~ exp(-z/H_RHO)
KAPPA = 1.0 / H_RHO   # kappa = -d(ln rho)/dz  (km^-1), cf. Eq. (4)

# ---- Truth wind field parameters ----
Z_TOP = 12.0                       # km, top of the ECHO (data limit)
Z_W = 9.5                          # km, top of the CIRCULATION (w=0 above)
W0 = 4.0                           # updraft amplitude (m/s)
DIV_CENTER = np.array([20.0, 40.0])
VORTEX_CENTER = np.array([20.0, 40.0])
VORTEX_OMEGA = 1.2                 # (m/s)/km : solid-body rotation at the center
VORTEX_SIGMA = 8.0                 # km

# ---- Sampling geometry of each radar (MP-PAWR-like) ----
ELEVATIONS_DEG = np.array([0.5, 1.5, 2.4, 3.4, 4.3, 5.3, 6.2, 7.5, 8.7,
                           10.0, 12.0, 14.0, 16.7, 19.5])
AZIMUTHS_DEG = np.arange(0.0, 360.0, 1.0)   # 360 rays, 1 deg step
RANGES_KM = np.arange(1.0, 80.0, 0.5)       # gates from 1 to 80 km, 500 m step

NOISE_STD = 5.0   # Gaussian noise on vr (m/s) -> error propagation study


# w(z) is analytically zero AT THE GROUND AND AT Z_W < Z_TOP: both boundary
# conditions of analyses A and A' are then exact, and data remain available
# above Z_W, as assumed in the paper ("observations near the tropopause
# level"). The divergence D(z) follows from anelastic continuity (Eq. 4):
#   D = kappa*w - dw/dz
def w_profile(z):
    zc = np.clip(z, 0.0, Z_W)
    return W0 * np.sin(np.pi * zc / Z_W) ** 2


def D_profile(z):
    """Uniform horizontal divergence D(z) (m/s/km), consistent with w."""
    zc = np.clip(z, 0.0, Z_W)
    dwdz = np.where(np.asarray(z, dtype=float) < Z_W,
                    W0 * (np.pi / Z_W) * np.sin(2.0 * np.pi * zc / Z_W),
                    0.0)
    return KAPPA * w_profile(z) - dwdz


def true_wind(x, y, z):
    """(u, v, w) in m/s ; x, y, z in km."""
    u = 6.0 + 0.4 * z
    v = 2.0 + 0.2 * z
    # uniform divergence : u += D/2 (x-xc), v += D/2 (y-yc)
    D = D_profile(z)
    u = u + 0.5 * D * (x - DIV_CENTER[0])
    v = v + 0.5 * D * (y - DIV_CENTER[1])
    # Gaussian vortex (non divergent)
    xp, yp = x - VORTEX_CENTER[0], y - VORTEX_CENTER[1]
    g = np.exp(-(xp**2 + yp**2) / (2.0 * VORTEX_SIGMA**2))
    u = u - VORTEX_OMEGA * yp * g
    v = v + VORTEX_OMEGA * xp * g
    w = w_profile(z)
    return u, v, np.broadcast_to(w, np.shape(u)).copy()


def simulate_radar(radar_pos, elev_deg=ELEVATIONS_DEG, azim_deg=AZIMUTHS_DEG,
                   rng_km=RANGES_KM):
    """vr (n_elv, n_azim, n_rng) seen by a radar located at radar_pos.
    Same convention as the mono-Doppler script: theta = azimuth from North,
    x = r sin(theta) cos(phi), y = r cos(theta) cos(phi), z = r sin(phi).
    vr > 0 : target moving AWAY from the radar."""
    phi = np.deg2rad(elev_deg)[:, None, None]
    th = np.deg2rad(azim_deg)[None, :, None]
    r = rng_km[None, None, :]
    ex = np.sin(th) * np.cos(phi)
    ey = np.cos(th) * np.cos(phi)
    ez = np.sin(phi) * np.ones_like(th)
    x = radar_pos[0] + r * ex
    y = radar_pos[1] + r * ey
    z = radar_pos[2] + r * ez
    u, v, w = true_wind(x, y, z)
    vr = u * ex + v * ey + (w + VT) * ez
    vr = vr + RNG.normal(0.0, NOISE_STD, vr.shape)
    vr[np.broadcast_to(z, vr.shape) > Z_TOP] = np.nan   # no echo outside the cloud
    return vr


def generate(radar1_pos, radar2_pos):
    """Radial velocities (vr1, vr2) sampled by the two radars.
    radar1_pos, radar2_pos : (x, y, z) in km in the common frame."""
    vr1 = simulate_radar(radar1_pos)
    vr2 = simulate_radar(radar2_pos)
    return vr1, vr2
