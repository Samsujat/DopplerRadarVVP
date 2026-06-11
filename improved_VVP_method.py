
# %% Imports
import numpy as np
import matplotlib.pyplot as plt
import pyart

# %%
# --------------------------------------------------------------------------- #
# 0. DATA SOURCES : REAL OR SYNTHETIC
# --------------------------------------------------------------------------- #

# USE_REAL_DATA = True  -> read true radar data MP-PAWR
# USE_REAL_DATA = False -> generate_radar_data (known "true" wind field for validation)
USE_REAL_DATA = True

VERBOSE = False

# Filter parameters for rejecting non-physical restitutions
MAX_WIND = 100.0       # m/s 
MAX_COND = None        # max conditionning (None = disabled)

if USE_REAL_DATA:
    from Data_Radar.Wada_mppawr_online import mppawr

    VELOCITY_FILE = (
        "Data_Radar/20240711_Z_V_W_HV_MTI/20240711_00_VH_MTI/"
        "Volumes/HD-PCFSU3-A/RAW/2024.07.11/00/"
        "20240711_000000.00-00-PPI.RAW-VH_MTI.NSK-AUTO-LEM2.suita.dat.gz"
    )

    RANGE_MAX_KM = 80.0   # km

    radar = mppawr(VELOCITY_FILE, switch=False)
    data = radar.read_data()              # (n_sect, n_elv, n_rng) = (azimuth, elevation, range)
    n_sect, n_elv, n_rng = data.shape

    # Physical coordinates of the radar from the file, in degrees and km
    ELEVATIONS_DEG = np.asarray(radar.elv_center, dtype=float)
    AZIMUTHS_DEG = (np.arange(n_sect) + 0.5) * (360.0 / n_sect)
    RANGES_KM = (np.arange(n_rng) + 0.5) * radar.range_res / 1000.0
    radar.close()

    # Physical limitations
    vr = data.astype(float)
    vr[vr < -200.0] = np.nan
    # Organization -> (azimuth, elevation, range) -> (elevation, azimuth, range)
    vr = -np.transpose(vr, (1, 0, 2))

    # range limitation 
    keep = RANGES_KM <= RANGE_MAX_KM
    RANGES_KM = RANGES_KM[keep]
    vr = vr[:, :, keep]

    true_wind = None   
    print("True Radar Data : vr shape =", vr.shape,
          "| valid = {:.1f} %".format(100 * np.mean(~np.isnan(vr))))
else:
    from generate_radar_data import generate, true_wind

    # Physical coordinates of the radar (synthetic case)
    ELEVATIONS_DEG = np.array([0.5, 1.5, 2.4, 3.4, 4.3, 5.3, 6.2, 7.5, 8.7, 10.0, 12.0, 14.0, 16.7, 19.5])
    AZIMUTHS_DEG = np.arange(0.0, 360.0, 1.0)        # 360 steps, 1° step range
    RANGES_KM = np.arange(2.0, 200.0, 0.5)           # from 2 to 200 km, 500 m step range

    # generate() returns a flat array -> reshape to (n_elv, n_azim, n_rng)
    vr = generate().reshape(len(ELEVATIONS_DEG), len(AZIMUTHS_DEG), len(RANGES_KM))


# %%
# --------------------------------------------------------------------------- #
# 1. RADAR COORDINATES : 1D axes in radians (regular grid -> no full meshgrid)
# --------------------------------------------------------------------------- #

ELEV_RAD = np.deg2rad(ELEVATIONS_DEG)   # axis 0 of vr
AZIM_RAD = np.deg2rad(AZIMUTHS_DEG)     # axis 1 of vr
# RANGES_KM                             # axis 2 of vr

# --------------------------------------------------------------------------- #
# 2. G MATRIX CONSTRUCTION : G (N x 9) from (r, theta, phi) of a volume,
# --------------------------------------------------------------------------- #

def design_matrix(r, theta, phi, x0, y0, z0):

    cphi, sphi = np.cos(phi), np.sin(phi)
    sth, cth = np.sin(theta), np.cos(theta)

    dx = r * sth * cphi - x0
    dy = r * cth * cphi - y0
    dz = r * sphi - z0
 
    G = np.empty((r.size, 9))
    G[:, 0] = sth * cphi              # df1 -> u0
    G[:, 1] = cth * cphi              # df2 -> v0
    G[:, 2] = sphi                    # df3 -> w0 

    G[:, 3] = dx * sth * cphi         # df4 -> u'x
    G[:, 4] = dy * sth * cphi         # df5 -> u'y
    G[:, 5] = dz * sth * cphi         # df6 -> u'z

    #G[:, 6] = dx * cth * cphi         # df7 -> v'x  
    G[:, 6] = dy * cth * cphi         # df8 -> v'y
    G[:, 7] = dz * cth * cphi         # df9 -> v'z

    G[:, 8] = dz * sphi               # df10 -> w'z  
    #G[:, 10] = dy * sphi              # df11 -> w'y  
    #G[:, 11] = dx * sphi              # df12 -> w'x  
    return G


# --------------------------------------------------------------------------- #
# 3. SYSTEME RESOLUTION : A X = B  -> X = (A^-1) B
# --------------------------------------------------------------------------- #

def solve(G, d, method="ridge", lam=0.1):
    if method == "direct":
        A = np.dot(G.T, G)
        B = np.dot(G.T, d)
        X = np.linalg.solve(A, B)
        return X, np.linalg.cond(A)
    if method == "ridge":
        U, s, Vt = np.linalg.svd(G, full_matrices=False)
        f = s / (s**2 + lam**2)            # facteurs de filtre de Tikhonov
        X = Vt.T @ (f * (U.T @ d))
        cond = s[0] / s[-1] if s[-1] > 0 else np.inf
        return X, cond
    
# --------------------------------------------------------------------------- #
# 4. OUTPUT FOR 1 POINT : retrieve_wind(x0, y0, z0) -> (u, v, w) OR None
# --------------------------------------------------------------------------- #

# Half-widths of the volume for selecting the data (km, deg, deg)
D_R     = 10.0    # km
D_THETA = 10.0   # deg
D_PHI   = 15.0   # deg

min_n = 30  # minimum number of valid data points in the volume to attempt restitution

def retrieve_wind(x0, y0, z0):

    # ---- Step 1 : (x0, y0, z0) is given as an argument ----

    # ---- Step 2 : convert to polar coordinates ----
    r0  = np.sqrt(x0**2 + y0**2 + z0**2)        # km
    th0 = np.arctan2(x0, y0)                     # azimut (rad)
    ph0 = np.arcsin(z0 / r0)                     # elevation (rad)

    # ---- Step 3 : select data within the volume ----
    ie = np.nonzero(np.abs(ELEV_RAD - ph0) < np.deg2rad(D_PHI))[0]

    # azimuth distance (taking into account periodicity)
    dth = (AZIM_RAD - th0) % (2 * np.pi)
    dth = np.minimum(dth, 2 * np.pi - dth)
    ia = np.nonzero(dth < np.deg2rad(D_THETA))[0]

    ir = np.nonzero(np.abs(RANGES_KM - r0) < D_R)[0]

    if ie.size == 0 or ia.size == 0 or ir.size == 0:
        return None

    sub = vr[np.ix_(ie, ia, ir)]      # small sub-volume of radial velocities
    valid = ~np.isnan(sub)
    n = np.count_nonzero(valid)
    if n < min_n:
        return None

    ph_sel, th_sel, r_sel = np.meshgrid(ELEV_RAD[ie], AZIM_RAD[ia], RANGES_KM[ir], indexing="ij")
    r_sel  = r_sel[valid]
    th_sel = th_sel[valid]
    ph_sel = ph_sel[valid]
    vr_sel = sub[valid]
 
    # ---- Step 4 : construction of G ----
    G = design_matrix(r_sel, th_sel, ph_sel, x0, y0, z0)
 
    # ---- Step 5 & 6 : formation of A X = B  and resolution ----
    try:
        X, cond = solve(G, vr_sel)
    except np.linalg.LinAlgError:
        return None
    if VERBOSE:
         print(f"Point ({x0:.1f}, {y0:.1f}, {z0:.1f}) km : n={n}, cond={cond:.2e}")
 
    # ---- Step 7 : Wind reconstruction ----
    u0, v0, w0     = X[0], X[1], X[2]
    #ux, uy, uz  = X[3], X[4], X[5]
    #vy, vz  = X[6], X[7]
    wz = X[8]

    # ---- Filter out unrealistic wind estimates ----
    # Speed excessive
    if not np.all(np.abs([u0, v0, w0]) <= MAX_WIND):
        return None
    # Badly conditioned system
    if MAX_COND is not None and cond > MAX_COND:
        return None

    if VERBOSE:
        print(f"ux={ux:.3e}, uy={uy:.3e}, uz={uz:.3e}")
        print(f"vy={vy:.3e}, vz={vz:.3e}, wz={wz:.3e}")

    return u0, v0, w0, wz

# %%
# --------------------------------------------------------------------------- #
# 5. SWEEPING THE WORKSPACE : sweep_workspace(x_vals, y_vals, z_vals) -> (xs, ys, zs, winds)
# --------------------------------------------------------------------------- #

def sweep_workspace(x_vals, y_vals, z_vals, **kwargs):
    xs, ys, zs, winds = [], [], [], []
    for z0 in z_vals:
        for y0 in y_vals:
            for x0 in x_vals:
                res = retrieve_wind(float(x0), float(y0), float(z0), **kwargs)
                if res is not None:
                    xs.append(x0); ys.append(y0); zs.append(z0)
                    winds.append(res)
    return (np.array(xs), np.array(ys), np.array(zs), np.array(winds))
 
 
# workspace grid for sweeping 
X_GRID = np.arange(-90.0, 90.0, 10.0)
Y_GRID = np.arange(-90.0, 90.0, 10.0)
Z_GRID = np.arange(1.0, 9.0, 2.0)
 
xs, ys, zs, winds = sweep_workspace(X_GRID, Y_GRID, Z_GRID)
print(f"Restitute points : {len(xs)} / {len(X_GRID) * len(Y_GRID) * len(Z_GRID)}")

if len(winds) > 0:
    U_rec, V_rec, W_rec = winds[:, 0], winds[:, 1], winds[:, 2]
else:
    print("No valid restitutions found in the workspace. Please adjust the grid parameters or volume dimensions.")

# %%
# --------------------------------------------------------------------------- #
# 6. VISUALIZATION : retrieved wind on 4 layers (2x2) + vertical profiles
# --------------------------------------------------------------------------- #

X_GRID   = np.arange(-90.0, 90.0, 5.0)
Y_GRID   = np.arange(-90.0, 90.0, 5.0)
Z_LAYERS = [1.0, 3.0, 5.0, 7.0]   # 4 selected layers (km), shown as a 2x2 grid
QUIVER_SCALE = 40         # scale for unit-length quiver arrows

# --- data collection for each layer : 2D grids (NaN = no retrieval) ---
data = {}
for z0 in Z_LAYERS:
    U = np.full((len(Y_GRID), len(X_GRID)), np.nan)
    V = np.full((len(Y_GRID), len(X_GRID)), np.nan)
    for iy, y0 in enumerate(Y_GRID):
        for ix, x0 in enumerate(X_GRID):
            res = retrieve_wind(float(x0), float(y0), z0)
            if res is not None:
                U[iy, ix] = res[0]
                V[iy, ix] = res[1]
    data[z0] = (U, V)

# --- Plotting : 2x2 layers ---
fig, axes = plt.subplots(2, 2, figsize=(13, 11))

XX, YY = np.meshgrid(X_GRID, Y_GRID)
# common color scale for the 4 layers
all_speeds = np.concatenate([np.hypot(U, V).ravel() for U, V in data.values()])
SPEED_MAX = np.nanmax(all_speeds) if np.any(~np.isnan(all_speeds)) else 1.0

# Radar visibility at altitude z : annulus bounded by max range / lowest
# elevation (outer) and the cone of silence at highest elevation (inner)
R_MAX = RANGES_KM[-1]
ELV_MIN_RAD, ELV_MAX_RAD = ELEV_RAD.min(), ELEV_RAD.max()
circle_th = np.linspace(0.0, 2.0 * np.pi, 200)

for ax, z0 in zip(axes.ravel(), Z_LAYERS):
    U, V = data[z0]
    speed = np.hypot(U, V)
    n_pts = int(np.count_nonzero(~np.isnan(speed)))

    pm = ax.pcolormesh(XX, YY, speed, cmap="YlOrRd", alpha=0.65,
                       shading="nearest", vmin=0.0, vmax=SPEED_MAX)
    fig.colorbar(pm, ax=ax, label="|V| (m/s)")

    # unit-length arrows : direction only, magnitude is in the background color
    norm = np.where(speed > 0, speed, np.nan)
    ax.quiver(XX, YY, U / norm, V / norm, scale=QUIVER_SCALE,
              width=0.003, color="k")
    ax.scatter([0], [0], color="k", marker="^", s=60)

    s_outer = np.sqrt(max(R_MAX**2 - z0**2, 0.0))
    if ELV_MIN_RAD > 0:   # lowest-beam cap only meaningful for positive elevation
        s_outer = min(s_outer, z0 / np.tan(ELV_MIN_RAD))
    s_inner = z0 / np.tan(ELV_MAX_RAD)
    ax.plot(s_outer * np.cos(circle_th), s_outer * np.sin(circle_th),
            "r-", lw=1.3, label="limite visibilité radar")
    if s_inner < s_outer:
        ax.plot(s_inner * np.cos(circle_th), s_inner * np.sin(circle_th),
                "r--", lw=1.0, label="cône de silence")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Restitué VVP — z = {z0} km ({n_pts} pts)")
    ax.set_xlabel("X Est (km)"); ax.set_ylabel("Y Nord (km)")
    ax.set_xlim(X_GRID[0] - 10, X_GRID[-1] + 10)
    ax.set_ylim(Y_GRID[0] - 10, Y_GRID[-1] + 10)
    ax.set_aspect("equal")

plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.25)

# --- Vertical profiles at selected points ---
z_profile = np.arange(1.0, 9.0, 0.5)
profile_points = [(0.0, 60.0), (40.0, 40.0)]
fig2, axes2 = plt.subplots(1, len(profile_points), figsize=(6 * len(profile_points), 5))

for ax, (px, py) in zip(np.atleast_1d(axes2), profile_points):
    w_rec, wz_rec = [], []
    for z0 in z_profile:
        res = retrieve_wind(px, py, float(z0))
        w_rec.append(res[2] if res is not None else np.nan)
        wz_rec.append(res[3] if res is not None else np.nan)
    line_w, = ax.plot(w_rec, z_profile, "b--s", label="w restitué")
    ax.set_title(f"Profil w(z) au point ({px:.0f}, {py:.0f}) km")
    ax.set_xlabel("w (m/s)"); ax.set_ylabel("altitude (km)")
    ax.grid(True)

    # w'z (s^-1) on a secondary x-axis: different unit and magnitude than w
    ax2 = ax.twiny()
    line_wz, = ax2.plot(wz_rec, z_profile, "r--o", label="w'z restitué")
    ax2.set_xlabel("w'z (s$^{-1}$)", color="r")
    ax2.tick_params(axis="x", labelcolor="r")

    ax.legend(handles=[line_w, line_wz], loc="best")

plt.tight_layout()
plt.show()

# %%
# --------------------------------------------------------------------------- #
# 7. PPI VIEW OF THE RAW RADIAL VELOCITY (Py-ART)
# --------------------------------------------------------------------------- #
# Requires the project .venv (Python 3.12 + arm_pyart) :
# in VSCode, select the ".venv" interpreter/kernel for this file.

PPI_SWEEPS = [2, 4, 6, 8]   # indices of the elevation sweeps to display (0 .. n_elv-1)
VEL_LIM = 30.0                 # color scale limit (m/s)

n_elv, n_azim, n_rng = vr.shape

# Build a Py-ART Radar object from the (elevation, azimuth, range) cube
radar_ppi = pyart.testing.make_empty_ppi_radar(n_rng, n_azim, n_elv)
radar_ppi.range["data"] = RANGES_KM * 1000.0                       # m
radar_ppi.azimuth["data"] = np.tile(AZIMUTHS_DEG, n_elv)
radar_ppi.elevation["data"] = np.repeat(ELEVATIONS_DEG, n_azim)
radar_ppi.fixed_angle["data"] = ELEVATIONS_DEG.copy()
radar_ppi.fields["velocity"] = {
    "data": np.ma.masked_invalid(vr.reshape(n_elv * n_azim, n_rng)),
    "units": "m/s",
    "long_name": "Radial velocity (towards radar positive)",
    "standard_name": "radial_velocity",
}

display = pyart.graph.RadarDisplay(radar_ppi)
fig3, axes3 = plt.subplots(2, 2, figsize=(13, 11))
for ax, sw in zip(axes3.ravel(), PPI_SWEEPS):
    display.plot_ppi("velocity", sweep=sw, ax=ax, fig=fig3,
                     vmin=-VEL_LIM, vmax=VEL_LIM, cmap="RdBu_r",
                     colorbar_label="vr (m/s)",
                     title=f"PPI vitesse radiale — élévation {ELEVATIONS_DEG[sw]:.1f}°")
    display.plot_range_rings([20, 40, 60, 80], ax=ax, lw=0.5, ls="--")
    ax.set_aspect("equal")

plt.tight_layout()
plt.show()