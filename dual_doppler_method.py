# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# %%
# --------------------------------------------------------------------------- #
# 0. DATA SOURCES : REAL OR SYNTHETIC
# --------------------------------------------------------------------------- #

# USE_REAL_DATA = True  -> two real radars must be provided (VELOCITY_FILE_1, VELOCITY_FILE_2) + their relative position (RADAR2_POS)
# USE_REAL_DATA = False -> generate_dual_radar_data (known "true" wind field for validation)
USE_REAL_DATA = False

VERBOSE = False  

# ---- Geometry of the two radars in the common frame (km, origin = radar 1)
RADAR1_POS = np.array([0.0, 0.0, 0.05])    
RADAR2_POS = np.array([40.0, 0.0, 0.05])   
BASELINE = np.linalg.norm(RADAR2_POS[:2] - RADAR1_POS[:2])

# ---- Physical parameters
VT = 0.0                # hydrometeor fall speed (m/s) 
H_RHO = 9.5             # density scale height (km) : rho ~ exp(-z/H_RHO)
KAPPA = 1.0 / H_RHO     # kappa = -d(ln rho)/dz  (km^-1) 

# ---- Dual-Doppler geometric quality
BETA_MIN_DEG = 25.0   # minimum beam-crossing angle (lobes) or the 2x2 system is ill-conditioned

if USE_REAL_DATA:
    from Data_Radar.Wada_mppawr_online import mppawr

    VELOCITY_FILE_1 = None   # <-- file of the FIRST radar
    VELOCITY_FILE_2 = None   # <-- file of the SECOND radar

    RANGE_MAX_KM = 80.0

    def load_mppawr(path):
        radar = mppawr(path, switch=False)
        data = radar.read_data()                  # (azimuth, elevation, range)
        n_sect, n_elv, n_rng = data.shape
        elev = np.asarray(radar.elv_center, dtype=float)
        azim = (np.arange(n_sect) + 0.5) * (360.0 / n_sect)
        rng_km = (np.arange(n_rng) + 0.5) * radar.range_res / 1000.0
        radar.close()
        v = data.astype(float)
        v[v < -200.0] = np.nan
        v = -np.transpose(v, (1, 0, 2))           # -> (elevation, azimuth, range)
        keep = rng_km <= RANGE_MAX_KM
        return v[:, :, keep], elev, azim, rng_km[keep]

    vr1, ELEV1_DEG, AZIM1_DEG, RNG1_KM = load_mppawr(VELOCITY_FILE_1)
    vr2, ELEV2_DEG, AZIM2_DEG, RNG2_KM = load_mppawr(VELOCITY_FILE_2)
    true_wind = None
    print("Radar 1 : vr shape =", vr1.shape,
          "| valid = {:.1f} %".format(100 * np.mean(~np.isnan(vr1))))
    print("Radar 2 : vr shape =", vr2.shape,
          "| valid = {:.1f} %".format(100 * np.mean(~np.isnan(vr2))))

else:
    from generate_dual_radar_data import (generate, true_wind, NOISE_STD, VORTEX_CENTER, ELEVATIONS_DEG, AZIMUTHS_DEG, RANGES_KM)

    # Polar sampling geometry of both radars (defined in the generator)
    ELEV1_DEG = ELEV2_DEG = ELEVATIONS_DEG
    AZIM1_DEG = AZIM2_DEG = AZIMUTHS_DEG
    RNG1_KM = RNG2_KM = RANGES_KM

    vr1, vr2 = generate(RADAR1_POS, RADAR2_POS)
    print("Synthetic Data : vr1", vr1.shape, "| vr2", vr2.shape,
          f"| noise = {NOISE_STD} m/s")

# %%
# --------------------------------------------------------------------------- #
# 1. CARTESIAN ANALYSIS GRID 
# --------------------------------------------------------------------------- #

# Cartesian grid (x,y,z) 
DXY, DZ = 2.0, 1.0                      # horizontal / vertical spacing (km)
X_GRID = np.arange(6.0, 34.1, DXY)
Y_GRID = np.arange(6.0, 56.1, DXY)  
Z_GRID = np.arange(0.5, 9.6, DZ)
NX, NY, NZ = len(X_GRID), len(Y_GRID), len(Z_GRID)

XX, YY, ZZ = np.meshgrid(X_GRID, Y_GRID, Z_GRID, indexing="ij")  # (NX,NY,NZ)
print(f"Grid : {NX} x {NY} x {NZ} = {NX*NY*NZ} points, "
      f"dx=dy={DXY} km, dz={DZ} km")

# %%
# --------------------------------------------------------------------------- #
# 2. CRESSMAN INTERPOLATION 
# --------------------------------------------------------------------------- #

R_INFLUENCE = 2            # radius R of the influence sphere for Cressman interpolation (km)
MIN_OBS = 4                 # minimum number of echoes inside the influence sphere

# Converts polar coordinates (elev, azim, rng) to Cartesian (x,y,z) for a radar
def polar_to_xyz(radar_pos, elev_deg, azim_deg, rng_km):
    phi = np.deg2rad(elev_deg)[:, None, None]
    th = np.deg2rad(azim_deg)[None, :, None]
    r = rng_km[None, None, :]
    x = radar_pos[0] + r * np.sin(th) * np.cos(phi)
    y = radar_pos[1] + r * np.cos(th) * np.cos(phi)
    z = radar_pos[2] + r * np.sin(phi) * np.ones_like(th)
    # broadcasting to 1D arrays of points (N,) for cKDTree : (N,3) = (N,1) * (1,N) * (1,N)
    shp = np.broadcast_shapes(x.shape, y.shape, z.shape)
    return (np.broadcast_to(x, shp).ravel(),
            np.broadcast_to(y, shp).ravel(),
            np.broadcast_to(z, shp).ravel())

# Cressman interpolation of the radial velocity vr onto the Cartesian grid (XX, YY, ZZ)
def cressman_interp(radar_pos, vr, elev_deg, azim_deg, rng_km):

    # Convert polar coordinates and select valid points 
    px, py, pz = polar_to_xyz(radar_pos, elev_deg, azim_deg, rng_km)

    # flatten vr
    val = vr.ravel()

    # select valid points
    ok = np.isfinite(val)
    px, py, pz, val = px[ok], py[ok], pz[ok], val[ok]

    # Use a KD-Tree for fast neighbor search within the influence radius R_INFLUENCE
    tree = cKDTree(np.column_stack([px, py, pz]))
    grid_pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    neigh = tree.query_ball_point(grid_pts, R_INFLUENCE, workers=-1)

    # Cressman interpolation formula : weights = (R^2 - d^2) / (R^2 + d^2)
    out = np.full(grid_pts.shape[0], np.nan)
    for i, idx in enumerate(neigh):
        # skip points with too few neighbors
        if len(idx) < MIN_OBS:
            continue
        # compute squared distances to neighbors
        d2 = np.sum((tree.data[idx] - grid_pts[i])**2, axis=1)
        wgt = (R_INFLUENCE**2 - d2) / (R_INFLUENCE**2 + d2)
        sum_w = wgt.sum()
        # in case of only point at R distance, wgt=0 -> use uniform weights
        if sum_w > 0:
            out[i] = np.dot(wgt, val[idx]) / sum_w
    return out.reshape(NX, NY, NZ)

print("Cressman interpolation radar 1 ...")
V1 = cressman_interp(RADAR1_POS, vr1, ELEV1_DEG, AZIM1_DEG, RNG1_KM)
print("Cressman interpolation radar 2 ...")
V2 = cressman_interp(RADAR2_POS, vr2, ELEV2_DEG, AZIM2_DEG, RNG2_KM)
print("Coverage : V1 {:.1f} %, V2 {:.1f} %".format(
    100 * np.mean(np.isfinite(V1)), 100 * np.mean(np.isfinite(V2))))

# %%
# --------------------------------------------------------------------------- #
# 3. DUAL-DOPPLER GEOMETRY : u(x-xi)/Ri + v(y-yi)/Ri + W(z-zi)/Ri = Vi
# --------------------------------------------------------------------------- #

# Unit vectors of the radar beams : (a,b,c) = (dx,dy,dz)/Ri
def unit_vectors(radar_pos):
    dx = XX - radar_pos[0]
    dy = YY - radar_pos[1]
    dz = ZZ - radar_pos[2]
    Ri = np.sqrt(dx**2 + dy**2 + dz**2)            
    return dx / Ri, dy / Ri, dz / Ri

a1, b1, c1 = unit_vectors(RADAR1_POS)
a2, b2, c2 = unit_vectors(RADAR2_POS)

# Valid points
h1 = np.sqrt(a1**2 + b1**2)
h2 = np.sqrt(a2**2 + b2**2)
DET = a1 * b2 - b1 * a2 
BETA = np.degrees(np.arcsin(np.clip(np.abs(DET/np.maximum(h1 * h2, 1e-12)), 0, 1)))
GOOD_GEOM = BETA >= BETA_MIN_DEG

VALID = np.isfinite(V1) & np.isfinite(V2) & GOOD_GEOM & (np.abs(DET) > 1e-6)
print("Valid points (data + geometry) : {:.1f} %".format(100 * VALID.mean()))

# %%
# --------------------------------------------------------------------------- #
# 4. ANALYSIS A (upward integration) AND A' (downward integration)
# --------------------------------------------------------------------------- #

# Solves the 2x2 system to get u,v as a function of w at each point.
def solve_uv(w):
    W = w + VT
    r1 = V1 - W * c1
    r2 = V2 - W * c2
    u = (r1 * b2 - r2 * b1) / DET
    v = (a1 * r2 - a2 * r1) / DET
    u[~VALID] = np.nan
    v[~VALID] = np.nan
    return u, v

# Calculates the horizontal divergence D = du/dx + dv/dy
def horizontal_divergence(u, v):
    du_dx = np.gradient(u, X_GRID, axis=0)
    dv_dy = np.gradient(v, Y_GRID, axis=1)
    return du_dx + dv_dy

RHO = np.exp(-Z_GRID / H_RHO)          # normalized density profile

# Vertical integration of the anelastic continuity equation to get w from D
def integrate_w(D, direction="up", w_boundary=0.0):
    """direction='up'   : w(z=0) = w_boundary, upward integration (analysis A)
       direction='down' : w(z_T) = w_boundary, downward integration (analysis A')"""
    w = np.zeros((NX, NY, NZ))
    # The integration is performed column-wise, using the trapezoidal rule to account for the vertical variation of density (anelastic continuity).
    if direction == "up":
        w[:, :, 0] = w_boundary + 0.5 * Z_GRID[0] * (-D[:, :, 0])  # ground -> 1st level
        for k in range(NZ - 1):
            dz = Z_GRID[k + 1] - Z_GRID[k]
            flux = RHO[k] * w[:, :, k] - 0.5 * dz * (RHO[k] * D[:, :, k]
                                                     + RHO[k + 1] * D[:, :, k + 1])
            w[:, :, k + 1] = flux / RHO[k + 1]
    else:
        w[:, :, -1] = w_boundary
        for k in range(NZ - 1, 0, -1):
            dz = Z_GRID[k] - Z_GRID[k - 1]
            flux = RHO[k] * w[:, :, k] + 0.5 * dz * (RHO[k] * D[:, :, k]
                                                     + RHO[k - 1] * D[:, :, k - 1])
            w[:, :, k - 1] = flux / RHO[k - 1]
    return w


# Dual-Doppler analyses A and A' are implemented as an iteration : w -> (u,v) -> D -> w
def dual_doppler_analysis(direction="up", n_iter=12, tol=0.05):
    """Analysis A (direction='up') or A' (direction='down')"""
    w = np.zeros((NX, NY, NZ))
    col_ok = np.ones((NX, NY), dtype=bool)
    for it in range(n_iter):
        u, v = solve_uv(w)
        D = horizontal_divergence(u, v)
        w_new = integrate_w(D, direction=direction)
        dw = np.nanmax(np.abs((w_new - w)[col_ok, :]))
        w = w_new
        if VERBOSE:
            print(f"  iter {it + 1} : max|dw| = {dw:.3f} m/s")
        if dw < tol:
            break
    u, v = solve_uv(w)
    mask = VALID & col_ok[:, :, None]
    return (np.where(mask, u, np.nan), np.where(mask, v, np.nan),
            np.where(mask, w, np.nan))

print("Analysis A  (upward integration, w=0 at the ground) ...")
uA, vA, wA = dual_doppler_analysis(direction="up")
print("Analysis A' (downward integration, w=0 at the top) ...")
uAp, vAp, wAp = dual_doppler_analysis(direction="down")

# %%
# --------------------------------------------------------------------------- #
# 5. VALIDATION 
# --------------------------------------------------------------------------- #

ANALYSES = {"A (up)": (uA, vA, wA), "A' (down)": (uAp, vAp, wAp)}

uT, vT, wT = true_wind(XX, YY, ZZ)
uT = np.where(VALID, uT, np.nan); vT = np.where(VALID, vT, np.nan)
wT = np.where(VALID, wT, np.nan)

def rms(x):
        return np.sqrt(np.nanmean(x**2))

if true_wind is not None and VERBOSE:
    print("\n--- Statistics ---")
    hdr = f"{'Analysis':<10}{'u_bar':>7}{'v_bar':>7}{'w_bar':>7}" \
          f"{'rms(du+dv)':>12}{'rms(dw)':>9}"
    print(hdr); print("-" * len(hdr))
    for name, (u, v, w) in ANALYSES.items():
        print(f"{name:<10}{np.nanmean(u):>7.2f}{np.nanmean(v):>7.2f}"
              f"{np.nanmean(w):>7.2f}"
              f"{rms(u - uT) + rms(v - vT):>12.2f}{rms(w - wT):>9.2f}")
    print(f"{'TRUTH':<10}{np.nanmean(uT):>7.2f}{np.nanmean(vT):>7.2f}"
          f"{np.nanmean(wT):>7.2f}{'':>12}{'':>9}")

# %%
# --------------------------------------------------------------------------- #
# 7. VISUALIZATION
# --------------------------------------------------------------------------- #

# Which dual-Doppler analysis to display : "A", "A'" or "both"
SHOW_ANALYSIS = "A'"

_DISPLAY_KEYS = {"A": ["A (up)"], "A'": ["A' (down)"],
                 "both": ["A (up)", "A' (down)"]}
if SHOW_ANALYSIS not in _DISPLAY_KEYS:
    raise ValueError(f"SHOW_ANALYSIS must be one of {list(_DISPLAY_KEYS)}")
DISPLAY_ANALYSES = {k: ANALYSES[k] for k in _DISPLAY_KEYS[SHOW_ANALYSIS]}

# ---- Fig. 1 : geometry, dual-Doppler lobes (beam-crossing angle) ----
iz0 = int(np.argmin(np.abs(Z_GRID - 3.0)))
x_disp = np.arange(RADAR1_POS[0] - 30.0, RADAR2_POS[0] + 30.1, 1.0)
y_disp = np.arange(-60.0, 60.1, 1.0)
XD, YD = np.meshgrid(x_disp, y_disp, indexing="ij")

show_beta_map = False

if show_beta_map:

    def beta_map(x, y, z):
        """Beam-crossing angle (deg) of the two radars at height z."""
        ab = []
        for pos in (RADAR1_POS, RADAR2_POS):
            dx, dy, dz = x - pos[0], y - pos[1], z - pos[2]
            Ri = np.sqrt(dx**2 + dy**2 + dz**2)
            ab.append((dx / Ri, dy / Ri))
        (A1, B1), (A2, B2) = ab
        h = np.maximum(np.hypot(A1, B1) * np.hypot(A2, B2), 1e-12)
        return np.degrees(np.arcsin(np.clip(np.abs(A1 * B2 - B1 * A2) / h, 0, 1)))

    BETA_DISP = beta_map(XD, YD, Z_GRID[iz0])

    fig, ax = plt.subplots(figsize=(7, 6))
    pm = ax.pcolormesh(x_disp, y_disp, BETA_DISP.T, cmap="viridis",
                    shading="nearest", vmin=0, vmax=90)
    fig.colorbar(pm, ax=ax, label="crossing angle beta (deg)")
    ax.contour(x_disp, y_disp, BETA_DISP.T, levels=[BETA_MIN_DEG],
            colors="r", linewidths=1.5)
    # analysis grid domain
    ax.add_patch(plt.Rectangle((X_GRID[0], Y_GRID[0]),
                            X_GRID[-1] - X_GRID[0], Y_GRID[-1] - Y_GRID[0],
                            fill=False, edgecolor="w", lw=1.5, ls="--"))
    ax.annotate("analysis grid", (X_GRID[0] + 1, Y_GRID[-1] - 3), color="w", fontsize=8)
    for pos, name in [(RADAR1_POS, "Radar 1"), (RADAR2_POS, "Radar 2")]:
        ax.plot(*pos[:2], "w^", ms=10, mec="k")
        ax.annotate(name, pos[:2], textcoords="offset points", xytext=(5, -12),
                    color="w")
    ax.set_title(f"Dual-Doppler lobes (z = {Z_GRID[iz0]} km) — "
                f"red contour : beta = {BETA_MIN_DEG}°")
    ax.set_xlabel("X East (km)"); ax.set_ylabel("Y North (km)")
    ax.set_aspect("equal"); plt.tight_layout()

# ---- Fig. 2 : retrieved horizontal wind ----
# WIND_PLOT_STYLE = "vvp" -> unit-length arrows + speed color (no w color)
# WIND_PLOT_STYLE = "w"   -> w color + wind arrows scaled by magnitude
# REMOVE_MEAN = True  -> plot the wind ANOMALY V' = V - mean : the ~uniform mean
#                       flow dominates and is removed, so the vortex / divergence
#                       structure becomes visible (this is NOT the real wind).
# REMOVE_MEAN = False -> plot the REAL total wind V (mean kept) : physically exact,
#                       but the vortex is largely hidden under the mean flow.
WIND_PLOT_STYLE = "vvp"
REMOVE_MEAN = True

QUIVER_SCALE = 40   # same arrow size as the VVP plot ("vvp" style)
if REMOVE_MEAN:
    U_MEAN, V_MEAN = np.nanmean(uA), np.nanmean(vA)
else:
    U_MEAN, V_MEAN = 0.0, 0.0
_PRIME = "'" if REMOVE_MEAN else ""   # label : |V'| (anomaly) vs |V| (total)
fig, axes = plt.subplots(1, len(DISPLAY_ANALYSES),
                         figsize=(5.6 * len(DISPLAY_ANALYSES), 5.4),
                         sharey=True)

if WIND_PLOT_STYLE == "vvp":
    # shared color scale on the relative wind speed |V'| across analyses
    CMAX = np.nanmax([np.nanmax(np.hypot((u - U_MEAN)[:, :, iz0],
                                         (v - V_MEAN)[:, :, iz0]))
                      for u, v, _ in DISPLAY_ANALYSES.values()])
elif WIND_PLOT_STYLE == "w":
    CMAX = np.nanmax([np.nanmax(np.abs(w[:, :, iz0]))
                      for *_, w in DISPLAY_ANALYSES.values()])
else:
    raise ValueError("WIND_PLOT_STYLE must be 'vvp' or 'w'")

for ax, (name, (u, v, w)) in zip(np.atleast_1d(axes), DISPLAY_ANALYSES.items()):
    # wind to display : anomaly (mean removed) or real total wind (U_MEAN=0)
    ur, vr = (u - U_MEAN)[:, :, iz0], (v - V_MEAN)[:, :, iz0]
    if true_wind is not None:
        urT, vrT = (uT - U_MEAN)[:, :, iz0], (vT - V_MEAN)[:, :, iz0]
    if WIND_PLOT_STYLE == "vvp":
        # background = wind speed, arrows = direction only
        speed = np.hypot(ur, vr)
        pm = ax.pcolormesh(X_GRID, Y_GRID, speed.T, cmap="YlOrRd", alpha=0.65,
                           shading="nearest", vmin=0.0, vmax=CMAX)
        fig.colorbar(pm, ax=ax, label=f"|V{_PRIME}| (m/s)")
        norm = np.where(speed > 0, speed, np.nan)
        ax.quiver(XX[:, :, iz0], YY[:, :, iz0], ur / norm, vr / norm,
                  scale=QUIVER_SCALE, width=0.003, color="k")
        if true_wind is not None:
            normT = np.hypot(urT, vrT)
            normT = np.where(normT > 0, normT, np.nan)
            ax.quiver(XX[:, :, iz0], YY[:, :, iz0], urT / normT, vrT / normT,
                      scale=QUIVER_SCALE, width=0.0018, color="limegreen", alpha=0.9)
    else:  # "w" : background = vertical velocity w, arrows = relative wind (proportional)
        pm = ax.pcolormesh(X_GRID, Y_GRID, w[:, :, iz0].T, cmap="RdBu_r",
                           vmin=-CMAX, vmax=CMAX, shading="nearest")
        fig.colorbar(pm, ax=ax, label="w (m/s)")
        ax.quiver(XX[:, :, iz0], YY[:, :, iz0], ur, vr,
                  scale=120, width=0.004, color="k")
        if true_wind is not None:
            ax.quiver(XX[:, :, iz0], YY[:, :, iz0], urT, vrT,
                      scale=120, width=0.0018, color="limegreen", alpha=0.9)
    ax.set_title(f"Analysis {name} — z = {Z_GRID[iz0]} km")
    ax.set_xlabel("X East (km)"); ax.set_aspect("equal")
    ax.set_xlim(X_GRID[0] - 1, X_GRID[-1] + 1)
    ax.set_ylim(Y_GRID[0] - 1, Y_GRID[-1] + 1)
np.atleast_1d(axes)[0].set_ylabel("Y North (km)")
_kind = "wind anomaly (mean removed)" if REMOVE_MEAN else "total wind"
_annot = f"{_kind} — unit arrows" if WIND_PLOT_STYLE == "vvp" else _kind
np.atleast_1d(axes)[0].annotate(
    _annot + (" — black = retrieved, green = truth" if true_wind is not None else ""),
    (0.02, 0.02), xycoords="axes fraction", fontsize=8)
plt.tight_layout()

# ---- Fig. 3 : mean w(z) profiles — error propagation ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
styles = {"A (up)": "b--s", "A' (down)": "g--o", "B' (adj.)": "m--^"}
for name, (_, _, w) in DISPLAY_ANALYSES.items():
    ax1.plot(np.nanmean(w, axis=(0, 1)), Z_GRID, styles[name], label=name)
if true_wind is not None:
    ax1.plot(np.nanmean(wT, axis=(0, 1)), Z_GRID, "k-", lw=2, label="truth")
    for name, (_, _, w) in DISPLAY_ANALYSES.items():
        rmsz = np.sqrt(np.nanmean((w - wT)**2, axis=(0, 1)))
        ax2.plot(rmsz, Z_GRID, styles[name], label=name)
    ax2.set_xlabel("rms( w - w_true ) (m/s)")
    ax2.set_title("Error of w : the downward integration (A')\n"
                  "damps the error (Appendix)")
    ax2.grid(True); ax2.legend()
ax1.set_xlabel("mean w (m/s)"); ax1.set_ylabel("altitude (km)")
ax1.set_title("Mean vertical profile of w"); ax1.grid(True); ax1.legend()
plt.tight_layout()

plt.show()