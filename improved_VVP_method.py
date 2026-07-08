
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

VERBOSE = True

# Methods for computing the vertical velocity w ; one plot panel per entry :
# "classic"         -> w = w0, taken directly from the linear VVP fit. Simple,
#                      but w projects weakly onto vr (especially at low
#                      elevation) so it is poorly constrained / noisy.
# "mass_continuity" -> w is rebuilt by vertically integrating the anelastic
#                      continuity equation d(rho w)/dz = -rho (du/dx + dv/dy),
#                      with rho = rho0 * exp(-z / H_RHO), using the horizontal
#                      divergence (du/dx, dv/dy) from the fit, which projects
#                      strongly onto vr and is well constrained. Assumes
#                      w = 0 at the base (lowest z) of the column being computed.
# Only mass_continuity is shown ; add "classic" back here to get the
# side-by-side comparison view again.
W_METHODS = ("mass_continuity",)

H_RHO = 9.5   # density scale height (km) : rho = rho0 * exp(-z / H_RHO)

# Filter parameters for rejecting non-physical restitutions
# MAX_WIND is set after data loading : a retrieved wind cannot legitimately
# exceed the radial velocities it is fitted on (margin 1.5x on the p99.9)
MAX_COND = None        # max conditionning (None = disabled)

if USE_REAL_DATA:
    from Data_Radar.Wada_mppawr_online import mppawr

    VELOCITY_FILE = (
        "Data_Radar/20240711_Z_V_W_HV_MTI/20240711_01_VH_MTI/"
        "Volumes/HD-PCFSU3-A/RAW/2024.07.11/01/"
        "20240711_010000.00-00-PPI.RAW-VH_MTI.NSK-AUTO-LEM2.suita.dat.gz"
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
    # The sampling geometry (elevations, azimuths, ranges) is defined by the
    # generator and depends on its radar_type ('doppler' or 'PAWR'). Importing it
    # here keeps the VVP grid consistent with whatever radar is simulated, so the
    # reshape below always matches generate()'s output size.
    from generate_radar_data import (generate, true_wind,
                                      ELEVATIONS_DEG, AZIMUTHS_DEG, RANGES_KM)

    # generate() returns a flat array -> reshape to (n_elv, n_azim, n_rng)
    vr = generate().reshape(len(ELEVATIONS_DEG), len(AZIMUTHS_DEG), len(RANGES_KM))

# Data-driven threshold for rejecting diverging fits (aliasing, clutter...)
MAX_WIND = 1.5 * np.nanpercentile(np.abs(vr), 99.9)   # m/s
print(f"MAX_WIND = {MAX_WIND:.1f} m/s (1.5 x p99.9 des |vr|)")

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
 
    G = np.empty((r.size, 6))
    G[:, 0] = sth * cphi              # df1 -> u0
    G[:, 1] = cth * cphi              # df2 -> v0
    G[:, 2] = sphi                    # df3 -> w0 

    G[:, 3] = dx * sth * cphi         # df4 -> u'x
    #G[:, 4] = dy * sth * cphi         # df5 -> u'y
    #G[:, 5] = dz * sth * cphi         # df6 -> u'z

    #G[:, 6] = dx * cth * cphi         # df7 -> v'x  
    G[:, 4] = dy * cth * cphi         # df8 -> v'y
    #G[:, 8] = dz * cth * cphi         # df9 -> v'z

    G[:, 5] = dz * sphi               # df10 -> w'z  
    #G[:, 10] = dy * sphi              # df11 -> w'y  
    #G[:, 11] = dx * sphi              # df12 -> w'x  
    return G


# --------------------------------------------------------------------------- #
# 3. SYSTEME RESOLUTION : A X = B  -> X = (A^-1) B
# --------------------------------------------------------------------------- #

def solve(G, d, method="direct", lam=0.1, rcond=1e-10):
    if method == "direct":
        A = np.dot(G.T, G)
        B = np.dot(G.T, d)
        X = np.linalg.solve(A, B)
        return X, np.linalg.cond(A)
    if method == "svd":
        X, _res, _rank, sv = np.linalg.lstsq(G, d, rcond=rcond)
        cond = sv[0] / sv[-1] if sv[-1] > 0 else np.inf
        return X, cond
    if method == "ridge":
        # Tikhonov without SVD 
        # (G^T G + lam^2 I) X = G^T d  ->  X = (G^T G + lam^2 I)^-1 G^T d
        A = G.T @ G + lam**2 * np.eye(G.shape[1])
        B = G.T @ d
        X = np.linalg.solve(A, B)
        cond = np.linalg.cond(A)
        return X, cond
    
# --------------------------------------------------------------------------- #
# 4. OUTPUT FOR 1 POINT : retrieve_wind(x0, y0, z0) -> (u, v, w) OR None
# --------------------------------------------------------------------------- #

# Half-widths of the volume for selecting the data (km, deg, deg)
D_R     = 10.0  # km
D_THETA = 10.0   # deg
D_PHI   = 5.0   # deg #prev 15

min_n = 10  # minimum number of valid data points in the volume to attempt restitution

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
    u0, v0, w0 = X[0], X[1], X[2]
    ux, vy, wz = X[3], X[4], X[5]   # du/dx, dv/dy, dw/dz (classic fit)

    # ---- Filter out unrealistic wind estimates ----
    # Speed excessive
    # only horizontal components : w0 is poorly constrained (low elevations)
    # and would reject most points, while the plots only use u and v
    if MAX_WIND is not None and not np.all(np.abs([u0, v0]) <= MAX_WIND):
        return None
    # Badly conditioned system
    if MAX_COND is not None and cond > MAX_COND:
        return None

    return u0, v0, w0, wz, ux, vy


# --------------------------------------------------------------------------- #
# 4bis. w BY MASS CONTINUITY : vertical integration of the anelastic equation
#       d(rho w)/dz = -rho (du/dx+dv/dy), rho = rho0 * exp(-z/H_RHO)
# --------------------------------------------------------------------------- #

def integrate_w_continuity(z_vals, ux_vals, vy_vals):
    """Given the horizontal divergence (ux+vy) sampled along z_vals (NaN where
    no retrieval), integrate the anelastic continuity equation
    d(rho w)/dz = -rho (ux+vy), with rho = rho0 * exp(-z / H_RHO), using the
    trapezoidal rule and imposing w = 0 at the lowest valid level of the
    column (rho0 cancels out). Returns the w array (NaN wherever it cannot be
    bridged by >=2 valid consecutive samples)."""
    z_vals = np.asarray(z_vals, dtype=float)
    div_h = np.asarray(ux_vals, dtype=float) + np.asarray(vy_vals, dtype=float)
    w = np.full(z_vals.shape, np.nan)

    idx = np.nonzero(~np.isnan(div_h))[0]
    if idx.size < 2:
        return w

    z_v, d_v = z_vals[idx], div_h[idx]
    rho_v = np.exp(-z_v / H_RHO)
    # rho w (z) = - cumulative integral of rho * div_h, then w = (rho w) / rho
    rho_d = rho_v * d_v
    rho_w = np.concatenate(([0.0], -np.cumsum(0.5 * (rho_d[1:] + rho_d[:-1]) * np.diff(z_v))))
    w[idx] = rho_w / rho_v
    return w


def retrieve_wind_column(x0, y0, z_vals, w_method="classic"):
    """Retrieve (u, v, w, w'z) along a vertical column at fixed (x0, y0), over
    z_vals. w is computed according to w_method :
      - "classic"         : w = w0, independently at each level.
      - "mass_continuity" : w rebuilt from the vertical integration of the
                            anelastic continuity equation on the column's
                            horizontal divergence (see integrate_w_continuity),
                            and w'z = -(ux+vy) + w/H_RHO is the local
                            divergence-based derivative.
    Returns 4 arrays (U, V, W, WZ), same length as z_vals, NaN where missing."""
    z_vals = np.asarray(z_vals, dtype=float)
    n = z_vals.size
    U = np.full(n, np.nan); V = np.full(n, np.nan)
    W0 = np.full(n, np.nan); WZ0 = np.full(n, np.nan)
    UX = np.full(n, np.nan); VY = np.full(n, np.nan)

    for i, z0 in enumerate(z_vals):
        res = retrieve_wind(float(x0), float(y0), float(z0))
        if res is None:
            continue
        U[i], V[i], W0[i], WZ0[i], UX[i], VY[i] = res

    if w_method == "mass_continuity":
        W = integrate_w_continuity(z_vals, UX, VY)
        # anelastic : dw/dz = -(ux+vy) + w / H_RHO
        WZ = -(UX + VY) + W / H_RHO
    else:
        W, WZ = W0, WZ0

    return U, V, W, WZ

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

# xs, ys, zs, winds = sweep_workspace(X_GRID, Y_GRID, Z_GRID)
# print(f"Restitute points : {len(xs)} / {len(X_GRID) * len(Y_GRID) * len(Z_GRID)}")

# if len(winds) > 0:
#     U_rec, V_rec, W_rec = winds[:, 0], winds[:, 1], winds[:, 2]
# else:
#     print("No valid restitutions found in the workspace. Please adjust the grid parameters or volume dimensions.")

# %%
# --------------------------------------------------------------------------- #
# 6. VISUALIZATION : retrieved wind on 4 layers (2x2) + vertical profiles
# --------------------------------------------------------------------------- #

length_scale = 45  # km
A = 2.5
X_GRID  = np.arange(-length_scale, length_scale, A)
Y_GRID  = np.arange(-length_scale, length_scale, A)
DXY, DZ = 2.0, 1.0                      # horizontal / vertical spacing (km)
#X_GRID = np.arange(6.0, 34.1, DXY)
#Y_GRID = np.arange(6.0, 56.1, DXY)  
Z_LAYER = 2.5             # altitude of the displayed layer (km)
QUIVER_SCALE = 40         # scale for unit-length quiver arrows
# REMOVE_MEAN = True  -> plot the wind ANOMALY V' = V - mean : the ~uniform mean
#                       flow dominates and is removed, so the vortex / divergence
#                       structure becomes visible (this is NOT the real wind).
# REMOVE_MEAN = False -> plot the REAL total wind V (mean kept) : physically exact,
#                       but the vortex is largely hidden under the mean flow.
REMOVE_MEAN = False

# --- data collection : 2D grids (NaN = no retrieval) ---
U = np.full((len(Y_GRID), len(X_GRID)), np.nan)
V = np.full((len(Y_GRID), len(X_GRID)), np.nan)
for iy, y0 in enumerate(Y_GRID):
    for ix, x0 in enumerate(X_GRID):
        res = retrieve_wind(float(x0), float(y0), Z_LAYER)
        if res is not None:
            U[iy, ix] = res[0]
            V[iy, ix] = res[1]

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 7))

XX, YY = np.meshgrid(X_GRID, Y_GRID)

# remove the domain-mean flow to reveal the vortex (anomaly V'), or keep it (real V)
if REMOVE_MEAN:
    U_MEAN, V_MEAN = np.nanmean(U), np.nanmean(V)
else:
    U_MEAN, V_MEAN = 0.0, 0.0
_PRIME = "'" if REMOVE_MEAN else ""   # label : |V'| (anomaly) vs |V| (total)
Ur, Vr = U - U_MEAN, V - V_MEAN

speed = np.hypot(Ur, Vr)
n_pts = int(np.count_nonzero(~np.isnan(speed)))

# Color scale : use the SAME upper bound as the true-wind layer plot so identical
# speeds map to identical colors. The retrieved field only exists inside the radar
# annulus, so its own max would normalize differently. When the truth is available
# (synthetic case, real V), take the max of the true |V| over the full grid; else
# fall back to the retrieved max.
if (true_wind is not None) and (not REMOVE_MEAN):
    Ut, Vt, _ = true_wind(XX, YY, Z_LAYER)
    SPEED_MAX = np.nanmax(np.hypot(Ut, Vt))
else:
    SPEED_MAX = np.nanmax(speed) if np.any(~np.isnan(speed)) else 1.0

pm = ax.pcolormesh(XX, YY, speed, cmap="YlOrRd", alpha=0.7,
                   shading="nearest", vmin=0.0, vmax=SPEED_MAX)
fig.colorbar(pm, ax=ax, label=f"|V{_PRIME}| (m/s)")

# unit-length arrows : direction only, magnitude is in the background color
norm = np.where(speed > 0, speed, np.nan)
ax.quiver(XX, YY, Ur / norm, Vr / norm, scale=QUIVER_SCALE,
          width=0.003, color="k")
ax.scatter([0], [0], color="k", marker="^", s=60)

# Radar visibility at altitude z : annulus bounded by max range / lowest
R_MAX = RANGES_KM[-1]
ELV_MIN_RAD, ELV_MAX_RAD = ELEV_RAD.min(), ELEV_RAD.max()
circle_th = np.linspace(0.0, 2.0 * np.pi, 200)

s_outer = np.sqrt(max(R_MAX**2 - Z_LAYER**2, 0.0))
if ELV_MIN_RAD > 0:   # lowest-beam cap only meaningful for positive elevation
    s_outer = min(s_outer, Z_LAYER / np.tan(ELV_MIN_RAD))
s_inner = Z_LAYER / np.tan(ELV_MAX_RAD)
ax.plot(s_outer * np.cos(circle_th), s_outer * np.sin(circle_th),
        "r-", lw=1.3, label="radar visibility limit")
ax.legend(loc="upper right", fontsize=8)
ax.set_title(f"VVP — z = {Z_LAYER} km ({n_pts} pts)")
ax.set_xlabel("X East (km)"); ax.set_ylabel("Y North (km)")
ax.set_xlim(X_GRID[0] - 10, X_GRID[-1] + 10)
ax.set_ylim(Y_GRID[0] - 10, Y_GRID[-1] + 10)
ax.set_aspect("equal")

plt.tight_layout()

# %%
# --------------------------------------------------------------------------- #
# 6b. VERTICAL CROSS-SECTION : retrieved wind in a vertical plane, |V| in color
# --------------------------------------------------------------------------- #
# Slice the 3D field with a vertical plane and color the background by wind
# speed. SLICE_AXIS="y" -> plane at fixed Y (horizontal axis = X, vertical = Z);
# SLICE_AXIS="x" -> plane at fixed X (horizontal axis = Y, vertical = Z).
# In-plane arrows show the (horizontal, w) wind; w is poorly constrained so the
# vertical component of the arrows is only indicative.
PROFILE_POINT = (-20, 25)          # (x, y) km of the retrieved column

SHOW_VSLICE = True

if SHOW_VSLICE:
    SLICE_AXIS  = "y"          # "y" : plane Y=SLICE_POS ; "x" : plane X=SLICE_POS
    SLICE_POS   = PROFILE_POINT[1]         # km, position of the slicing plane
    H_GRID      = np.arange(-length_scale, length_scale, A)   # in-plane horizontal axis (km)
    Z_GRID_V    = np.arange(1.0, 9.0, 0.5)                      # vertical axis (km)
    # arrows are unit-length (direction only) since the color already shows the field ;
    # larger scale -> SHORTER arrows (matplotlib quiver convention)
    VSLICE_QUIVER_SCALE = 40
    # w is poorly constrained in single-Doppler VVP and blows up to non-physical
    # values : reject any point with |w| > VSLICE_W_MAX (m/s). Set None to disable.
    VSLICE_W_MAX = 10

    HH, ZZ = np.meshgrid(H_GRID, Z_GRID_V)
    _hlbl = "X East (km)" if SLICE_AXIS == "y" else "Y North (km)"

    # one panel per w-method (classic | mass_continuity), same axes / color scale
    figv, axesv = plt.subplots(1, len(W_METHODS), figsize=(9 * len(W_METHODS), 5),
                               sharex=True, sharey=True)
    for axv, w_method in zip(np.atleast_1d(axesv), W_METHODS):
        Uh = np.full((len(Z_GRID_V), len(H_GRID)), np.nan)   # in-plane horizontal comp.
        Wv = np.full((len(Z_GRID_V), len(H_GRID)), np.nan)   # vertical comp. (color)
        for ih, h0 in enumerate(H_GRID):
            if SLICE_AXIS == "y":
                x0, y0 = h0, SLICE_POS
            else:
                x0, y0 = SLICE_POS, h0
            Ucol, Vcol, Wcol, _ = retrieve_wind_column(x0, y0, Z_GRID_V, w_method)
            # horizontal wind projected onto the in-plane horizontal direction
            Uh[:, ih] = Ucol if SLICE_AXIS == "y" else Vcol
            Wv[:, ih] = Wcol

        # reject non-physical vertical velocities : mask points with |w| > VSLICE_W_MAX
        if VSLICE_W_MAX is not None:
            bad = np.abs(Wv) > VSLICE_W_MAX
            n_rej = int(np.count_nonzero(bad & ~np.isnan(Wv)))
            n_tot = int(np.count_nonzero(~np.isnan(Wv)))
            Wv = np.where(bad, np.nan, Wv)
            Uh = np.where(bad, np.nan, Uh)
            print(f"V-slice ({w_method}) : {n_rej}/{n_tot} points rejected "
                  f"(|w| > {VSLICE_W_MAX} m/s)")

        # color = vertical velocity w (signed) -> diverging colormap centered on 0.
        # scale capped at the rejection threshold so the kept values fill the colormap.
        W_MAX = VSLICE_W_MAX if VSLICE_W_MAX is not None else (
            np.nanmax(np.abs(Wv)) if np.any(~np.isnan(Wv)) else 1.0)
        pmv = axv.pcolormesh(HH, ZZ, Wv, cmap="RdBu_r", alpha=0.9,
                             shading="nearest", vmin=-W_MAX, vmax=W_MAX)
        figv.colorbar(pmv, ax=axv, label="w (m/s)")

        # unit-length in-plane arrows : direction only, magnitude is in the background
        # color. Default angles="uv" -> equal on-screen length regardless of axis scaling.
        mag = np.hypot(Uh, Wv)
        mag = np.where(mag > 0, mag, np.nan)
        axv.quiver(HH, ZZ, Uh / mag, Wv / mag, color="k", width=0.003,
                   scale=VSLICE_QUIVER_SCALE, pivot="mid")
        axv.scatter([0], [0], color="k", marker="^", s=60)   # radar at origin (z=0)

        axv.set_title(f"VVP vertical cross-section — {SLICE_AXIS.upper()} = "
                      f"{SLICE_POS:.0f} km  (w : {w_method})")
        axv.set_xlabel(_hlbl); axv.set_ylabel("altitude Z (km)")
        axv.set_xlim(H_GRID[0] - 5, H_GRID[-1] + 5)
        axv.set_ylim(0.0, Z_GRID_V[-1] + 0.5)
    plt.tight_layout()

# --- Vertical profile at a single point : one panel per w-method ---

Z_PROFILE_SHOW = True  # set to False to hide the vertical profile

# profile column shared with the RHI view (section 6c)
z_profile = np.arange(0.5, 9.0, 0.5)
px, py = PROFILE_POINT

if Z_PROFILE_SHOW:
    # one panel per w-method (classic | mass_continuity), same as the v-slice
    fig2, axes2 = plt.subplots(1, len(W_METHODS), figsize=(6 * len(W_METHODS), 5),
                               sharey=True)
    for ax, w_method in zip(np.atleast_1d(axes2), W_METHODS):
        _, _, w_rec, wz_rec = retrieve_wind_column(px, py, z_profile, w_method)

        ax.plot(w_rec, z_profile, "b--s", label="retrieved w")

        # arrow on each w point : direction of the vertical derivative w'z = dw/dz
        # drawn as the local tangent (dw/dz, 1) over a small altitude step, so the
        # arrow lies along the profile and points toward increasing altitude
        # (tilts right if w increases with height, left if it decreases).
        DZ_ARROW = 0.3                                # altitude step of the arrow (km)
        ax.quiver(w_rec, z_profile, wz_rec * DZ_ARROW, np.full_like(wz_rec, DZ_ARROW),
                  angles="xy", scale_units="xy", scale=1, color="r",
                  width=0.005, label="direction of w'z")

        ax.set_title(f"w(z) at ({px:.0f}, {py:.0f}) km  (w : {w_method})")
        ax.set_xlabel("w (m/s)"); ax.set_ylabel("altitude (km)")
        ax.grid(True)
        ax.legend(loc="best")

    plt.tight_layout()

#plt.savefig(f"wind_restitution_vvp_{Z_LAYER}km_12P.png", dpi=300)

# %%
# --------------------------------------------------------------------------- #
# 6c. REFLECTIVITY RHI THROUGH THE PROFILE POINT
# --------------------------------------------------------------------------- #
# If SHOW_PROFILE_RHI is True, read the reflectivity twin (ZH) of the velocity
# volume and draw the RHI (vertical cross-section, same native-cell pcolormesh
# as reflectivity_RHI.py) at the azimuth of PROFILE_POINT, with an indicator
# showing where the retrieved column sits in the cross-section (ground
# distance s0, altitudes z_profile).

SHOW_PROFILE_RHI = True

if SHOW_PROFILE_RHI and USE_REAL_DATA:
    REFLECTIVITY_FILE = (
        "Data_Radar/20240711_Z_V_W_HV_MTI/20240711_01_ZH_MTI/"
        "20240711_010000.00-00-PPI.RAW-ZH_MTI.NSK-AUTO-LEM2.suita.dat.gz"
    )
    DBZ_MIN, DBZ_MAX = 0.0, 60.0   # color scale (dBZ)
    FILL_THRESHOLD = -100.0        # dBZ : below this is the "no data" fill
    RHI_Z_MAX_KM = 15.0            # altitude limit of the RHI (km)

    # --- read the ZH volume (same handling as reflectivity_RHI.py) ---
    _zfile = mppawr(REFLECTIVITY_FILE, switch=False)
    _zraw = _zfile.read_data()               # (azimuth, elevation, range)
    _nsect, _nelv, _nrng = _zraw.shape
    ELEV_Z_DEG = np.asarray(_zfile.elv_center, dtype=float)
    AZIM_Z_DEG = (np.arange(_nsect) + 0.5) * (360.0 / _nsect)
    RNG_Z_KM = (np.arange(_nrng) + 0.5) * _zfile.range_res / 1000.0
    _zfile.close()

    z_dbz = _zraw.astype(float)
    z_dbz[z_dbz < FILL_THRESHOLD] = np.nan
    z_dbz = np.transpose(z_dbz, (1, 0, 2))   # (elevation, azimuth, range)
    _keep = RNG_Z_KM <= RANGE_MAX_KM
    RNG_Z_KM = RNG_Z_KM[_keep]
    z_dbz = z_dbz[:, :, _keep]

    # --- azimuth / ground distance of PROFILE_POINT (same convention as the
    #     VVP fit : theta = arctan2(x, y), azimuth clockwise from North) ---
    s0 = float(np.hypot(px, py))                          # ground distance (km)
    az0 = float(np.degrees(np.arctan2(px, py)) % 360.0)   # azimuth (deg)
    ia_z = int(np.argmin(np.abs(AZIM_Z_DEG - az0)))
    print(f"Profile RHI : point ({px:.1f}, {py:.1f}) km -> azimuth {az0:.1f} deg "
          f"(nearest beam {AZIM_Z_DEG[ia_z]:.1f} deg), s0 = {s0:.1f} km")

    # --- (range, elevation) -> (ground distance, altitude), 4/3 earth radius ---
    KE_KM = 4.0 / 3.0 * 6371.0
    _el2d, _r2d = np.meshgrid(np.deg2rad(ELEV_Z_DEG), RNG_Z_KM, indexing="ij")
    _h2d = np.sqrt(_r2d**2 + KE_KM**2 + 2.0 * _r2d * KE_KM * np.sin(_el2d)) - KE_KM
    _s2d = KE_KM * np.arcsin(_r2d * np.cos(_el2d) / (KE_KM + _h2d))

    # --- RHI (native cells, no interpolation) + zoom on the profile column ---
    RHI_ZOOM_HALF_KM = 5.0   # half-width of the zoom panel around the column (km)

    _zh_masked = np.ma.masked_invalid(z_dbz[:, ia_z, :])
    fig5, (ax5, ax5z) = plt.subplots(
        1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2.2, 1]},
        constrained_layout=True)

    # left panel : full RHI
    pm5 = ax5.pcolormesh(_s2d, _h2d, _zh_masked, cmap="NWSRef",
                         vmin=DBZ_MIN, vmax=DBZ_MAX, shading="nearest")

    # indicator : arrow pointing at the top of the retrieved VVP column, so the
    # echoes along the column stay fully visible (no line drawn over them)
    ax5.annotate(
        f"VVP profile at ({px:.1f}, {py:.1f}) km\n"
        f"z = {z_profile[0]:.1f} to {z_profile[-1]:.1f} km",
        xy=(s0, z_profile[-1]),                     # arrow tip : top of the column
        xytext=(s0 + 12.0, RHI_Z_MAX_KM - 2.5),     # label above, to the right
        arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5),
        fontsize=9, ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="k", alpha=0.85),
    )

    ax5.set_xlim(0.0, RANGE_MAX_KM)
    ax5.set_ylim(0.0, RHI_Z_MAX_KM)
    ax5.set_xlabel("ground distance (km)")
    ax5.set_ylabel("altitude (km)")
    ax5.grid(True, ls="--", lw=0.4, alpha=0.5)
    ax5.set_title(f"RHI reflectivity — azimuth {AZIM_Z_DEG[ia_z]:.1f} deg "
                  f"through PROFILE_POINT ({px:.1f}, {py:.1f}) km")

    # right panel : zoom around the column (same data, tight x window) ; here a
    # thin dashed line + level markers are fine, the zoom keeps the echoes readable
    pm5z = ax5z.pcolormesh(_s2d, _h2d, _zh_masked, cmap="NWSRef",
                           vmin=DBZ_MIN, vmax=DBZ_MAX, shading="nearest")
    ax5z.axvline(s0, color="k", ls="--", lw=1.0, alpha=0.7)
    ax5z.plot(np.full(z_profile.size, s0), z_profile, "wo", mec="k", ms=5)

    ax5z.set_xlim(s0 - RHI_ZOOM_HALF_KM, s0 + RHI_ZOOM_HALF_KM)
    ax5z.set_ylim(0.0, z_profile[-1] + 1.5)
    ax5z.set_xlabel("ground distance (km)")
    ax5z.set_ylabel("altitude (km)")
    ax5z.grid(True, ls="--", lw=0.4, alpha=0.5)
    ax5z.set_title(f"zoom on the VVP column (s = {s0:.1f} km)")

    fig5.colorbar(pm5z, ax=[ax5, ax5z], pad=0.02, label="ZH (dBZ)")

# %%
# --------------------------------------------------------------------------- #
# 6d. 3D ISOSURFACES OF THE REFLECTIVITY : "storm objects"
# --------------------------------------------------------------------------- #
# The polar ZH volume is resampled (nearest neighbour) on a regular Cartesian
# grid, lightly smoothed, then closed 3D surfaces are extracted with marching
# cubes at increasing dBZ levels : nested translucent shells show WHERE the
# signal is strong, like storm cells, instead of drawing radar planes.
# Requires scikit-image (marching cubes) and scipy (smoothing).
# Reuses the ZH volume loaded in section 6c.

SHOW_3D_ISO = True

if SHOW_3D_ISO and SHOW_PROFILE_RHI and USE_REAL_DATA:
    from scipy.ndimage import gaussian_filter
    from skimage import measure

    # NB : the gaussian smoothing lowers the peaks (raw 35-40 dBZ cores end up
    # around 30-33 dBZ on the smoothed grid), so keep the top level modest.
    ISO_LEVELS = (10.0, 20.0, 30.0)   # dBZ levels of the nested shells
    ISO_ALPHAS = (0.12, 0.35, 1.0)    # transparency : outer shells see-through
    GRID_DXY   = 1.5    # Cartesian grid step, horizontal (km)
    GRID_DZ    = 0.5    # Cartesian grid step, vertical (km)
    Z3D_MAX_KM = 15.0   # altitude limit of the grid / 3D box (km)
    SMOOTH_SIG = 1.0    # gaussian smoothing (in grid cells) : cleaner objects
    VOL_FLOOR  = -5.0   # dBZ assigned outside coverage / no echo (below levels)

    # iso-w shells (mass-continuity w) overlaid on the ZH shells : updrafts
    # should sit in/under the reflectivity cores, which is the physical
    # cross-check of the retrieval. The w grid is much coarser than ZH (its
    # resolution is the VVP fitting volume) : compare positions, not shapes.
    W_ISO_MS   = 1.5    # m/s : draw w = +W_ISO (updraft) and -W_ISO (downdraft)
    W_GRID_DXY = 6.0    # horizontal step of the w column grid (km)
    W_RANGE_KM = 60.0   # horizontal extent of the w grid (km)
    W_GRID_DZ  = 0.5    # vertical step of the w columns (km)
    W_Z_MAX_KM = 9.0    # top of the w columns (km)
    W_MAX_OK   = 10.0   # reject |w| above this (same filter as the v-slice)

    # ---- 1. polar (elevation, azimuth, range) -> Cartesian (x, y, z) ----
    xg = np.arange(-RANGE_MAX_KM, RANGE_MAX_KM + 1e-9, GRID_DXY)
    yg = np.arange(-RANGE_MAX_KM, RANGE_MAX_KM + 1e-9, GRID_DXY)
    zg = np.arange(GRID_DZ, Z3D_MAX_KM + 1e-9, GRID_DZ)
    Xg, Yg, Zg = np.meshgrid(xg, yg, zg, indexing="ij")

    sg = np.hypot(Xg, Yg)                                # ground distance (km)
    rg = np.hypot(sg, Zg)                                # slant range (km)
    azg = np.degrees(np.arctan2(Xg, Yg)) % 360.0         # azimuth (deg)
    elg = np.degrees(np.arctan2(Zg, sg))                 # elevation (deg)

    # nearest radar cell for every grid point
    iaz = (azg / (360.0 / AZIM_Z_DEG.size)).astype(int) % AZIM_Z_DEG.size
    irg = (rg / (RNG_Z_KM[1] - RNG_Z_KM[0])).astype(int)
    _pos = np.searchsorted(ELEV_Z_DEG, elg).clip(1, ELEV_Z_DEG.size - 1)
    iel = np.where(np.abs(elg - ELEV_Z_DEG[_pos - 1])
                   <= np.abs(ELEV_Z_DEG[_pos] - elg), _pos - 1, _pos)

    # outside the scanned cone (range, elevation) -> VOL_FLOOR
    inside = (irg < RNG_Z_KM.size) & (elg >= ELEV_Z_DEG[0] - 1.0) \
             & (elg <= ELEV_Z_DEG[-1] + 1.0)
    vol = np.full(Xg.shape, VOL_FLOOR)
    _samp = z_dbz[iel[inside], iaz[inside], irg[inside]]
    vol[inside] = np.where(np.isfinite(_samp), _samp, VOL_FLOOR)

    vol = gaussian_filter(vol, sigma=SMOOTH_SIG)
    print(f"3D isosurfaces : grid {vol.shape}, "
          f"ZH max after smoothing = {vol.max():.1f} dBZ")

    # ---- 2. one closed surface per dBZ level (skip empty levels) ----
    cmap_ref = plt.colormaps["NWSRef"]
    norm_ref = plt.Normalize(vmin=DBZ_MIN, vmax=DBZ_MAX)
    iso_specs = [(lev, alpha, cmap_ref(norm_ref(lev))[:3])
                 for lev, alpha in zip(ISO_LEVELS, ISO_ALPHAS)
                 if lev < vol.max()]

    # ---- 2bis. w volume by mass continuity on a coarse column grid ----
    xw = np.arange(-W_RANGE_KM, W_RANGE_KM + 1e-9, W_GRID_DXY)
    yw = np.arange(-W_RANGE_KM, W_RANGE_KM + 1e-9, W_GRID_DXY)
    zw = np.arange(W_GRID_DZ, W_Z_MAX_KM + 1e-9, W_GRID_DZ)
    wvol = np.full((xw.size, yw.size, zw.size), np.nan)
    _verbose_save, VERBOSE = VERBOSE, False   # silence the per-point prints
    for ixw, x0 in enumerate(xw):
        for iyw, y0 in enumerate(yw):
            _, _, Wcol, _ = retrieve_wind_column(x0, y0, zw, "mass_continuity")
            wvol[ixw, iyw, :] = Wcol
    VERBOSE = _verbose_save
    wvol = np.where(np.abs(wvol) <= W_MAX_OK, wvol, np.nan)
    n_wok = int(np.count_nonzero(~np.isnan(wvol)))
    print(f"iso-w : {n_wok}/{wvol.size} points retrieved, "
          f"w in [{np.nanmin(wvol):.1f}, {np.nanmax(wvol):.1f}] m/s")
    # no retrieval -> 0 : stays between -W_ISO and +W_ISO, so no surface there
    wvol = gaussian_filter(np.nan_to_num(wvol, nan=0.0), sigma=0.8)
    w_iso_specs = ((W_ISO_MS, "crimson", f"w >= +{W_ISO_MS:.1f} m/s"),
                   (-W_ISO_MS, "royalblue", f"w <= -{W_ISO_MS:.1f} m/s"))

    # ---- 3. render : PyVista (GPU) if available, matplotlib otherwise ----
    # matplotlib 3D re-sorts and redraws every triangle on the CPU at each
    # mouse move, so rotating tens of thousands of triangles lags whatever
    # the machine. The PyVista window (VTK / OpenGL, GPU-accelerated) stays
    # fluid ; it opens FIRST and blocks the script until it is closed, then
    # the matplotlib figures appear.
    try:
        import pyvista as pv
        USE_PYVISTA = True
    except ImportError:
        USE_PYVISTA = False
        print("pyvista not installed -> matplotlib 3D fallback "
              "(pip install pyvista for a fluid interactive view)")

    if USE_PYVISTA:
        grid3 = pv.ImageData(dimensions=vol.shape,
                             spacing=(GRID_DXY, GRID_DXY, GRID_DZ),
                             origin=(xg[0], yg[0], zg[0]))
        grid3.point_data["ZH"] = vol.ravel(order="F")   # x varies fastest

        pl3 = pv.Plotter(window_size=(1100, 850))
        for lev, alpha, col in iso_specs:
            iso = grid3.contour([lev], scalars="ZH")
            if iso.n_points == 0:
                continue
            pl3.add_mesh(iso, color=col, opacity=alpha, smooth_shading=True,
                         label=f"ZH >= {lev:.0f} dBZ")
            print(f"  level {lev:.0f} dBZ : {iso.n_cells} triangles")

        # iso-w shells on their own (coarser) grid, same scene
        gridw = pv.ImageData(dimensions=wvol.shape,
                             spacing=(W_GRID_DXY, W_GRID_DXY, W_GRID_DZ),
                             origin=(xw[0], yw[0], zw[0]))
        gridw.point_data["w"] = wvol.ravel(order="F")
        for lev, col, lbl in w_iso_specs:
            isow = gridw.contour([lev], scalars="w")
            if isow.n_points == 0:
                print(f"  {lbl} : empty, skipped")
                continue
            pl3.add_mesh(isow, color=col, opacity=0.55, smooth_shading=True,
                         label=lbl)
            print(f"  {lbl} : {isow.n_cells} triangles")

        # radar marker : small cone pointing up at the origin
        pl3.add_mesh(pv.Cone(center=(0.0, 0.0, 0.5), direction=(0, 0, 1),
                             height=1.0, radius=1.0), color="black")
        pl3.add_legend(bcolor="white", size=(0.2, 0.12))
        pl3.set_scale(zscale=7.0)   # thin layers : stretch z (like box_aspect)
        pl3.show_grid(xtitle="X East (km)", ytitle="Y North (km)",
                      ztitle="altitude Z (km)")
        pl3.show(title="Reflectivity isosurfaces — 3D objects (PyVista)")
    else:
        fig3d = plt.figure(figsize=(10, 8))
        ax3d = fig3d.add_subplot(111, projection="3d")

        from matplotlib.patches import Patch
        handles = []
        for lev, alpha, col in iso_specs:
            verts, faces, _, _ = measure.marching_cubes(
                vol, level=lev, spacing=(GRID_DXY, GRID_DXY, GRID_DZ))
            ax3d.plot_trisurf(verts[:, 0] + xg[0], verts[:, 1] + yg[0],
                              verts[:, 2] + zg[0], triangles=faces,
                              color=col, alpha=alpha, linewidth=0, shade=True)
            handles.append(Patch(facecolor=col, alpha=max(alpha, 0.4),
                                 label=f"ZH >= {lev:.0f} dBZ"))
            print(f"  level {lev:.0f} dBZ : {faces.shape[0]} triangles")

        # iso-w shells on their own (coarser) grid, same axes
        for lev, col, lbl in w_iso_specs:
            if not (wvol.min() < lev < wvol.max()):
                print(f"  {lbl} : empty, skipped")
                continue
            verts, faces, _, _ = measure.marching_cubes(
                wvol, level=lev, spacing=(W_GRID_DXY, W_GRID_DXY, W_GRID_DZ))
            ax3d.plot_trisurf(verts[:, 0] + xw[0], verts[:, 1] + yw[0],
                              verts[:, 2] + zw[0], triangles=faces,
                              color=col, alpha=0.55, linewidth=0, shade=True)
            handles.append(Patch(facecolor=col, label=lbl))
            print(f"  {lbl} : {faces.shape[0]} triangles")

        ax3d.scatter([0], [0], [0], color="k", marker="^", s=80)   # radar
        ax3d.legend(handles=handles, loc="upper right", fontsize=9)

        ax3d.set_xlabel("X East (km)"); ax3d.set_ylabel("Y North (km)")
        ax3d.set_zlabel("altitude Z (km)")
        ax3d.set_xlim(-RANGE_MAX_KM, RANGE_MAX_KM)
        ax3d.set_ylim(-RANGE_MAX_KM, RANGE_MAX_KM)
        ax3d.set_zlim(0.0, Z3D_MAX_KM)
        ax3d.set_box_aspect((1.0, 1.0, 0.65))   # thin layers : stretch z
        ax3d.view_init(elev=25, azim=-60)
        ax3d.set_title("Reflectivity isosurfaces — 3D objects where the echo "
                       "is strong")

# %%
# --------------------------------------------------------------------------- #
# 7. PPI VIEW OF THE RAW RADIAL VELOCITY (Py-ART)
# --------------------------------------------------------------------------- #
# Requires the project .venv (Python 3.12 + arm_pyart) :
# in VSCode, select the ".venv" interpreter/kernel for this file.

SHOW_PPI = False  # set to True to display the PPI plots of the raw radial velocity data using Py-ART

if SHOW_PPI:
    PPI_ELEVATION_DEG = 5.47   # desired elevation (deg) : nearest available sweep is shown
    VEL_LIM = None           # color scale limit (m/s) ; None -> auto-fit to the sweep data

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

    # nearest available sweep to the requested elevation
    sw = int(np.argmin(np.abs(ELEVATIONS_DEG - PPI_ELEVATION_DEG)))
    print(f"PPI : elevation {PPI_ELEVATION_DEG:.1f}° -> sweep {sw} ({ELEVATIONS_DEG[sw]:.2f}°)")

    # Color scale fitted to the displayed sweep : a fixed, too-wide limit makes the
    # diverging colormap wash out to its pale centre (the "too white" effect). When
    # VEL_LIM is None, fit it to the 99th percentile of |vr| on that sweep.
    if VEL_LIM is None:
        finite = np.abs(vr[sw][np.isfinite(vr[sw])])
        VEL_LIM = max(float(np.ceil(np.percentile(finite, 99))) if finite.size else 30.0, 1.0)
    print(f"PPI : color scale +/- {VEL_LIM:.0f} m/s")

    display = pyart.graph.RadarDisplay(radar_ppi)
    fig3, ax3 = plt.subplots(figsize=(8, 7))
    display.plot_ppi("velocity", sweep=sw, ax=ax3, fig=fig3,
                     vmin=-VEL_LIM, vmax=VEL_LIM, cmap="RdBu_r",
                     colorbar_label="vr (m/s)",
                     title=f"PPI radial velocity — elevation {ELEVATIONS_DEG[sw]:.2f}°")
    display.plot_range_rings([20, 40, 60, 80], ax=ax3, lw=0.5, ls="--")

    # Circle = where this sweep crosses the Z_LAYER altitude (2.5 km).
    # At a fixed elevation phi, a constant altitude z maps to a constant slant
    # range r0 = z/sin(phi), plotted on the PPI as a ground-range ring s = z/tan(phi).
    phi = np.deg2rad(ELEVATIONS_DEG[sw])
    r0 = Z_LAYER / np.sin(phi)            # slant range (km) reaching z = 2.5 km
    s_ring = Z_LAYER / np.tan(phi)        # ground range (km) shown on the PPI
    th_c = np.linspace(0.0, 2.0 * np.pi, 361)
    ax3.plot(s_ring * np.sin(th_c), s_ring * np.cos(th_c), "g-", lw=2.0,
             label=f"z = {Z_LAYER} km  (r = {r0:.1f} km)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.set_aspect("equal")

    plt.tight_layout()

    # ---- vr along that circle : intensity vs azimuth (VAD-like curve) ----
    # For a uniform horizontal wind, vr(azimuth) is a sinusoid of amplitude |V|cos(phi).
    ir = int(np.argmin(np.abs(RANGES_KM - r0)))
    vr_az = vr[sw, :, ir]
    fig4, ax4 = plt.subplots(figsize=(9, 4))
    ax4.plot(AZIMUTHS_DEG, vr_az, "b.-", ms=3, lw=0.8)
    ax4.axhline(0.0, color="k", lw=0.6)
    ax4.set_title(f"vr=f(azimuth) — z = {Z_LAYER} km "
                  f"(elev {ELEVATIONS_DEG[sw]:.2f}°, r = {RANGES_KM[ir]:.1f} km)")
    ax4.set_xlabel("azimuth (deg)"); ax4.set_ylabel("vr (m/s)")
    ax4.set_xlim(0.0, 360.0); ax4.grid(True)
    plt.tight_layout()

# single show at the very end : every enabled figure appears at the same time
plt.show()
