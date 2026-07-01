
# %% Imports
import numpy as np
import matplotlib.pyplot as plt

# ---- physical parameters (continuity / fall speed) ----
VT = 0.0          # hydrometeor fall speed (m/s) ; vr senses (w - Vt)
H_RHO = 9.5       # density scale height (km) : rho ~ exp(-z / H_RHO)
ANELASTIC = True # False -> incompressible dw/dz = -div (standard VAD) ;
                  # True  -> anelastic, density-weighted (w grows ~exp(z/H) aloft)

# %%
# --------------------------------------------------------------------------- #
# 0. DATA SOURCES : REAL OR SYNTHETIC   (same block as improved_VVP_method.py)
# --------------------------------------------------------------------------- #

USE_REAL_DATA = False

VERBOSE = False

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
    SYNTHETIC_FIELD = "generator"      # "generator" | "divergent"

    from generate_radar_data import ELEVATIONS_DEG, AZIMUTHS_DEG, RANGES_KM

    def _radial_velocity(field):
        """Project a (u, v, w) field onto every beam -> vr cube (elev, azim, range)."""
        el, az, r = np.meshgrid(np.deg2rad(ELEVATIONS_DEG), np.deg2rad(AZIMUTHS_DEG),
                                RANGES_KM, indexing="ij")
        cphi, sphi = np.cos(el), np.sin(el)
        sth, cth = np.sin(az), np.cos(az)
        x, y, z = r * sth * cphi, r * cth * cphi, r * sphi
        u, v, w = field(x, y, z)
        return u * sth * cphi + v * cth * cphi + (w - VT) * sphi

    if SYNTHETIC_FIELD == "generator":
        from generate_radar_data import generate, true_wind, dudx, dvdy
        vr = generate().reshape(len(ELEVATIONS_DEG), len(AZIMUTHS_DEG), len(RANGES_KM))
        DIV_TRUE = dudx + dvdy
    else:
        # Horizontally divergent field. w(z) is the EXACT continuity solution for a
        # constant divergence with w(0) = 0, matching whichever scheme the VAD
        # integration uses, so a perfect VAD returns exactly this profile :
        #     incompressible : w(z) = -DIV * z
        #     anelastic      : w(z) = -DIV * H_RHO * (exp(z / H_RHO) - 1)
        DIV_TRUE = 0.2          # (m/s)/km : du/dx + dv/dy (split equally on x and y)
        U0, V0 = 4.0, 6.0
        def true_wind(x, y, z):
            u = U0 + 0.5 * DIV_TRUE * x
            v = V0 + 0.5 * DIV_TRUE * y
            if ANELASTIC:
                w = -DIV_TRUE * H_RHO * (np.exp(z / H_RHO) - 1.0)
            else:
                w = -DIV_TRUE * z
            shp = np.broadcast_shapes(np.shape(u), np.shape(v), np.shape(w))
            return (np.broadcast_to(u, shp).copy(),
                    np.broadcast_to(v, shp).copy(),
                    np.broadcast_to(w, shp).copy())
        vr = _radial_velocity(true_wind)

    print(f"Synthetic ({SYNTHETIC_FIELD}) : vr shape = {vr.shape}, DIV_true = {DIV_TRUE:+.3f}")

# Data-driven threshold for rejecting diverging fits (aliasing, clutter...)
MAX_WIND = 1.5 * np.nanpercentile(np.abs(vr), 99.9)   # m/s
print(f"MAX_WIND = {MAX_WIND:.1f} m/s (1.5 x p99.9 des |vr|)")

# %%
# --------------------------------------------------------------------------- #
# 1. RADAR COORDINATES + PHYSICAL PARAMETERS
# --------------------------------------------------------------------------- #

ELEV_RAD = np.deg2rad(ELEVATIONS_DEG)   
AZIM_RAD = np.deg2rad(AZIMUTHS_DEG)     

# VAD fit / quality-control parameters
N_HARM   = 3            # number of azimuthal harmonics fitted (1 gives u,v ; >=3 is standard VAD)
MIN_AZ   = 12           # minimum valid azimuth gates on a ring to attempt the fit
MIN_COVER = 0.55        # minimum azimuthal coverage fraction (valid gates / total)
MAX_RMS  = 8.0          # m/s : reject a ring whose sinusoidal fit residual is too large

# Elevations usable for the VAD
ELEV_MIN_DEG = 1.0
ELEV_MAX_DEG = 30.0      
ELEV_MAX_DIV_DEG = 10.0

# %%
# --------------------------------------------------------------------------- #
# 2. VAD CORE : harmonic fit of vr(azimuth) on one range ring
# --------------------------------------------------------------------------- #

# Design matrix [1, sin t, cos t, sin 2t, cos 2t, ...] for a Fourier fit.
def harmonic_design(theta, n_harm=N_HARM):
    cols = [np.ones_like(theta)]
    for k in range(1, n_harm + 1):
        cols.append(np.sin(k * theta))
        cols.append(np.cos(k * theta))
    return np.column_stack(cols)

# VAD fit function
def vad_fit_ring(vr_ring):
    """Fit vr(theta) = a0 + a1 sin t + b1 cos t (+ harmonics) on one range ring.
    Returns (a0, a1, b1, n, rms) or None if the ring is unusable.
    Coefficient order from harmonic_design : [a0, a1, b1, a2, b2, ...].
    """
    valid = ~np.isnan(vr_ring)
    n = int(np.count_nonzero(valid))
    if n < MIN_AZ or n < MIN_COVER * vr_ring.size:
        return None

    theta = AZIM_RAD[valid]
    d = vr_ring[valid]

    G = harmonic_design(theta)
    GtG = G.T @ G          # (p, p), symétrique définie positive si rang plein
    Gtd = G.T @ d          # (p,)
    try:
        coef = np.linalg.solve(GtG, Gtd)
    except np.linalg.LinAlgError:
        return None         # anneau de rang déficient -> inexploitable
    resid = d - G @ coef
    rms = float(np.sqrt(np.mean(resid ** 2)))
    if rms > MAX_RMS:
        return None

    a0, a1, b1 = coef[0], coef[1], coef[2]
    return a0, a1, b1, n, rms

# VAD sweep function
def vad_ring_wind(sweep, ir):
    """VAD on the (sweep, range gate ir) ring.
    Returns dict with z, ground-radius s, u, v, a0, div, or None.
    """
    phi = ELEV_RAD[sweep]
    cphi, sphi = np.cos(phi), np.sin(phi)
    r = RANGES_KM[ir]
    if cphi < 0.1 or r <= 0:        # near-zenith or zero range : skip
        return None

    res = vad_fit_ring(vr[sweep, :, ir])
    if res is None:
        return None
    a0, a1, b1, n, rms = res

    # First harmonic -> horizontal wind at the radar axis
    u = a1 / cphi
    v = b1 / cphi
    if not np.all(np.abs([u, v]) <= MAX_WIND):   # reject aliased / diverging rings
        return None

    # Mean a0 -> horizontal divergence (neglecting the small sin(phi)*w term)
    #   a0 = (1/2) r cos^2(phi) * DIV + sin(phi) (w - Vt)
    div = 2.0 * a0 / (r * cphi ** 2)             # 1/s if r in m ; here r in km -> (m/s)/km

    return dict(z=r * sphi, s=r * cphi, u=u, v=v, a0=a0, div=div, n=n, rms=rms)


# %%
# --------------------------------------------------------------------------- #
# 3. VERTICAL PROFILE : collect rings over all usable sweeps, bin by height
# --------------------------------------------------------------------------- #

Z_MAX = 12.0    # km : top of the retrieved profile
DZ    = 0.5     # km : height-bin thickness

Z_EDGES = np.arange(0.0, Z_MAX + DZ, DZ)
Z_CENT  = 0.5 * (Z_EDGES[:-1] + Z_EDGES[1:])


def collect_rings():
    """Run the VAD on every usable ring and return per-ring (z, u, v, div) lists."""
    sweeps = np.nonzero((ELEVATIONS_DEG >= ELEV_MIN_DEG) &
                        (ELEVATIONS_DEG <= ELEV_MAX_DEG))[0]
    zr, ur, vrr, dvr = [], [], [], []
    for s in sweeps:
        # u, v are taken from every usable sweep ; the divergence (-> w) only from
        # the low elevations where the a0 w-term contamination stays negligible.
        div_ok = ELEVATIONS_DEG[s] <= ELEV_MAX_DIV_DEG
        for ir in range(len(RANGES_KM)):
            res = vad_ring_wind(s, ir)
            if res is None or res["z"] > Z_MAX:
                continue
            zr.append(res["z"]); ur.append(res["u"]); vrr.append(res["v"])
            dvr.append(res["div"] if div_ok else np.nan)
            if VERBOSE:
                print(f"sweep {s:3d} (elev {ELEVATIONS_DEG[s]:5.2f}) "
                      f"r={RANGES_KM[ir]:5.1f} z={res['z']:4.1f} "
                      f"u={res['u']:6.2f} v={res['v']:6.2f} div={res['div']:+.3f}")
    return (np.asarray(zr), np.asarray(ur), np.asarray(vrr), np.asarray(dvr))


def bin_profile(zr, ur, vrr, dvr):
    """Average the per-ring estimates into height bins -> u(z), v(z), div(z)."""
    idx = np.digitize(zr, Z_EDGES) - 1
    nb = len(Z_CENT)
    U = np.full(nb, np.nan); V = np.full(nb, np.nan)
    D = np.full(nb, np.nan); N = np.zeros(nb, dtype=int)
    for b in range(nb):
        m = idx == b
        N[b] = int(np.count_nonzero(m))
        if N[b] == 0:
            continue
        U[b] = np.mean(ur[m]); V[b] = np.mean(vrr[m])
        dm = dvr[m][~np.isnan(dvr[m])]      # divergence is NaN for high-elevation rings
        if dm.size:
            D[b] = np.mean(dm)
    return U, V, D, N


def integrate_w(z_cent, div, anelastic=ANELASTIC, H=H_RHO):
    """Integrate mass continuity from the ground (w=0) up to get w(z).

    Incompressible : dw/dz = -DIV.
    Anelastic      : d(rho w)/dz = -rho DIV, with rho = exp(-z / H).
    Internal NaN divergence bins are linearly interpolated so the integral is
    continuous, but w is NOT extrapolated above the highest data-constrained
    level : there the divergence is unknown, so w is left NaN.
    """
    good = ~np.isnan(div)
    if np.count_nonzero(good) < 2:
        return Z_CENT, np.full_like(Z_CENT, np.nan)

    zc = z_cent[good]
    z_top = zc.max()                             # highest altitude with a real divergence
    dv = np.interp(z_cent, zc, div[good])        # fill internal gaps within the covered range

    # prepend the ground point (z=0, w=0)
    z = np.insert(z_cent, 0, 0.0)
    dv = np.insert(dv, 0, dv[0])
    rho = np.exp(-z / H) if anelastic else np.ones_like(z)

    # cumulative trapezoidal integral of rho * DIV
    incr = 0.5 * (rho[1:] * dv[1:] + rho[:-1] * dv[:-1]) * np.diff(z)
    integ = np.concatenate([[0.0], np.cumsum(incr)])
    w = -integ / rho

    # stop w where the divergence stops : above z_top it would be pure
    # extrapolation of the last divergence value, not a measurement.
    w[z > z_top] = np.nan
    return z, w


# --- run the retrieval ---
zr, ur, vrr, dvr = collect_rings()
print(f"VAD : {len(zr)} usable rings "
      f"over elevations {ELEV_MIN_DEG}-{ELEV_MAX_DEG} deg")
U_z, V_z, D_z, N_z = bin_profile(zr, ur, vrr, dvr)
z_w, W_z = integrate_w(Z_CENT, D_z)

# %%
# --------------------------------------------------------------------------- #
# 4. ICONIC VAD PLOT : vr = f(azimuth) on one ring + fitted sinusoid
# --------------------------------------------------------------------------- #

SHOW_AZIMUTH_FIT = True

if SHOW_AZIMUTH_FIT:
    PPI_ELEVATION_DEG = 5.0    # elevation of the demonstrated sweep (nearest available)
    Z_TARGET = 2.5             # km : pick the range gate reaching this height

    sw = int(np.argmin(np.abs(ELEVATIONS_DEG - PPI_ELEVATION_DEG)))
    phi = ELEV_RAD[sw]
    r0 = Z_TARGET / np.sin(phi) if np.sin(phi) > 0 else RANGES_KM[len(RANGES_KM) // 2]
    ir = int(np.argmin(np.abs(RANGES_KM - r0)))

    vr_az = vr[sw, :, ir]
    res = vad_fit_ring(vr_az)

    fig1, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(AZIMUTHS_DEG, vr_az, "b.", ms=4, label="vr data")
    if res is not None:
        a0, a1, b1, n, rms = res
        th = np.deg2rad(np.linspace(0, 360, 361))
        ax1.plot(np.rad2deg(th), a0 + a1 * np.sin(th) + b1 * np.cos(th),
                 "r-", lw=1.8, label=f"VAD fit (rms={rms:.2f} m/s)")
        u = a1 / np.cos(phi); v = b1 / np.cos(phi)
        spd = np.hypot(u, v)
        wdir = (np.rad2deg(np.arctan2(u, v)) + 180.0) % 360.0   # meteo : FROM direction
        ax1.set_title(f"VAD  elev {ELEVATIONS_DEG[sw]:.2f} deg, "
                      f"r={RANGES_KM[ir]:.1f} km (z={RANGES_KM[ir]*np.sin(phi):.1f} km)  |  "
                      f"u={u:.1f} v={v:.1f}  |V|={spd:.1f} m/s from {wdir:.0f} deg")
    ax1.axhline(0.0, color="k", lw=0.6)
    ax1.set_xlabel("azimuth (deg)"); ax1.set_ylabel("vr (m/s)")
    ax1.set_xlim(0, 360); ax1.grid(True); ax1.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

# %%
# --------------------------------------------------------------------------- #
# 5. VERTICAL PROFILES : u(z), v(z), divergence(z), w(z)
# --------------------------------------------------------------------------- #

fig2, (axU, axD, axW) = plt.subplots(1, 3, figsize=(13, 6), sharey=True)

# --- u, v profiles ---
axU.plot(U_z, Z_CENT, "b.-", label="u (East) VAD")
axU.plot(V_z, Z_CENT, "g.-", label="v (North) VAD")
axU.axvline(0.0, color="k", lw=0.6)
axU.set_xlabel("u, v (m/s)"); axU.set_ylabel("altitude z (km)")
axU.set_title("Horizontal wind"); axU.grid(True)

# --- divergence profile ---
axD.plot(D_z, Z_CENT, "m.-", label="DIV VAD")
axD.axvline(0.0, color="k", lw=0.6)
axD.set_xlabel("horizontal divergence ((m/s)/km)")
axD.set_title("Divergence  du/dx + dv/dy"); axD.grid(True)

# --- vertical velocity profile ---
axW.plot(W_z, z_w, "r.-", label="w VAD (integrated)")
axW.axvline(0.0, color="k", lw=0.6)
axW.set_xlabel("w (m/s)"); axW.set_title("Vertical wind (continuity)"); axW.grid(True)

# --- overlay the TRUE field for synthetic-data validation ---
if (not USE_REAL_DATA) and true_wind is not None:
    zt = Z_CENT
    ut, vt, wt = true_wind(np.zeros_like(zt), np.zeros_like(zt), zt)   # profile above radar
    axU.plot(ut, zt, "b--", lw=1, alpha=0.6, label="u true")
    axU.plot(vt, zt, "g--", lw=1, alpha=0.6, label="v true")
    axD.axvline(DIV_TRUE, color="m", ls="--", lw=1, alpha=0.6, label="DIV true")
    axW.plot(wt, zt, "r--", lw=1, alpha=0.6, label="w true")

axU.legend(loc="best", fontsize=8)
axD.legend(loc="best", fontsize=8)
axW.legend(loc="best", fontsize=8)
fig2.suptitle("VAD wind retrieval — single mono-static radar", y=1.02)
plt.tight_layout()
plt.show()

# %%
# --------------------------------------------------------------------------- #
# 6. PPI VIEW OF THE RAW RADIAL VELOCITY (Py-ART) + the VAD ring
# --------------------------------------------------------------------------- #
# Requires the project .venv (Python 3.12 + arm_pyart).

SHOW_PPI = False

if SHOW_PPI:
    import pyart

    PPI_ELEVATION_DEG = 5.0    # elevation of the displayed sweep (nearest available)
    Z_RING = 2.5               # km : the green circle marks where this sweep crosses z
    VEL_LIM = None             # color scale (m/s) ; None -> auto-fit to the sweep

    n_elv, n_azim, n_rng = vr.shape

    radar_ppi = pyart.testing.make_empty_ppi_radar(n_rng, n_azim, n_elv)
    radar_ppi.range["data"] = RANGES_KM * 1000.0                       # m
    radar_ppi.azimuth["data"] = np.tile(AZIMUTHS_DEG, n_elv)
    radar_ppi.elevation["data"] = np.repeat(ELEVATIONS_DEG, n_azim)
    radar_ppi.fixed_angle["data"] = ELEVATIONS_DEG.copy()
    radar_ppi.fields["velocity"] = {
        "data": np.ma.masked_invalid(vr.reshape(n_elv * n_azim, n_rng)),
        "units": "m/s",
        "long_name": "Radial velocity (away from radar positive)",
        "standard_name": "radial_velocity",
    }

    sw = int(np.argmin(np.abs(ELEVATIONS_DEG - PPI_ELEVATION_DEG)))
    print(f"PPI : elevation {PPI_ELEVATION_DEG:.1f} deg -> sweep {sw} ({ELEVATIONS_DEG[sw]:.2f} deg)")

    if VEL_LIM is None:
        finite = np.abs(vr[sw][np.isfinite(vr[sw])])
        VEL_LIM = max(float(np.ceil(np.percentile(finite, 99))) if finite.size else 30.0, 1.0)

    display = pyart.graph.RadarDisplay(radar_ppi)
    fig3, ax3 = plt.subplots(figsize=(8, 7))
    display.plot_ppi("velocity", sweep=sw, ax=ax3, fig=fig3,
                     vmin=-VEL_LIM, vmax=VEL_LIM, cmap="RdBu_r",
                     colorbar_label="vr (m/s)",
                     title=f"PPI radial velocity — elevation {ELEVATIONS_DEG[sw]:.2f} deg")
    display.plot_range_rings([20, 40, 60, 80], ax=ax3, lw=0.5, ls="--")

    # green circle = ground-range ring of the VAD circle used for the azimuth fit
    phi = ELEV_RAD[sw]
    if np.sin(phi) > 0:
        s_ring = Z_RING / np.tan(phi)
        th_c = np.linspace(0.0, 2.0 * np.pi, 361)
        ax3.plot(s_ring * np.sin(th_c), s_ring * np.cos(th_c), "g-", lw=2.0,
                 label=f"VAD ring (z = {Z_RING} km)")
        ax3.legend(loc="upper right", fontsize=8)
    ax3.set_aspect("equal")

    plt.tight_layout()
    plt.show()

# %%
