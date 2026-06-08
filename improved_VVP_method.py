import numpy as np
import matplotlib.pyplot as plt
from generate_radar_data import generate,true_wind

vr = generate() #on recupère les données radar générées

# --------------------------------------------------------------------------- #
# 1. RECONSTRUCTION DES COORDONNÉES RADAR
# --------------------------------------------------------------------------- #

# Domaine de mesure du radar : 13 élévations, 360 azimuts, 96 portes (2 à 200 km)
ELEVATIONS_DEG = np.array([0.5, 1.5, 2.4, 3.4, 4.3, 5.3, 6.2, 7.5, 8.7, 10.0, 12.0, 14.0, 16.7, 19.5])
AZIMUTHS_DEG = np.arange(0.0, 360.0, 1.0)        # 360 rayons, pas de 1°
RANGES_KM = np.arange(2.0, 200.0, 0.5)     # portes de 2 à 200 km, pas 500 m

phi, theta, r = np.meshgrid(np.deg2rad(ELEVATIONS_DEG),np.deg2rad(AZIMUTHS_DEG),RANGES_KM,indexing="ij")
cphi, sphi = np.cos(phi), np.sin(phi)
sth, cth = np.sin(theta), np.cos(theta)
x = r * sth * cphi
y = r * cth * cphi
z = r * sphi

# --------------------------------------------------------------------------- #
# 2. CONSTRUCTION DE LA MATRICE G (N x 12)
# --------------------------------------------------------------------------- #

def design_matrix(r, theta, phi, x0, y0, z0):
    """
    Construit G (N x 9) à partir des points (r, theta, phi) d'un volume,
    pour une altitude de référence z0.
    """
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
# 3. RÉSOLUTION DU SYSTÈME 
# --------------------------------------------------------------------------- #

def solve(G, d, method="direct"):
    if method == "direct":
        A = np.dot(G.T, G)
        B = np.dot(G.T, d)
        X = np.linalg.solve(A, B)
        return X, np.linalg.cond(A)
    
# --------------------------------------------------------------------------- #
# 4. RESTITUTION POUR UN POINT
# --------------------------------------------------------------------------- #

# Demi-largeurs du volume d'analyse
D_R     = 10.0    # km
D_THETA = 10.0   # deg
D_PHI   = 15.0   # deg  (GRAND pour garder toutes les élévations)
 
# Aplatissement 
rf  = r.ravel()
thf = theta.ravel()
phf = phi.ravel()
zf  = z.ravel()
vf  = vr.ravel()

min_n = 10  # nombre minimum de points pour faire une estimation du vent 
 
def retrieve_wind(x0, y0, z0):
    """
    Restitue le vent au point (x0, y0, z0).
    Retourne (u, v, w) en m/s, ou None si pas assez de données.
    """
 
    # ---- Étape 1 : (x0, y0, z0) est donné en argument ----
 
    # ---- Étape 2 : conversion en coordonnées polaires ----
    r0  = np.sqrt(x0**2 + y0**2 + z0**2)        # distance au radar (km)
    th0 = np.arctan2(x0, y0)                     # azimut (rad, depuis le Nord)
    ph0 = np.arcsin(z0 / r0)                     # élévation (rad)
 
    # ---- Étape 3 : sélection des mesures dans le volume ----
    centre = np.array([r0, np.rad2deg(th0), np.rad2deg(ph0)])

    # écart d'azimut en gérant le passage 0/360°
    dth = np.abs(thf - np.deg2rad(centre[1]))
    dth = np.minimum(dth, 2 * np.pi - dth)

    mask = (dth < np.deg2rad(D_THETA)) & \
           (np.abs(phf - np.deg2rad(centre[2])) < np.deg2rad(D_PHI)) & \
           (np.abs(rf - centre[0]) < D_R)
    n = np.count_nonzero(mask)
    if n < min_n:
        return None
 
    r_sel  = rf[mask]
    th_sel = thf[mask]
    ph_sel = phf[mask]
    vr_sel = vf[mask]
 
    # ---- Étape 4 : construction de G ----
    G = design_matrix(r_sel, th_sel, ph_sel, x0, y0, z0)
 
    # ---- Étape 5 & 6 : formation de A X = B  et résolution ----
    try:
        X, cond = solve(G, vr_sel)
    except np.linalg.LinAlgError:
        return None
    #print("cond =", cond)
    #print("N =", len(vr_sel))
 
    # ---- Étape 7 : reconstruction du vent au point choisi ----

    u0, v0, w0     = X[0], X[1], X[2]
    ux, vy, cross  = X[3], X[4], X[5]

 
    return u0, v0, w0

# --------------------------------------------------------------------------- #
# 5. RESTITUTION POUR TOUT L'ESPACE
# --------------------------------------------------------------------------- #

def sweep_workspace(x_vals, y_vals, z_vals, **kwargs):
    """
    Restitue le vent sur une grille cartesienne (km).
    Retourne 4 tableaux des points valides : xs, ys, zs, vents (Nx3).
    """
    xs, ys, zs, winds = [], [], [], []
    for z0 in z_vals:
        for y0 in y_vals:
            for x0 in x_vals:
                res = retrieve_wind(float(x0), float(y0), float(z0), **kwargs)
                if res is not None:
                    xs.append(x0); ys.append(y0); zs.append(z0)
                    winds.append(res)
    return (np.array(xs), np.array(ys), np.array(zs), np.array(winds))
 
 
# Grille couvrant l'espace de travail 
X_GRID = np.arange(-120.0, 121.0, 30.0)
Y_GRID = np.arange(-120.0, 121.0, 30.0)
Z_GRID = np.arange(1.0, 9.0, 2.0)
 
xs, ys, zs, winds = sweep_workspace(X_GRID, Y_GRID, Z_GRID)
print(f"Points restitues : {len(xs)} / {len(X_GRID) * len(Y_GRID) * len(Z_GRID)}")
 
U_rec, V_rec, W_rec = winds[:, 0], winds[:, 1], winds[:, 2]
 
# Erreur par rapport au champ vrai (validation)
U_t, V_t, W_t = true_wind(xs, ys, zs)
err = np.sqrt((U_rec - U_t)**2 + (V_rec - V_t)**2 + (W_rec - W_t)**2)
print(f"Erreur vectorielle moyenne : {err.mean():.3e} m/s")
 
# --------------------------------------------------------------------------- #
# 6. COMPARAISON PAR COUCHES : (u,v) en flèches, w en couleur + profil w(z)
# --------------------------------------------------------------------------- #

X_GRID   = np.arange(-100.0, 101.0, 20.0)
Y_GRID   = np.arange(-100.0, 101.0, 20.0)
Z_LAYERS = [2.0, 6.0]      # couches comparées (km)
QUIVER_SCALE = 300              # plus grand = flèches plus courtes

# --- collecte des données pour chaque couche ---
data, w_all = {}, []
for z0 in Z_LAYERS:
    xr, yr, ur, vr_rec, wr = [], [], [], [], []
    xt, yt, ut, vt, wt = [], [], [], [], []
    for y0 in Y_GRID:
        for x0 in X_GRID:
            tu, tv, tw = true_wind(np.array(x0), np.array(y0), np.array(z0))
            xt.append(x0); yt.append(y0)
            ut.append(float(tu)); vt.append(float(tv)); wt.append(float(tw))
            res = retrieve_wind(float(x0), float(y0), z0)
            if res is not None:
                xr.append(x0); yr.append(y0)
                ur.append(res[0]); vr_rec.append(res[1]); wr.append(res[2])
    data[z0] = (xr, yr, ur, vr_rec, wr, xt, yt, ut, vt, wt)
    w_all += wr + wt

# échelle de couleur commune et symétrique
vmax = max(1e-6, np.max(np.abs(w_all)))

# --- figure : 1 ligne par couche (quivers colorés) + 1 ligne profil ---
n = len(Z_LAYERS)
fig, axes = plt.subplots(n + 1, 2, figsize=(13, 5 * (n + 1)))

for i, z0 in enumerate(Z_LAYERS):
    xr, yr, ur, vr_rec, wr, xt, yt, ut, vt, wt = data[z0]
    axL, axR = axes[i]

    qL = axL.quiver(xr, yr, ur, vr_rec, wr, cmap="RdBu_r",
                    clim=(-vmax, vmax), scale=QUIVER_SCALE)
    axL.scatter([0], [0], color="k", marker="^", s=60)
    axL.set_title(f"Restitué VVP — z = {z0} km ({len(xr)} pts)")
    axL.set_xlabel("X Est (km)"); axL.set_ylabel("Y Nord (km)")
    axL.set_aspect("equal")
    fig.colorbar(qL, ax=axL, label="w (m/s)")

    qR = axR.quiver(xt, yt, ut, vt, wt, cmap="RdBu_r",
                    clim=(-vmax, vmax), scale=QUIVER_SCALE)
    axR.scatter([0], [0], color="k", marker="^", s=60)
    axR.set_title(f"Vrai champ — z = {z0} km")
    axR.set_xlabel("X Est (km)"); axR.set_ylabel("Y Nord (km)")
    axR.set_aspect("equal")
    fig.colorbar(qR, ax=axR, label="w (m/s)")

# --- dernière ligne : profil vertical w(z) en 2 points ---
z_profile = np.arange(1.0, 9.0, 0.5)
for ax, (px, py) in zip(axes[n], [(0.0, 60.0), (40.0, 40.0)]):
    w_rec, w_true = [], []
    for z0 in z_profile:
        res = retrieve_wind(px, py, float(z0))
        w_rec.append(res[2] if res is not None else np.nan)
        w_true.append(float(true_wind(np.array(px), np.array(py), np.array(z0))[2]))
    ax.plot(w_true, z_profile, "g-o", label="vrai w")
    ax.plot(w_rec,  z_profile, "b--s", label="w restitué")
    ax.set_title(f"Profil w(z) au point ({px:.0f}, {py:.0f}) km")
    ax.set_xlabel("w (m/s)"); ax.set_ylabel("altitude (km)")
    ax.grid(True); ax.legend()

plt.tight_layout()
plt.show()

