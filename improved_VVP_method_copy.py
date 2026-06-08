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
    # différence angulaire en azimut (gestion du passage 0/360)
    centre = np.array([r0, np.rad2deg(th0), np.rad2deg(ph0)]) # centre du volume d'analyse 
    mask = (np.abs(thf - np.deg2rad(centre[1])) < np.deg2rad(D_THETA)) & \
           (np.abs(phf - np.deg2rad(centre[2])) < np.deg2rad(D_PHI)) & \
           (np.abs(rf - centre[0]) < D_R)
 
    n = np.count_nonzero(mask)
 
    r_sel  = rf[mask]
    th_sel = thf[mask]
    ph_sel = phf[mask]
    vr_sel = vf[mask]
 
    # ---- Étape 4 : construction de G ----
    G = design_matrix(r_sel, th_sel, ph_sel, x0, y0, z0)
 
    # ---- Étape 5 & 6 : formation de A X = B  et résolution ----
    X, cond = solve(G, vr_sel)
    print("cond =", cond)
    print("N =", len(vr_sel))
 
    # ---- Étape 7 : reconstruction du vent au point choisi ----

    u0, v0, w0     = X[0], X[1], X[2]
    ux, vy, cross  = X[3], X[4], X[5]

 
    return u0, v0, w0

x0 = 50.0
y0 = 30.0
z0 = 3.0
 
u_rec, v_rec, w_rec = retrieve_wind(x0, y0, z0)


print("VVP  :", u_rec, v_rec, w_rec)

