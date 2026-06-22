"""
================================================================================
 Génération de données radar Doppler mono-statique FICTIVES pour la méthode VVP
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# 1. CHAMP DE VENT "VRAI"  (modèle LINÉAIRE)
# --------------------------------------------------------------------------- #
# Ce générateur VVP mono-statique possède son PROPRE champ de vent, INDÉPENDANT
# de celui du générateur dual-Doppler (generate_dual_radar_data.py).
#
# Le vent est un champ LINÉAIRE (développement de Taylor au 1er ordre) :
#     u(x,y,z) = u0 + dudx*x + dudy*y + dudz*z
#     v(x,y,z) = v0 + dvdx*x + dvdy*y + dvdz*z
#     w(x,y,z) = w0 + dwdx*x + dwdy*y + dwdz*z
# avec x,y,z en km et u,v,w en m/s. C'est exactement la forme que la VVP ajuste
# localement, donc le champ est entièrement décrit par (u0,v0,w0) et leurs
# dérivées spatiales (constantes).

# ---- Composantes du vent à l'origine (0,0,0)  [m/s] ----
u0, v0, w0 = 4.0, 6.0, 0.0     # vent moyen ~7 m/s du SO

# ---- Dérivées spatiales du vent  [ (m/s) / km ] ----
# Champ "pas trop simple" : structure HORIZONTALE (rotation + déformation), donc
# NON uniforme sur une couche. Aucune variation verticale (dérivées en z = 0) :
# le champ est identique à toutes les altitudes, et w = 0 partout.
dudx = 0.15    # du/dx ─┐  divergence horizontale  D = dudx + dvdy = 0 (non divergent)
dvdy = -0.15   # dv/dy ─┘  -> déformation d'étirement (stretching = dudx - dvdy = 0.30)
dwdz = 0.0     # dw/dz     (dérivée en z nulle)
dudy = -0.20   # du/dy ─┐  vorticité verticale = dvdx - dudy = 0.40 (rotation cyclonique)
dvdx = 0.20    # dv/dx ─┘  déformation de cisaillement = dudy + dvdx = 0
dudz = 0.0     # du/dz ─┐  dérivées en z nulles : pas de cisaillement vertical
dvdz = 0.0     # dv/dz ─┘
dwdx = 0.0     # dw/dx
dwdy = 0.0     # dw/dy


def true_wind(x, y, z):
    """(u, v, w) en m/s ; x, y, z en km. Champ linéaire à gradients constants."""
    u = u0 + dudx * x + dudy * y + dudz * z
    v = v0 + dvdx * x + dvdy * y + dvdz * z
    w = w0 + dwdx * x + dwdy * y + dwdz * z
    # garantit la même forme pour les 3 composantes (cas de gradients tous nuls)
    shp = np.broadcast_shapes(np.shape(u), np.shape(v), np.shape(w))
    return (np.broadcast_to(u, shp).copy(),
            np.broadcast_to(v, shp).copy(),
            np.broadcast_to(w, shp).copy())


x_value = np.linspace(-10, 10, 10)  # km
y_value = np.linspace(-10, 10, 10)  # km
z_value = np.linspace(0, 10, 5)     # km

X, Y, Z = np.meshgrid(x_value, y_value, z_value, indexing='ij')

U, V, W = true_wind(X, Y, Z)

# --------------------------------------------------------------------------- #
# 2. AFFICHAGE DU CHAMP DE VENT VRAI
# --------------------------------------------------------------------------- #

afficher_champ_vrai = False  # mettre à True pour afficher le champ vrai

if afficher_champ_vrai:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Affichage des flèches
    ax.quiver(X, Y, Z, U, V, W, length=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

show_layer = True

if show_layer:
    # Affiche le vent VRAI sur une couche horizontale à Z_LAYER (même style que le
    # plot de couche de la VVP : fond = |V|, flèches unitaires = direction).
    Z_LAYER = 2.5                          # km, altitude de la couche affichée
    xg = np.arange(-90.0, 90.0, 5.0)       # km
    yg = np.arange(-90.0, 90.0, 5.0)       # km
    XL, YL = np.meshgrid(xg, yg)
    UL, VL, _ = true_wind(XL, YL, Z_LAYER)

    speed = np.hypot(UL, VL)
    fig, ax = plt.subplots(figsize=(8, 7))
    pm = ax.pcolormesh(XL, YL, speed, cmap="YlOrRd", alpha=0.7, shading="nearest")
    fig.colorbar(pm, ax=ax, label="|V| (m/s)")
    norm = np.where(speed > 0, speed, np.nan)   # flèches unité : direction seule
    ax.quiver(XL, YL, UL / norm, VL / norm, scale=40, width=0.003, color="k")
    ax.scatter([0], [0], color="k", marker="^", s=60)   # radar à l'origine
    ax.set_title(f"True wind field — z layer = {Z_LAYER} km")
    ax.set_xlabel("X Est (km)"); ax.set_ylabel("Y Nord (km)")
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# 3. GÉNÉRATION DES DONNÉES RADAR DOPPLER
# --------------------------------------------------------------------------- #

radar_type = 'PAWR'

if radar_type == 'doppler' :
    # VCP type WSR-88D (VCP-12/212)
    ELEVATIONS_DEG = np.array([0.5, 0.9, 1.3, 1.8, 2.4, 3.1, 4.0,
                            5.1, 6.4, 8.0, 10.0, 12.5, 15.6, 19.5]) # 14 angles
    AZIMUTHS_DEG = np.arange(0.0, 360.0, 1.0)        # 360 rayons, pas 1 deg
    RANGES_KM = np.arange(0.25, 80, 0.25)          # 
elif radar_type == 'PAWR' :
    ELEVATIONS_DEG = np.concatenate([
    np.arange(-0.03, 17.0, 0.5),    # -0.03° → 16.97°   (35 élévations, pas 0.5°)
    np.arange(17.97, 60.0, 1.0),    # 17.97° → 59.97°   (43 élévations, pas 1.0°)
    ]) 
    AZIMUTHS_DEG = (np.arange(300) + 0.5) * 1.2     # 300 secteurs, pas 1.2° (0.6° → 359.4°)
    RANGES_KM = (np.arange(534) + 0.5) * 0.15       # 534 portes de 150 m (0.075 → 80.025 km)


def generate():
    """Génère les données radar Doppler (vr) à partir du champ de vent VRAI."""

    el, az, r = np.meshgrid(np.deg2rad(ELEVATIONS_DEG),np.deg2rad(AZIMUTHS_DEG),RANGES_KM,indexing="ij")

    el = el.ravel()      # phi   (rad)
    az = az.ravel()      # theta (rad)
    r = r.ravel()        # km

    # 3. Coordonnées cartésiennes (faisceau rectiligne)
    cphi, sphi = np.cos(el), np.sin(el)
    sth, cth = np.sin(az), np.cos(az)
    x = r * sth * cphi     # Est
    y = r * cth * cphi     # Nord
    z = r * sphi           # Haut

    # 4. Vent vrai aux points de mesure
    u, v, w = true_wind(x, y, z)

    # 5. Vitesse radiale = projection du vent sur le faisceau
    #    (vecteur unitaire du faisceau : [sth*cphi, cth*cphi, sphi])
    vr = u * (sth * cphi) + v * (cth * cphi) + w * sphi
    #vr = vr.reshape(len(ELEVATIONS_DEG), len(AZIMUTHS_DEG), len(RANGES_KM))
    return vr

