"""
================================================================================
 Génération de données radar Doppler mono-statique FICTIVES pour la méthode VVP
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# 1. CHAMP DE VENT "VRAI"
# --------------------------------------------------------------------------- #
# On importe le MÊME champ de vent vrai que le générateur dual-Doppler, afin
# que la méthode VVP mono-statique et l'analyse dual-Doppler échantillonnent
# exactement le même vent. Les restitutions des deux méthodes sont ainsi
# directement comparables.
from generate_dual_radar_data import true_wind


x_value = np.linspace(-10, 10, 10)  # km
y_value = np.linspace(-10, 10, 10)  # km
z_value = np.linspace(0, 10, 5)     # km

X, Y, Z = np.meshgrid(x_value, y_value, z_value, indexing='ij')

U, V, W = true_wind(X, Y, Z)

# --------------------------------------------------------------------------- #
# 2. AFFICHAGE DU CHAMP DE VENT VRAI
# --------------------------------------------------------------------------- #

afficher_champ_vrai = True  # mettre à True pour afficher le champ vrai

if afficher_champ_vrai:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Affichage des flèches
    ax.quiver(X, Y, Z, U, V, W, length=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

afficher_couches_vrai = True  # mettre à True pour afficher les coupes du champ vrai


# --------------------------------------------------------------------------- #
# 3. GÉNÉRATION DES DONNÉES RADAR DOPPLER 
# --------------------------------------------------------------------------- #

# Domaine de mesure du radar : 13 élévations, 360 azimuts, 96 portes (2 à 200 km)
ELEVATIONS_DEG = np.array([0.5, 1.5, 2.4, 3.4, 4.3, 5.3, 6.2, 7.5, 8.7, 10.0, 12.0, 14.0, 16.7, 19.5])
AZIMUTHS_DEG = np.arange(0.0, 360.0, 1.0)        # 360 rayons, pas de 1°
RANGES_KM = np.arange(2.0, 200.0, 0.5)     # portes de 2 à 200 km, pas 500 m

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

