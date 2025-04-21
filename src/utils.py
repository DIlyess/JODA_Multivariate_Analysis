from scipy.interpolate import griddata

import numpy as np
import matplotlib.pyplot as plt


def plot_3d_surface(x, y, z, xlabel, ylabel, title):
    """Trace une surface 3D colorée en fonction de la valeur de z."""
    # Création d'une grille régulière
    xi = np.linspace(x.min(), x.max(), 30)
    yi = np.linspace(y.min(), y.max(), 30)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata((x, y), z, (Xi, Yi), method="cubic")

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    # Surface avec mappage de couleurs selon Zi
    surf = ax.plot_surface(
        Xi,
        Yi,
        Zi,
        cmap="viridis",  # colormap automatique
        edgecolor="none",  # pas de trait de maille pour la surface
        antialiased=True,
    )
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="RMSE")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("RMSE")
    ax.set_title(title)
    plt.tight_layout()
