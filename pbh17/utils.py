import numpy as np
import matplotlib.pyplot as plt

mass_labels = {1: r"$1\,M_\odot$", 10: r"$10\,M_\odot$", 1e2: r"$10^2\,M_\odot$", 1e3: r"$10^3\,M_\odot$", 1e4: r"$10^4\,M_\odot$", 1e5: r"$10^5\,M_\odot$"}

def setup(title, ylabel, xlim, ylim):
    plt.xscale("log"); plt.yscale("log"); plt.xlim(*xlim); plt.ylim(*ylim)
    plt.xlabel(r"$z$"); plt.ylabel(ylabel); plt.title(title); plt.legend(); plt.tight_layout()

def stop_photo(z, y_photo, y_coll):
    valid = (y_photo > 0) & (y_coll > 0) & np.isfinite(y_photo) & np.isfinite(y_coll)
    z, y_photo, y_coll = z[valid], y_photo[valid], y_coll[valid]
    diff = y_photo - y_coll
    cross = np.where(diff[:-1] * diff[1:] <= 0)[0]
    return (z, y_photo) if len(cross) == 0 else (z[:cross[0]+1], y_photo[:cross[0]+1])
