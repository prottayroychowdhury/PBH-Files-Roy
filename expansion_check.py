import numpy as np
import matplotlib.pyplot as plt
from pbh17 import *

def plot_tB_H(masses=mass_labels.keys()):
    z = np.logspace(0, 11, 1000)
    plt.figure(figsize=(7.2, 5.0))
    for M in masses:
        plt.loglog(z, t_B(M, z) * Hubble(z), label=mass_labels[M])
    plt.axhline(1, linestyle=":", linewidth=1, label=r"$t_BH=1$")
    plt.xlabel(r"$z$"); plt.ylabel(r"$t_BH$");
    plt.title(r"Bondi timescale vs expansion timescale")
    plt.xlim(1, 1e11); plt.ylim(1e-6, 1e4); plt.legend(); plt.tight_layout()

def main():
    plot_tB_H()
    plt.show()

if __name__ == "__main__":
    main()