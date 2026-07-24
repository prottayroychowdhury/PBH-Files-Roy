import numpy as np
import matplotlib.pyplot as plt
from pbh17 import *

z_grid = np.logspace(np.log10(30), 5, 1000)
fig_masses = [1, 1e2, 1e4]

def figure6():
    plt.figure(figsize=(6.5, 4.8)); vrel = v_rel_eff(z_grid)
    for M in fig_masses:
        yc = epsilon_over_mdot(M, z_grid, "collisional", vrel)
        yp = epsilon_over_mdot(M, z_grid, "photoionization", vrel)
        zp, yp = stop_photo(z_grid, yp, yc)
        line, = plt.plot(z_grid, yc, label=mass_labels[M])
        plt.plot(zp, yp, "--", color=line.get_color())
    setup("YAH'17 Figure 6", r"$\epsilon/\dot m$", (50, 2e4), (1e-7, 3e-3))

def figure7():
    plt.figure(figsize=(6.5, 4.8))
    plt.plot(z_grid, v_B(z_grid)/1e3, ":", label=r"$v_B$")
    plt.plot(z_grid, v_L_rms(z_grid)/1e3, "--", label=r"$\langle v_L^2\rangle^{1/2}$")
    plt.plot(z_grid, v_eff(z_grid)/1e3, label=r"$v_{\rm eff}$")
    setup("YAH'17 Figure 7", r"Velocity [km s$^{-1}$]", (50, 1e5), (3e-1, 8e1))

def figure8():
    plt.figure(figsize=(6.5, 4.8))
    for M in fig_masses:
        yc = average_L(M, z_grid, "collisional") / L_Edd(M)
        yp = average_L(M, z_grid, "photoionization") / L_Edd(M)
        zp, yp = stop_photo(z_grid, yp, yc)
        line, = plt.plot(z_grid, yc, label=mass_labels[M])
        plt.plot(zp, yp, "--", color=line.get_color())
    setup("YAH'17 Figure 8", r"$\langle L\rangle/L_{\rm Edd}$", (50, 1e5), (1e-16, 1e-2))

def main():
    figure6()
    figure7()
    figure8()
    plt.show()

if __name__ == "__main__":
    main()