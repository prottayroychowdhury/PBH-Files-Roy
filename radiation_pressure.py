'''
Author: Roy Chowdhury
Description: We compute a self-consistent luminosity function which takes the radiation pressure feedback from the black hole into account,
reducing the effective gravitational mass experienced by the infalling gas by a factor of (1 - L/L_Edd). The equation L_func(M_eff(L)) = L
is solved to obtain this self-consistency. Notably, the effect of radiation pressure becomes significant for PBHs with subsonic relative velocity
when the rest mass of the black hole reaches around 10^4 solar masses.
'''

import numpy as np
import matplotlib.pyplot as plt
from pbh17 import *

M_vals = np.logspace(0, 5, 220)
z_vals = np.logspace(2, 5, 120)
fig_masses = [10, 1e3, 1e5]

def clean_L(Lval, Lmin, Lmax):
    if Lval <= 0: 
        return Lmin
    if Lval >= Lmax: 
        return Lmax
    return Lval

def luminosity_error(M, L_guess, z, mode):
    Ledd = L_Edd(M)
    Lmin, Lmax = 1e-30 * Ledd, 0.999 * Ledd
    M_eff = M * (1 - L_guess / Ledd)
    L_new = clean_L(L(M_eff, z, mode, 0), Lmin, Lmax)
    residual = (L_new - L_guess) / Ledd
    return L_new, residual, M_eff

def luminosity_iteration(M, z, mode, L_initial=None):
    Ledd = L_Edd(M)
    Lmin, Lmax = 1e-30 * Ledd, 0.999 * Ledd
    L_guess = 0.5 * Ledd if L_initial is None else clean_L(L_initial, Lmin, Lmax)
    damping, max_iter = (0.005, 4000) if M >= 1e5 else ((0.01, 2500) if M >= 1e4 else (0.05, 1000))
    for i in range(max_iter):
        L_guess = clean_L(L_guess, Lmin, Lmax)
        L_new, residual, M_eff = luminosity_error(M, L_guess, z, mode)
        if abs(residual) < 0.01: 
            break
        L_guess = (1 - damping) * L_guess + damping * L_new
    return {"L": L_new, "L_over_Ledd": L_new/Ledd, "M_eff": M_eff, "M_eff_over_M": M_eff/M, "residual": residual, "iterations": i+1}

def solve_redshift_curve(M, mode, z_grid=None):
    z_grid = z_vals if z_grid is None else z_grid
    L_initial, L_over_Ledd, M_eff_over_M = None, [], []
    for z in z_grid:
        sol = luminosity_iteration(M, z, mode, L_initial)
        L_initial = sol["L"]
        L_over_Ledd.append(sol["L_over_Ledd"]); M_eff_over_M.append(sol["M_eff_over_M"])
    return {"z": z_grid, "L_over_Ledd": np.array(L_over_Ledd), "M_eff_over_M": np.array(M_eff_over_M)}

def solve_Meff_curve(z, mode, M_grid=None):
    M_grid = M_vals if M_grid is None else M_grid
    L_initial, M_eff, M_eff_over_M = None, [], []
    for M in M_grid:
        sol = luminosity_iteration(M, z, mode, L_initial)
        L_initial = sol["L"]
        M_eff.append(sol["M_eff"]); M_eff_over_M.append(sol["M_eff_over_M"])
    return {"M": M_grid, "M_eff": np.array(M_eff), "M_eff_over_M": np.array(M_eff_over_M)}

def plot_luminosity_vs_z(masses=fig_masses):
    plt.figure(figsize=(7.5, 5.2))
    for M in masses:
        coll, photo = solve_redshift_curve(M, "collisional"), solve_redshift_curve(M, "photoionization")
        line, = plt.loglog(coll["z"], coll["L_over_Ledd"], linewidth=2, label=mass_labels[M] + " collisional")
        plt.loglog(photo["z"], photo["L_over_Ledd"], "--", linewidth=2, color=line.get_color(), label=mass_labels[M] + " photoionization")
    setup(r"Self-consistent luminosity", r"$L/L_{\rm Edd}$", (1e2, 1e5), (1e-16, 2))

def plot_Meff_ratio_vs_z(masses=mass_labels.keys()):
    plt.figure(figsize=(7.5, 5.2))
    for M in masses:
        coll, photo = solve_redshift_curve(M, "collisional"), solve_redshift_curve(M, "photoionization")
        line, = plt.semilogx(coll["z"], coll["M_eff_over_M"], linewidth=2, label=mass_labels[M] + " collisional")
        plt.semilogx(photo["z"], photo["M_eff_over_M"], "--", linewidth=2, color=line.get_color(), label=mass_labels[M] + " photoionization")
    plt.axhline(1, linestyle=":", linewidth=1.2)
    plt.xlabel(r"$z$"); plt.ylabel(r"$M_{\rm eff}/M$"); plt.title(r"Effective mass suppression")
    plt.xlim(1e2, 1e5); plt.ylim(0, 1.05); plt.legend(); plt.tight_layout()

def main():
    plot_luminosity_vs_z()
    plot_Meff_ratio_vs_z()
    plt.show()

if __name__ == "__main__":
    main()