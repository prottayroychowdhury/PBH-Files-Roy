'''
Author: Roy Chowdhury
Description: We enforce the self-consistency condition r_ion_end = R, where R is the Stromgren radius, which physically corresponds to an improved
ionization model where the gas first undergoes pure photoionization from r = R_S to r = R and then pure collisional ionization from r = R to r = r_ion.
To achieve this self-consistency, we solve T_S_func(R_func(T_S)) = T_S, where R_func(T_S) is obtained by redefining chi with r_ion_end = R
and R_func(T_S) is given by Eq. 72 in Ali-Haimoud and Kamionkowski (2017). Finally, we plotted luminosity as a function of the self-consistent
temperature at Schwarzchild radius using Eq. 57, which yields a curve that falls between those corresponding to the two limiting cases explored
in the 2017 paper.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from pbh.constants import *
from pbh.history import *
from pbh.bondi import *
from pbh.utils import *
import pbh.luminosity as old

E_ion = 13.6 * 1.6e-19

def r_ion(M, z):
    Tau, T_inf, r_b = tau(M,z), T_b(z), r_B(M,z)
    T_ion = 1.5e4
    return Tau * (T_inf/T_ion) * r_b
    
def chi(R, M, z):
    r_is = r_ion(M,z)
    xe = x_e_pbh(z)
    r_ie = r_is * (1+xe)**8 / 256
    if R >= r_is:
        return 1
    elif R < r_ie:
        return (2/(1+xe))**12
    return (r_is/R)**(3/2)

def Y_S(R,M,z):
    xe, ta, ch = x_e_pbh(z), tau(M, z), chi(R,M,z)
    return ch**(-2/3) * 2/(1+xe) * ta/4 * (1 - 2.5*ta)**(1/3) * m_p/m_e

def F(Y):
    return Y * (1 + Y / 0.27)**(-1/3)

def T_S_func(R,M,z):
    return m_e * c**2 / k * F(Y_S(R,M,z))

def J(X):
    X = np.asarray(X); out = np.empty_like(X, dtype=float); low = X < 1
    out[low] = 4/pi * np.sqrt(2/(pi * X[low])) * (1 + 5.5 * X[low]**1.25)
    out[~low] = 27/(2*pi) * (np.log(2 * X[~low] * np.exp(-np.euler_gamma) + 0.08) + 4/3)
    return out

def R_func(TS, M, z):
    X = k * TS / (m_e * c**2)
    return 2e-4 * r_ion(M, z) * (J(X) * np.log(k * TS / E_ion))**1.16

def temperature_residual(log_TS, M, z):
    TS_guess = np.exp(log_TS)
    TS_new = T_S_func(R_func(TS_guess, M, z), M, z)
    return np.log(TS_new / TS_guess)

def solve_TS(M, z):
    Tc, Tp = old.T_S(M, z, "collisional"), old.T_S(M, z, "photoionization")
    lo, hi = np.log(min(Tc, Tp)), np.log(max(Tc, Tp))
    flo, fhi = temperature_residual(lo, M, z), temperature_residual(hi, M, z)
    if np.isclose(flo, 0, atol=1e-12):
        return np.exp(lo)
    if np.isclose(fhi, 0, atol=1e-12):
        return np.exp(hi)
    log_TS = brentq(temperature_residual, lo, hi, args=(M, z))
    return np.exp(log_TS)

def L_from_TS(TS, M, z):
    Mdot = 4 * pi * rho_b(z) * r_B(M, z)**2 * v_B(z) * lambda_acc(M, z)
    L_Edd = 4 * pi * G * M * M_sun * m_p * c / sigma_T
    X = k * TS / (m_e * c**2)
    return alpha * k * TS / (m_p * c**2) * J(X) * (Mdot * c**2)**2 / L_Edd

def solve_redshift_curve(M, z_grid=None):
    z = np.logspace(np.log10(30), 5, 160) if z_grid is None else np.asarray(z_grid)
    TS = np.array([solve_TS(M, zi) for zi in z])
    TS_coll = old.T_S(M, z, "collisional")
    TS_photo = old.T_S(M, z, "photoionization")
    L = L_from_TS(TS, M, z)
    L_coll = L_from_TS(TS_coll, M, z)
    L_photo = L_from_TS(TS_photo, M, z)
    Ledd = old.L_Edd(M)
    return {"z": z, "T_S": TS, "L": L, "L_over_L_Edd": L/Ledd, "L_coll_over_L_Edd": L_coll/Ledd, "L_photo_over_L_Edd": L_photo/Ledd}

def plot_single_branch_luminosity(masses=(1, 1e2, 1e4), z_grid=None):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for M, color in zip(masses, colors):
        s = solve_redshift_curve(M, z_grid)
        z, L = s["z"], s["L_over_L_Edd"]
        Lc, Lp = s["L_coll_over_L_Edd"], s["L_photo_over_L_Edd"]
        plt.loglog(z, Lc, ":", color=color, label=mass_labels[M]+" collisional")
        plt.loglog(z, L, "-", color=color, label=mass_labels[M]+" combined")
        plt.loglog(z, Lp, "--", color=color, label=mass_labels[M]+" photo")
    plt.xlabel(r"$z$")
    plt.ylabel(r"$L/L_{\rm Edd}$")
    plt.xlim(30, 1e5)
    plt.legend(fontsize=8)
    plt.tight_layout()
    return plt.gcf()

if __name__ == "__main__":
    plot_single_branch_luminosity()
    plt.show()