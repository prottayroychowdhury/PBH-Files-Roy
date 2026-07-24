import numpy as np
from .constants import *
from .history import *
from .bondi import *

def chi(z, mode):
    xe = x_e_pbh(z)
    if mode == "collisional": return (2 / (1 + xe))**12
    if mode == "photoionization": return np.ones_like(xe)
    raise ValueError("mode must be 'collisional' or 'photoionization'") 

def Y_S(M, z, mode, v_rel=0):
    xe, ta, ch = x_e_pbh(z), tau(M, z, v_rel), chi(z, mode)
    return ch**(-2/3) * 2/(1+xe) * ta/4 * (1 - 2.5*ta)**(1/3) * m_p/m_e

def F(Y):
    return Y * (1 + Y / 0.27)**(-1/3)

def T_S(M, z, mode, v_rel=0):
    return m_e * c**2 / k * F(Y_S(M, z, mode, v_rel))

def J(X):
    X = np.asarray(X); out = np.empty_like(X, dtype=float); low = X < 1
    out[low] = 4/pi * np.sqrt(2/(pi * X[low])) * (1 + 5.5 * X[low]**1.25)
    out[~low] = 27/(2*pi) * (np.log(2 * X[~low] * np.exp(-np.euler_gamma) + 0.08) + 4/3)
    return out

def epsilon_over_mdot(M, z, mode, v_rel=0):
    TS = T_S(M, z, mode, v_rel)
    return alpha * k * TS / (m_p * c**2) * J(k * TS / (m_e * c**2))

def Mdot(M, z, v_rel=0):
    return 4 * pi * rho_b(z) * r_B(M, z, v_rel)**2 * v_B(z, v_rel) * lambda_acc(M, z, v_rel)

def L_Edd(M):
    return 4 * pi * G * M * M_sun * m_p * c / sigma_T

def mdot(M, z, v_rel=0):
    return Mdot(M, z, v_rel) * c**2 / L_Edd(M)

def L(M, z, mode, v_rel=0):
    return epsilon_over_mdot(M, z, mode, v_rel) * mdot(M, z, v_rel) * Mdot(M, z, v_rel) * c**2

def average_L(M, z, mode, nv=600):
    sigma = v_L_rms(z)[:, None] / np.sqrt(3)
    q = np.linspace(0, 8, nv)
    pdf_q = np.sqrt(2/pi) * q**2 * np.exp(-q**2/2); pdf_q /= np.trapz(pdf_q, q)
    return np.trapz(L(M, z[:, None], mode, v_rel=sigma*q[None, :]) * pdf_q[None, :], q, axis=1)