import numpy as np
from .constants import *
from .history import *

def v_L_rms(z):
    return np.minimum(1.0, z / 1000.0) * 30e3

def v_B(z, v_rel=0):
    cs2 = (1 + x_e_pbh(z)) * k * T_b(z) / m_p
    return np.sqrt(cs2 + v_rel**2)

def r_B(M, z, v_rel=0):
    return G * M * M_sun / v_B(z, v_rel)**2

def t_B(M, z, v_rel=0):
    return G * M * M_sun / v_B(z, v_rel)**3

def v_eff(z):
    vB, vL = v_B(z), v_L_rms(z)
    return np.where(vB < vL, np.sqrt(vB * vL), vB)

def v_rel_eff(z):
    return np.sqrt(np.maximum(v_eff(z)**2 - v_B(z)**2, 0))

def beta(M, z, v_rel=0):
    return (4/3) * x_e_pbh(z) * sigma_T * rho_cmb(z) * t_B(M, z, v_rel) / (m_p * c)

def gamma(M, z, v_rel=0):
    return 2 * m_p * beta(M, z, v_rel) / (m_e * (1 + x_e_pbh(z)))

def lambda_of_gamma(gam):
    return lambda_ad + (lambda_iso - lambda_ad) * (gam**2 / (88 + gam**2))**0.22

def lambda_of_beta(bet):
    return np.exp((9/2) / (3 + bet**0.75)) / (np.sqrt(1 + bet) + 1)**2

def lambda_acc(M, z, v_rel=0):
    return lambda_of_gamma(gamma(M, z, v_rel)) * lambda_of_beta(beta(M, z, v_rel)) / lambda_iso

def tau(M, z, v_rel=0):
    return 1.5 / (5 + gamma(M, z, v_rel)**(2/3))