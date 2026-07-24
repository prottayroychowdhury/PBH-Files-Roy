import numpy as np
from .constants import *

data = np.loadtxt("output_xe.dat")

z_hyrec = data[::-1, 0]
xe_hyrec = data[::-1, 1]
Tb_hyrec = data[::-1, 2]

z_hyrec_max = z_hyrec[-1]
xe_hyrec_max = xe_hyrec[-1]
Tb_hyrec_max = Tb_hyrec[-1]


def x_e(z):
    z = np.asarray(z)
    xe = np.interp(z, z_hyrec, xe_hyrec)
    return np.where(z > z_hyrec_max, xe_hyrec_max, xe)

def x_e_pbh(z):
    return np.minimum(x_e(z), 1.0)

def T_b(z):
    z = np.asarray(z)
    Tb = np.interp(z, z_hyrec, Tb_hyrec)
    Tb_high = Tb_hyrec_max * (1 + z) / (1 + z_hyrec_max)
    return np.where(z > z_hyrec_max, Tb_high, Tb)

def T_cmb(z):
    return T0 * (1 + z)

def rho_cmb(z):
    return rho_crit0 * omega_rad * c**2 * (1 + z)**4

def rho_b(z):
    return rho_crit0 * omega_b * (1 + z)**3

def Hubble(z):
    return H0 * np.sqrt(omega_rad*(1+z)**4 + omega_m*(1+z)**3 + omega_lambda)