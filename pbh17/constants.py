import numpy as np

G, c, k = 6.67e-11, 3.00e8, 1.38e-23
pi = np.pi
H0 = 67.4 * 1000 / 3.09e22
rho_crit0 = 3 * H0**2 / (8 * pi * G)
omega_b, omega_rad, omega_m = 0.049, 0.91e-4, 0.315
omega_lambda = 1 - omega_m - omega_rad
m_p, m_e, M_sun = 1.67e-27, 9.11e-31, 1.99e30
T0 = 2.725
sigma_T = 6.65e-29
alpha = 1 / 137
lambda_ad = 0.25 * (3 / 5)**1.5
lambda_iso = 0.25 * np.exp(1.5)
alpha_B_ion, q = 1.8e-19, 0.86