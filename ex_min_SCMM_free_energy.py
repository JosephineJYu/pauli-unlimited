# executable file for minimizing full free energy 
# no magnetization, just attractive interaction, opposite species
import numpy as np
import os
import scipy.linalg as la
from scipy.optimize import minimize
import SCMM_free_energy as SCMM
from multiprocessing import Pool


# Define parameters
k_B = 1
# inputname = 'free_energy_input.txt'
inputname = 'input_single_field.txt'
#fullpath = os.path.join(file_path, inputname)

#with open(fullpath, 'r') as f:
with open(inputname, 'r') as f:
    # TODO update this
    num_k_pts = int(f.readline().split('=')[1])
    mcx = float(f.readline().split("=")[1])
    mcy = float(f.readline().split("=")[1])
    mcz = float(f.readline().split("=")[1])
    mfx = float(f.readline().split("=")[1])
    mfy = float(f.readline().split("=")[1])
    mfz = float(f.readline().split("=")[1])
    dhyb = float(f.readline().split("=")[1])
    mu = float(f.readline().split("=")[1])
    gamma_x = float(f.readline().split("=")[1])
    gamma_y = float(f.readline().split("=")[1])
    gamma_z = float(f.readline().split("=")[1])
    hmin = float(f.readline().split('=')[1])
    hmax = float(f.readline().split('=')[1])
    num_h = int(f.readline().split('=')[1])
    hind = int(f.readline().split('=')[1])
    g = float(f.readline().split('=')[1])
    Uc = float(f.readline().split('=')[1])
    Uf = float(f.readline().split('=')[1])
    tolerance = float(f.readline().split('=')[1]) 
    T = float(f.readline().split('=')[1])

print("everything loaded")
# get relevant k points
kx_vals = np.linspace(-4, 4, num_k_pts)
ky_vals = np.linspace(-4, 4, num_k_pts)
kz_vals = np.linspace(-2, 2, int(num_k_pts/2))
KX, KY, KZ = np.meshgrid(kx_vals, ky_vals, kz_vals)
dk = (kx_vals[1]-kx_vals[0])

kx_flat = KX.flatten()
ky_flat = KY.flatten()
kz_flat = KZ.flatten()
k_pairs = np.column_stack((kx_flat, ky_flat, kz_flat))

all_h_values = np.linspace(hmin, hmax, num_h)
h = all_h_values[hind]
print("h = ", h)
theta_fixed = np.pi*0.2
phi_fixed = np.pi/2
# minimize the free energy
result = SCMM.minimize_free_energy(k_pairs, T, mu, h, theta_fixed, phi_fixed, dhyb, gamma_x,gamma_y, gamma_z, 
                        mcx, mcy, mcz, mfx, mfy,mfz, g, Uc, Uf, dk, tolerance)
# process result 
psi, dx, dy, dz = 0+1j*0, 0+1j*0, 0+1j*0, 0+1j*0 
psi = result[0]
dx = result[1] + 1j*result[2]
dy = result[3] + 1j*result[4]
dz = result[5] + 1j*result[6]
M_cx = result[7]
M_cy = result[8]
M_cz = result[9]
M_fx = result[10]
M_fy = result[11]
M_fz = result[12]

# Save the gap parameters to output file
save_path = "SAVEFILEPATH"
# flag whether there is SOC
SOC_flag = 0
if gamma_x != 0 or gamma_y != 0 or gamma_z != 0:
    SOC_flag = 1

datafile_name = f'min_sol_g{g:.3f}_Uc{Uc:.3f}_Uf{Uf:.3f}_h{h:.2f}_T{T:.4f}_mu{mu:.2f}_phi{phi_fixed/np.pi:.2f}pi_theta{theta_fixed/np.pi:.2f}pi_mfx{mfx:.1f}_SOC{SOC_flag}.txt' 
output_filename = os.path.join(save_path, datafile_name)
with open(output_filename, 'w') as f:
    f.write(f'psi = {psi.real:.8f} + {psi.imag:.8f}j\n')
    f.write(f'dx = {dx.real:.8f} + {dx.imag:.8f}j\n') 
    f.write(f'dy = {dy.real:.8f} + {dy.imag:.8f}j\n')
    f.write(f'dz = {dz.real:.8f} + {dz.imag:.8f}j\n')
    f.write(f'Mc = {M_cx:.6f}, {M_cy:.6f}, {M_cz:.6f}\n')
    f.write(f'Mf = {M_fx:.6f}, {M_fy:.6f}, {M_fz:.6f}\n')
    f.write(f'mcx = {mcx:.6f}\n')
    f.write(f'mcy = {mcy:.6f}\n')
    f.write(f'mcz = {mcz:.6f}\n')
    f.write(f'mfx = {mfx:.6f}\n')
    f.write(f'mfy = {mfy:.6f}\n')
    f.write(f'mfz = {mfz:.6f}\n')
    f.write(f'dhyb = {dhyb:.6f}\n')
    f.write(f'gamma_x = {gamma_x:.6f}\n')
    f.write(f'gamma_y = {gamma_y:.6f}\n')
    f.write(f'gamma_z = {gamma_z:.6f}\n')
    f.write(f'num_k_pts = {num_k_pts}\n')
    f.write(f'tolerance = {tolerance:.6f}\n')

print("done saving h = ", h)
