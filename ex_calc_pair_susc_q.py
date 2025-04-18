import numpy as np
import pair_susc_q as ps
from multiprocessing import Pool

datapath="SAVEPATH"

# read in the parameters
with open('pair_input.txt', 'r') as file:
    T = float(file.readline().split("=")[1])
    mcx = float(file.readline().split("=")[1])
    mcy = float(file.readline().split("=")[1])
    mcz = float(file.readline().split("=")[1])
    mfx = float(file.readline().split("=")[1])
    mfy = float(file.readline().split("=")[1])
    mfz = float(file.readline().split("=")[1])
    dhyb = float(file.readline().split("=")[1])
    mu = float(file.readline().split("=")[1])
    gamma_x = float(file.readline().split("=")[1])
    gamma_y = float(file.readline().split("=")[1])
    gamma_z = float(file.readline().split("=")[1])
    N_FS = int(file.readline().split("=")[1])

# set up arrays for calculations
# h_list = np.linspace(0.01, 1, 31)
# theta_list = np.linspace(0, np.pi/2, 31)
# phi_list = np.asarray([0, np.pi/4, np.pi/2])

h_list_sparse = np.linspace(0.0, 0.4, 5)
theta_fixed = np.pi/4
phi_fixed = np.pi/4

num_basis = 4

# fixed directions of q vector: along x and along y
qx_list = np.linspace(-2, 2, 25)
qy_list = np.linspace(-2, 2, 25)
qx_fixed = 0
qy_fixed = 0
qz_fixed = 0

for h_fixed in h_list_sparse:
    # calculate the susceptibility as function of qx
    largest_eigval_qx = np.zeros(len(qx_list))
    corresponding_eigvec_qx = np.zeros((len(qx_list), num_basis),dtype=np.complex_)

    def process_qx(args):
        idx, qx = args
        chi_mat = ps.get_susceptibility_matrix(h_fixed, theta_fixed, phi_fixed, dhyb, gamma_x,gamma_y, gamma_z, mcx, mcy, mcz, mfx, mfy,mfz, mu, T, N_FS, qx, qy_fixed, qz_fixed)
        chi_eigvals, chi_eigvecs = np.linalg.eig(chi_mat)
        max_idx = np.argmax(np.real(chi_eigvals))
        return idx, np.max(np.real(chi_eigvals)), chi_eigvecs[:, max_idx]

    with Pool() as pool:
        results = pool.map(process_qx, enumerate(qx_list))
        
    for idx, max_eigval, eigvec in results:
        largest_eigval_qx[idx] = max_eigval
        corresponding_eigvec_qx[idx, :] = eigvec

    fname='chi_qx_h_%.2f_phi_%.2f_theta_%.2f_gammax_%.2f_gammay_%.2f_gammaz_%.2f.txt'%(h_fixed, phi_fixed, theta_fixed, gamma_x, gamma_y, gamma_z)
    with open(datapath+fname, 'w') as file:
        file.write(f"T={T}\n")
        for idx, qx in enumerate(qx_list):
            file.write("%.5f, %.5f, [%.4f+%.4fj, %.4f+%.4fj, %.4f+%.4fj, %.4f+%.4fj]\n"%(
                qx, largest_eigval_qx[idx],
                corresponding_eigvec_qx[idx,0].real, corresponding_eigvec_qx[idx,0].imag,
                corresponding_eigvec_qx[idx,1].real, corresponding_eigvec_qx[idx,1].imag,
                corresponding_eigvec_qx[idx,2].real, corresponding_eigvec_qx[idx,2].imag,
                corresponding_eigvec_qx[idx,3].real, corresponding_eigvec_qx[idx,3].imag))


    # calculate the susceptibility as function of qy
    largest_eigval_qy = np.zeros(len(qy_list))
    corresponding_eigvec_qy = np.zeros((len(qy_list), num_basis),dtype=np.complex_)

    def process_qy(args):
        idx, qy = args
        chi_mat = ps.get_susceptibility_matrix(h_fixed, theta_fixed, phi_fixed, dhyb, gamma_x,gamma_y, gamma_z, mcx, mcy, mcz, mfx, mfy,mfz, mu, T, N_FS, qx_fixed, qy, qz_fixed)
        chi_eigvals, chi_eigvecs = np.linalg.eig(chi_mat)
        max_idx = np.argmax(np.real(chi_eigvals))
        return idx, np.max(np.real(chi_eigvals)), chi_eigvecs[:, max_idx]

    with Pool() as pool:
        results = pool.map(process_qy, enumerate(qy_list))
        
    for idx, max_eigval, eigvec in results:
        largest_eigval_qy[idx] = max_eigval
        corresponding_eigvec_qy[idx, :] = eigvec

    fname='chi_qy_h_%.2f_phi_%.2f_theta_%.2f_gammax_%.2f_gammay_%.2f_gammaz_%.2f.txt'%(h_fixed, phi_fixed, theta_fixed, gamma_x, gamma_y, gamma_z)
    with open(datapath+fname, 'w') as file:
        file.write(f"T={T}\n")
        for idx, qy in enumerate(qy_list):
            file.write("%.5f, %.5f, [%.4f+%.4fj, %.4f+%.4fj, %.4f+%.4fj, %.4f+%.4fj]\n"%(
                qy, largest_eigval_qy[idx],
                corresponding_eigvec_qx[idx,0].real, corresponding_eigvec_qx[idx,0].imag,
                corresponding_eigvec_qy[idx,1].real, corresponding_eigvec_qy[idx,1].imag,
                corresponding_eigvec_qy[idx,2].real, corresponding_eigvec_qy[idx,2].imag,
                corresponding_eigvec_qy[idx,3].real, corresponding_eigvec_qy[idx,3].imag))








