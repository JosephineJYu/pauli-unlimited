import numpy as np
from numba import njit

# Define the Pauli matrices 
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli X
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)  # Pauli Y
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)  # Pauli Z
identity = np.eye(2, dtype=complex)  # 2x2 Identity matrix
k_B = 1

# Define Fermi function
def fermi_dirac(E, T, mu):
    if T == 0:
        answer = 1-np.heaviside(E-mu,0.5)
    else:
        beta = 1 / (k_B * T)
        answer = 1 / (np.exp(beta * (E - mu)) + 1)
    return answer

def get_M_list():
    # define the 4 gap structures
    # these are the things that describe the structure of gap in the SPECIES basis
    # normalized so tr(M^dagger*M) = 1
    M_singlet = np.kron(sigma_x, 1j*sigma_y)/2 #M1
    M_dx = np.kron(1j*sigma_y, sigma_z)/2 #M2 
    M_dy = np.kron(1j*sigma_y, identity)/2 #M3
    M_dz = np.kron(1j*sigma_y, sigma_x)/2 #M4

    M_list = [M_singlet, M_dx, M_dy, M_dz]
    return M_list

@njit
def get_susceptibility_element(h, theta,phi, dhyb, gamma_x,gamma_y, gamma_z, mcx, mcy, mcz, mfx, mfy,mfz, mu, M_i, M_j, T, N_FS, qx, qy, qz=0):
    tolerance = 3e-2

    def hfree_fct(kx,ky,kz, h, theta, phi, dhyb, gamma_x,gamma_y, gamma_z, mcx, mcy, mcz, mfx, mfy,mfz, mu):
        # basis: c_up, c_down, f_up, f_down
        hfree = np.zeros((4,4), dtype=np.complex_)
        
        # diagonal part
        mu_f = 0.5 # fixed mu_f
        # field h is species-independent because no MFT on magnetization
        hfree[0,0] = kx**2/(2*mcx) + ky**2/(2*mcy)  + kz**2/(2*mcz) - mu
        hfree[1,1] = kx**2/(2*mcx) + ky**2/(2*mcy) + kz**2/(2*mcz) - mu
        hfree[2,2] = kx**2/(2*mfx) + ky**2/(2*mfy) + kz**2/(2*mfz) - mu_f
        hfree[3,3] = kx**2/(2*mfx) + ky**2/(2*mfy) + kz**2/(2*mfz) - mu_f
        

        # field in any orientation
        # phi = angle from a to c [phi = 0 means in ab plane]
        # theta = azimuthal from b [theta = 0 means along b axis]
        hy = h*np.cos(theta)
        hx = h*np.sin(theta)*np.cos(phi)
        hz = h*np.sin(theta)*np.sin(phi)
        if np.abs(hx) > 1e-14:
            hfree -= hx*np.kron(identity, sigma_x)
        if np.abs(hy) > 1e-14:
            hfree -= hy*np.kron(identity, sigma_y)
        if np.abs(hz) > 1e-14:
            hfree -= hz*np.kron(identity, sigma_z)      

        # off-diagonal: hybridization
        hfree += -dhyb*np.kron(sigma_x, identity)
              # SOC terms
        hfree += gamma_x * ky*kz * np.kron(sigma_y, sigma_x)
        hfree += gamma_y * kx*kz * np.kron(sigma_y, sigma_y)
        hfree += gamma_z * kx*ky * np.kron(sigma_y, sigma_z)

        return hfree    
    
    def fermi_dirac(E, T, mu):
        if T == 0:
            answer = 1.0 if E <= mu else 0.0
        else:
            beta = 1 / (k_B * T)
            answer = 1 / (np.exp(beta * (E - mu)) + 1)
        return answer
    
    klist = np.linspace(-4, 4, N_FS)
    kzlist = np.linspace(-1,1,int(N_FS/4))
    chi_contribution = 0
    # construct H 

    num_pts = 0
    for kx in klist:
        for ky in klist:
            for kz in kzlist:

                Hfree = hfree_fct(kx,ky, kz, h, theta, phi, dhyb, gamma_x,gamma_y, gamma_z, mcx, mcy, mcz, mfx, mfy,mfz, mu)
                # diagonalize H to get spectrum
                eigvals, eigvecs = np.linalg.eigh(Hfree)
                Uk = np.copy(eigvecs)

                Hfree_kq = hfree_fct(-kx+qx,-ky+qy, -kz+qz, h, theta, phi, dhyb, gamma_x,gamma_y, gamma_z, mcx, mcy, mcz, mfx, mfy,mfz, mu)
                # diagonalize H to get spectrum
                eigvals_kq, eigvecs_kq = np.linalg.eigh(Hfree_kq)
                U_kq = np.copy(eigvecs_kq)

                for n, En_k in enumerate(eigvals):
                    if np.abs(En_k) > tolerance:
                        continue
                    else:
                        for m, Em_kq in enumerate(eigvals_kq):
                            if np.abs(Em_kq) > tolerance:
                                continue
                            else:
                                fdn = fermi_dirac(En_k, T, 0)
                                fdm = fermi_dirac(-Em_kq, T, 0)
                           
                                energy_sum = Em_kq + En_k
                                factor = (fdn-fdm)/energy_sum

                                # transform M_i, M_j into band basis 
                                # What we find by carefully deriving susceptibility.
                                Mj_dagger_band = np.conjugate(Uk).T @ np.conjugate(M_j).T @  np.conjugate(U_kq) 
                                Mi_band = U_kq.T @ M_i @ Uk
                                Mi_T_band = U_kq.T @ M_i.T @ Uk
                                
                                first_term = Mj_dagger_band[n,m]*Mi_band[m,n]
                                second_term = Mj_dagger_band[n,m]*Mi_T_band[m,n]                                
                                chi_contribution += (factor * (first_term - second_term))

                                num_pts += 1

    return chi_contribution/num_pts


def get_susceptibility_matrix(h, theta, phi, dhyb, gamma_x,gamma_y, gamma_z, mcx, mcy, mcz, mfx, mfy,mfz, mu, T, N_FS, qx, qy, qz=0):
    gap_structures = get_M_list()
    chi_matrix = np.zeros((len(gap_structures), len(gap_structures)),dtype=np.complex_) 
    for i in range(len(gap_structures)):
        for j in range(len(gap_structures)):
            M_i = gap_structures[i]
            M_j = gap_structures[j]
            chi_ij = -1*get_susceptibility_element(h, theta, phi, dhyb, gamma_x,gamma_y, gamma_z, mcx, mcy, mcz, mfx, mfy,mfz, mu, M_i, M_j, T, N_FS, qx, qy, qz)
            chi_matrix[i,j] = chi_ij
    return chi_matrix
