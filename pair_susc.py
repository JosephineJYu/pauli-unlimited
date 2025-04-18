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
def get_susceptibility_element(h, theta,phi, dhyb, gamma_x,gamma_y, gamma_z, mcx, mcy, mcz, mfx, mfy,mfz, mu, M_i, M_j, T, N_FS):
    tolerance = 1e-1

    def hfree(kx,ky,kz, h, theta, phi, dhyb, gamma_x,gamma_y, gamma_z, mcx, mcy, mcz, mfx, mfy,mfz, mu):
        # basis: c_up, c_down, f_up, f_down
        hfree = np.zeros((4,4), dtype=np.complex_)
        
        # diagonal part
        mu_f = 1 # fixed mu_f
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
        hfree -= hx*np.kron(identity, sigma_x) + hy*np.kron(identity, sigma_y) + hz*np.kron(identity, sigma_z)

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
    
    klist = np.linspace(-6, 6, N_FS)
    kzlist = np.linspace(-6,6,N_FS)
    chi_contribution = 0
    # construct H 
    num_pts = 0
    for kx in klist:
        for ky in klist:
            for kz in kzlist:
                Hfree = hfree(kx,ky, kz, h, theta, phi, dhyb, gamma_x,gamma_y, gamma_z, mcx, mcy, mcz, mfx, mfy,mfz, mu)
                Hfree.real[np.abs(Hfree.real) < 1e-13] = 0  # Zero out small real parts
                Hfree.imag[np.abs(Hfree.imag) < 1e-13] = 0  # Zero out small imaginary parts
                # diagonalize H to get spectrum
                eigvals, eigvecs = np.linalg.eigh(Hfree)
                U = eigvecs

                for n, En in enumerate(eigvals):
                    if np.abs(En) > tolerance:
                        continue
                    else:
                        num_pts += 1
                        for m, Em in enumerate(eigvals):
                            if np.abs(Em) > tolerance:
                                continue
                            else:
                                fdn = fermi_dirac(En, T, 0)
                                fdm = fermi_dirac(Em, T, 0)
                                # TODO: how to deal with degeneracy?
                                energy_diff = Em - En
                                if np.abs(energy_diff)<1e-10:
                                    factor = -1/(k_B*T)*fdn*fermi_dirac(-En, T, 0)# derivative of FD distribution
                                else:
                                    factor = (fdm-fdn)/energy_diff
                                # matrix_elements
                                # transform M_i, M_j into band basis TODO: did I do this correctly? 
                                Mi_band = U.T @ M_i @ U
                                Mj_band = np.conjugate(U).T @ np.conjugate(M_j).T @ np.conjugate(U)
                                Mi_mn = Mi_band[m,n]
                                Mj_nm = Mj_band[n,m]
    
                                chi_contribution += (factor * Mi_mn * Mj_nm)
    return chi_contribution/num_pts


def get_susceptibility_matrix(h, theta, phi, dhyb, gamma_x,gamma_y, gamma_z, mcx, mcy, mcz, mfx, mfy,mfz, mu, T, N_FS):
    gap_structures = get_M_list()
    chi_matrix = np.zeros((len(gap_structures), len(gap_structures)),dtype=np.complex_) 
    for i in range(len(gap_structures)):
        for j in range(len(gap_structures)):
            M_i = gap_structures[i]
            M_j = gap_structures[j]
            chi_ij = -(k_B*T)*get_susceptibility_element(h, theta, phi, dhyb, gamma_x,gamma_y, gamma_z, mcx, mcy, mcz, mfx, mfy,mfz, mu, M_i, M_j, T, N_FS)
            chi_matrix[i,j] = chi_ij
    return chi_matrix
