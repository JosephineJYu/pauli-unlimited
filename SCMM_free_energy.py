import numpy as np
import numpy.linalg as la
from scipy.optimize import minimize
import numba
from numba import njit, complex128, float64, prange

k_B = 1.0
# Define the Pauli matrices with explicit complex128 type
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)  # Changed from complex
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
identity = np.eye(2, dtype=np.complex128)
projector_c = np.array([[1, 0], [0, 0]], dtype=np.complex128)
projector_f = np.array([[0, 0], [0, 1]], dtype=np.complex128)

num_basis = 4
@njit(fastmath=True)
def safe_heaviside(x, h):
    if x > 0:
        return 1.0
    elif x == 0:
        return h
    else:
        return 0.0

@njit(float64(float64, float64, float64))
def fermi_dirac(E, T, mu):
    if T == 0:
        answer = 1.0 - safe_heaviside(E-mu, 0.5)
    else:
        beta = 1.0 / (k_B * T)
        answer = 1.0 / (np.exp(beta * (E - mu)) + 1.0)
    return answer

@njit(float64(float64))
def safe_log(x):
    if x == 0:
        return -np.inf
    else:
        return np.log(x)

@njit(fastmath=True)
def h0(kx, ky, kz, h, theta, phi, hcvec, hfvec, dhyb, gamma_x, gamma_y, gamma_z,
       mcx, mcy, mcz, mfx, mfy, mfz, mu):
    #hcvec and hfvec are the effective fields due to magnetization
    # basis: c_up, c_down, f_up, f_down
    hfree = np.zeros((4,4), dtype=np.complex128)
    
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
    # for c fermions
    hx = h*np.sin(theta)*np.cos(phi) + hcvec[0]
    hy = h*np.cos(theta) + hcvec[1]
    hz = h*np.sin(theta)*np.sin(phi) + hcvec[2]
    hfree -= hx*np.kron(projector_c, sigma_x) + hy*np.kron(projector_c, sigma_y) + hz*np.kron(projector_c, sigma_z)

    # for f fermions
    hx = h*np.sin(theta)*np.cos(phi) + hfvec[0]
    hy = h*np.cos(theta) + hfvec[1]
    hz = h*np.sin(theta)*np.sin(phi) + hfvec[2]
    hfree -= hx*np.kron(projector_f, sigma_x) + hy*np.kron(projector_f, sigma_y) + hz*np.kron(projector_f, sigma_z)

    # off-diagonal: hybridization
    hfree += -dhyb*np.kron(sigma_x, identity)

    # SOC terms
    hfree += gamma_x * ky*kz * np.kron(sigma_y, sigma_x)
    hfree += gamma_y * kx*kz * np.kron(sigma_y, sigma_y)
    hfree += gamma_z * kx*ky * np.kron(sigma_y, sigma_z)

    return hfree

@njit(fastmath=True)
def hBdG(kx,ky,kz, h, theta, phi, hcvec, hfvec, dhyb, gamma_x,gamma_y, gamma_z, 
         mcx, mcy, mcz, mfx, mfy,mfz, mu, delta_matrix):
    #hcvec and hfvec are the effective fields due to magnetization
    hamiltonian = np.zeros((8,8), dtype=np.complex128)
    hamiltonian[0:4,0:4] = h0(kx,ky,kz, h, theta, phi, hcvec, hfvec, dhyb, gamma_x,gamma_y, gamma_z,
                            mcx, mcy, mcz, mfx, mfy,mfz, mu)
    hamiltonian[4:8,4:8] = -h0(-kx,-ky,-kz, h, theta, phi, hcvec, hfvec, dhyb, gamma_x,gamma_y, gamma_z, 
                            mcx, mcy, mcz, mfx, mfy,mfz, mu)
    
    # pairing part
    hamiltonian[0:4,4:8] = delta_matrix
    hamiltonian[4:8,0:4] = delta_matrix.conj().T
    
    return hamiltonian*0.5

# Move these outside the function and make them constant
M_SINGLET = np.kron(sigma_x, 1j*sigma_y)
M_DX = np.kron(1j*sigma_y, sigma_z)
M_DY = np.kron(1j*sigma_y, identity)
M_DZ = np.kron(1j*sigma_y, sigma_x)
M_LIST = np.array([M_SINGLET, M_DX, M_DY, M_DZ])  # Make it a numpy array

@njit(parallel=True, fastmath=True)
def free_energy_density(variable_list, k_coords, T, mu, h, theta, phi, dhyb, gamma_x,gamma_y, gamma_z, 
                        mcx, mcy, mcz, mfx, mfy,mfz, g, Uc, Uf, dk, cutoff):

    # variable_list has magnitudes of all possible pairing states allowed by the interaction
    # anything which is opposite-species (any spin) is fair game
    # variable_list = [singlet, dx_r, dx_i, dy_r, dy_i, dz_r, dz_i, M_cx, M_cy, M_cz, M_fx, M_fy, M_fz]
    # cutoff applies only to SC part
    delta_list = variable_list[:7]
    psi = delta_list[0]
    dx = np.complex128(delta_list[1] + 1j*delta_list[2])
    dy = np.complex128(delta_list[3] + 1j*delta_list[4])
    dz = np.complex128(delta_list[5] + 1j*delta_list[6])
    coeff_array = np.array([psi, dx, dy, dz], dtype=np.complex128)
    
    M_coeff = variable_list[7:]
    M_cx = M_coeff[0]
    M_cy = M_coeff[1]
    M_cz = M_coeff[2]
    M_fx = M_coeff[3]
    M_fy = M_coeff[4]
    M_fz = M_coeff[5]

    # define species-dependent fields
    hcvec = np.array([M_cx, M_cy, M_cz], dtype=np.float64)*Uc/2
    hfvec = np.array([M_fx, M_fy, M_fz], dtype=np.float64)*Uf/2
    
    # free energy contribs should all be real numbers
    f_quad =  0 # Delta^2 contribution + M^2 contribution
    f_band = 0 # condensation energy contribution
    f_partition = 0 # Band U - TS contribution
    
    delta_matrix = np.zeros((4,4), dtype=np.complex128)
    for n in range(num_basis):
        delta_matrix += coeff_array[n]*M_LIST[n]
        # g = tilde{V} in notes
        f_quad += float(np.abs(coeff_array[n])**2)/(2.0*g)

    nc, nf = 0, 0 

    # Initialize reduction arrays with the correct types
    f_band_arr = np.zeros(len(k_coords), dtype=np.float64)  # Explicitly float64
    f_partition_arr = np.zeros(len(k_coords), dtype=np.float64)
    nc_arr = np.zeros(len(k_coords), dtype=np.float64)
    nf_arr = np.zeros(len(k_coords), dtype=np.float64)

    for idx in prange(len(k_coords)):
        k_set = k_coords[idx]
        kxval = k_set[0]
        kyval = k_set[1]
        kzval = k_set[2]

        # spectrum in absence of gap
        hk0 = h0(kxval,kyval,kzval, h, theta, phi, hcvec, hfvec, dhyb, gamma_x,gamma_y, gamma_z, 
                 mcx, mcy, mcz, mfx, mfy,mfz, mu)
        Ediag = np.zeros(4, dtype=np.float64)
        for i in range(4):
            Ediag[i] = np.real(hk0[i,i])
        E_nogap, vs_0 = la.eigh(hk0) # gives 4 eigvals

        # get the spectrum with Delta potentially nonzero   
        hk_ingap = hBdG(kxval,kyval,kzval, h, theta, phi, hcvec, hfvec, dhyb, gamma_x,gamma_y, gamma_z, 
                        mcx, mcy, mcz, mfx, mfy,mfz, mu, delta_matrix)
        Es_ingap, vs_ingap = la.eigh(hk_ingap)
        
        # only consider the positive energy qptcles
        all_options = np.arange(4,8)
        used = np.zeros(4, dtype=np.bool_)  # Track used indices

        # Store local results in arrays instead of accumulating
        f_band_local = 0.0
        f_partition_local = 0.0
        nc_local = 0.0
        nf_local = 0.0

        for s in range(4):
            init_energy = E_nogap[s] 
            if np.abs(init_energy) <= cutoff:
                # Find minimum among unused indices
                min_val = 1e10  
                min_idx = 0
                for i in range(4):
                    if not used[i]:
                        val = np.abs(0.5*np.abs(init_energy)-Es_ingap[all_options[i]])
                        if val < min_val:
                            min_val = val
                            min_idx = i
                
                Eqp = Es_ingap[all_options[min_idx]]
                used[min_idx] = True
            else:
                Eqp = 0.5*np.abs(init_energy)

            f_band_local += -Eqp
            f_band_local += 0.5*Ediag[s]

            fd = fermi_dirac(2*Eqp, T, 0)
            f_partition_local += 2*Eqp*fd
            
            if T!=0:
                f_partition_local += k_B*T*(fd*safe_log(fd) + (1-fd)*safe_log(1-fd))

        # Calculate local nc, nf
        hknomag = h0(kxval,kyval,kzval, 0, theta, phi, np.array([0,0,0]), np.array([0,0,0]), 
                     dhyb, gamma_x,gamma_y, gamma_z, mcx, mcy, mcz, mfx, mfy,mfz, mu)
        zero_field_band_energy, vs_nomag = la.eigh(hknomag)

        n1 = fermi_dirac(zero_field_band_energy[0], T, 0)
        n2 = fermi_dirac(zero_field_band_energy[1], T, 0)
        n3 = fermi_dirac(zero_field_band_energy[2], T, 0)
        n4 = fermi_dirac(zero_field_band_energy[3], T, 0)
        nvec = np.array([n1, n2, n3, n4], dtype=np.float64)

        to_up1 =  np.real(np.multiply(vs_nomag.conj()[0,:],  vs_nomag.T[:,0]))
        to_down1 = np.real(np.multiply(vs_nomag.conj()[1,:],  vs_nomag.T[:,1]))
        to_up2 = np.real(np.multiply(vs_nomag.conj()[2,:],  vs_nomag.T[:,2]))
        to_down2 = np.real(np.multiply(vs_nomag.conj()[3,:],  vs_nomag.T[:,3]))
        # eigvec_mat_prod = eigvec_matrix.conj() @ eigvec_matrix.T 
        # eigvec_mat_prod = np.real(eigvec_mat_prod)

        nleg1_up = np.dot(to_up1, nvec)
        nleg1_down = np.dot(to_down1, nvec)
        nleg2_up = np.dot(to_up2, nvec)
        nleg2_down = np.dot(to_down2, nvec)

        nc_local = nleg1_up + nleg1_down
        nf_local = nleg2_up + nleg2_down

        # Store local results - use np.real() to ensure real numbers
        f_band_arr[idx] = float(np.real(f_band_local))  # Convert to real explicitly
        f_partition_arr[idx] = float(np.real(f_partition_local))
        nc_arr[idx] = float(np.real(nc_local))
        nf_arr[idx] = float(np.real(nf_local))

    # Sum up all contributions after the parallel loop
    f_band = np.sum(f_band_arr)*(dk/(2*np.pi))**3
    f_partition = np.sum(f_partition_arr)*(dk/(2*np.pi))**3
    nc = np.sum(nc_arr)/len(k_coords)
    nf = np.sum(nf_arr)/len(k_coords)

    # Calculate final free energy
    Mc2 = M_cx**2 + M_cy**2 + M_cz**2
    Mf2 = M_fx**2 + M_fy**2 + M_fz**2
    f_quad += (Uc/4)*(Mc2 + nc**2) + (Uf/4)*(Mf2 + nf**2)

    return np.real(f_partition + f_band + f_quad)

def minimize_free_energy(k_coords, T, mu, h, theta, phi, dhyb, gamma_x,gamma_y, gamma_z, 
                        mcx, mcy, mcz, mfx, mfy,mfz, g, Uc, Uf, dk, cutoff):
    initial_guess = 0.001*np.ones(13)
    args = (k_coords, T, mu, h, theta, phi, dhyb, gamma_x,gamma_y, gamma_z, 
                        mcx, mcy, mcz, mfx, mfy,mfz, g, Uc, Uf, dk, cutoff)
    
    iteration = [0]  # Use list to allow modification in closure
    def callback(x):
        f = free_energy_density(x, *args)
        print(f"Iteration {iteration[0]}:", flush=True)
        print(f"Current x: {x}", flush=True)
        print(f"Current free energy: {f}", flush=True)
        print(f"Magnetic field h: {h}\n", flush=True)
        iteration[0] += 1
        
    result = minimize(free_energy_density, x0=initial_guess, args=args, method='BFGS', 
                     callback=callback, options={'maxiter': 120})

    if result.success:
        print(f"Optimal Delta: {result.x}")
        print(f"Minimum Free Energy: {result.fun}")
    else:
        error_message = f"""
        Optimization failed at:
        T = {T}
        B = {h}
        mu = {mu}
        v = {g}
        Uf = {Uf}
        Error message: {result.message}
        """
        print("Optimization failed:", result.message)
        
        # Write to error log file
        with open('optimization_errors.log', 'a') as f:
            f.write(error_message + '\n' + '-'*50 + '\n')
        print("Optimization failed:", result.message)
    return result.x
