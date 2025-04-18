import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize

k_B = 1

# Define Fermi function
def fermi_dirac(E, T, mu):
    if T == 0:
        answer = 1-np.heaviside(E-mu,0.5)
    else:
        beta = 1 / (k_B * T)
        answer = 1 / (np.exp(beta * (E - mu)) + 1)
    return answer

# Define Gaussian function
def gaussian(x, mu, sigma):
    """Compute Gaussian centered at mu with standard deviation sigma."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

# Function to compute particle density as a function of Fermi energy
def compute_particle_density(mu, energy_range, dos, T):
    """Compute particle density n(E_F) as a function of Fermi energy mu."""
    to_integrate = np.multiply(fermi_dirac(energy_range, T, mu), dos)
    particle_density = np.trapz(to_integrate, energy_range)
    return particle_density

def compute_particle_density_full(energy_range, dos):
    """Compute particle density n(E_F) as a function of Fermi energy E_F."""
    particle_density = np.zeros_like(energy_range)
    for i, ef in enumerate(energy_range):
        # Integrate the DOS from the lowest energy to the Fermi energy ef
        particle_density[i] = np.trapz(dos[energy_range <= ef], energy_range[energy_range <= ef])
    return particle_density


def free_energy_band(Es, dos, T, mu):
    zero_temp_Erho = np.multiply(Es, dos)
    F_free = np.multiply(zero_temp_Erho, 1-np.heaviside(Es-mu, 0.5))

    if T !=0:
        f_E = fermi_dirac(Es, T, mu)
        F_free = np.multiply(zero_temp_Erho, f_E)

    return np.trapz(F_free, Es)

def find_dos(E_values, numbands, energy_range):
    # Flatten all energy values into a single 1D array
    all_energies = np.hstack(E_values)
    # Define the energy range and resolution for DOS
    sigma = 0.015  # Gaussian smearing width (tune as needed)

    # Compute the DOS using Gaussian smearing
    dos = np.zeros_like(energy_range)
    for energy in all_energies:
        dos += gaussian(energy_range, energy, sigma)

    # Normalize the DOS by the number of states
    dos = dos/ (len(all_energies)/numbands)
    return dos
