# qftsim/hmc.py
import numpy as np
from .simulate_phi4 import compute_action

def hmc_update(phi, m, lam, epsilon, n_steps):
    """Hybrid Monte Carlo algorithm for better sampling efficiency"""
    phi_old = phi.copy()
    pi = np.random.randn(*phi.shape)  # Conjugate momentum
    
    # Hamiltonian = kinetic + potential
    H_old = 0.5 * np.sum(pi**2) + compute_action(phi_old, m, lam)
    
    # Leapfrog integration
    phi_new = phi_old.copy()
    pi_new = pi - 0.5 * epsilon * compute_force(phi_new, m, lam)
    
    for step in range(n_steps):
        phi_new += epsilon * pi_new
        if step < n_steps - 1:
            pi_new -= epsilon * compute_force(phi_new, m, lam)
    
    pi_new -= 0.5 * epsilon * compute_force(phi_new, m, lam)
    
    H_new = 0.5 * np.sum(pi_new**2) + compute_action(phi_new, m, lam)
    
    # Accept/reject
    if np.random.rand() < np.exp(H_old - H_new):
        return phi_new, True
    return phi_old, False

def compute_force(phi, m, lam):
    """Compute force for HMC dynamics"""
    L = phi.shape[0]
    force = np.zeros_like(phi)
    
    # Potential force
    force -= m**2 * phi + 4 * lam * phi**3
    
    # Kinetic force (lattice derivative)
    force += np.roll(phi, 1, axis=0) + np.roll(phi, -1, axis=0) - 2*phi
    force += np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) - 2*phi
    
    return force
