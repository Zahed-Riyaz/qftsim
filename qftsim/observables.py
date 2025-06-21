# qftsim/observables.py
import numpy as np
from scipy.fft import fft2, ifft2

class ObservableCalculator:
    def __init__(self):
        self.measurements = []
    
    def two_point_correlation(self, phi, max_distance=None):
        """Calculate two-point correlation function"""
        L = phi.shape[0]
        if max_distance is None:
            max_distance = L // 2
            
        correlations = []
        distances = []
        
        for r in range(max_distance):
            corr_sum = 0
            count = 0
            for i in range(L):
                for j in range(L):
                    i2 = (i + r) % L
                    corr_sum += phi[i, j] * phi[i2, j]
                    count += 1
            correlations.append(corr_sum / count)
            distances.append(r)
            
        return np.array(distances), np.array(correlations)
    
    def susceptibility(self, configurations):
        """Calculate magnetic susceptibility"""
        magnetizations = [np.mean(config) for config in configurations]
        return np.var(magnetizations) * configurations[0].size
    
    def energy_density(self, phi, m, lam):
        """Calculate local energy density"""
        L = phi.shape[0]
        energy = np.zeros_like(phi)
        
        # Potential energy density
        energy += 0.5 * m**2 * phi**2 + lam * phi**4
        
        # Kinetic energy density
        grad_x = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / 2
        grad_y = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / 2
        energy += 0.5 * (grad_x**2 + grad_y**2)
        
        return energy
