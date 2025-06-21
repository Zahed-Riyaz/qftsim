import numpy as np
import matplotlib.pyplot as plt
from qftsim.simulations import metropolis_simulation
from qftsim.lagrangians import scalar_field_lagrangian
from qftsim.fields import ScalarField

def main():
    # Simulation parameters
    lattice_size = 100         # number of lattice sites
    num_sweeps = 1000          # number of full sweeps over the lattice
    mass = 1.0                 # mass parameter
    coupling = 0.1             # coupling constant
    delta = 0.5                # maximum update change in the Metropolis algorithm
    record_interval = 50       # record configuration every 50 sweeps
    
    print("Starting Metropolis simulation for a 1D scalar field...")
    
    # Run the simulation: configurations and their corresponding action values
    configurations, actions = metropolis_simulation(
        lattice_size, num_sweeps, mass, coupling, delta, record_interval
    )
    
    # Plot the final field configuration
    final_config = configurations[-1]
    plt.figure(figsize=(8, 4))
    plt.plot(final_config, marker='o', linestyle='-', label='Field configuration')
    plt.xlabel('Lattice Site')
    plt.ylabel('Field Value')
    plt.title('Final Field Configuration')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot the history of the action during the simulation
    plt.figure(figsize=(8, 4))
    plt.plot(actions, marker='o', linestyle='-', color='r')
    plt.xlabel('Recorded Sweep (every {} sweeps)'.format(record_interval))
    plt.ylabel('Action')
    plt.title('Action vs. Monte Carlo Sweeps')
    plt.grid(True)
    plt.show()
    
    # Optionally, display the symbolic Lagrangian (for illustration)
    lagrangian = scalar_field_lagrangian(mass, coupling)
    print("Symbolic Lagrangian (for illustration):")
    lagrangian.display()

if __name__ == '__main__':
    main()
