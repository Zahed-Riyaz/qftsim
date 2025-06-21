# main_phi4.py
import numpy as np
import matplotlib.pyplot as plt
import qftsim.simulate_phi4
import metropolis_simulation, compute_action

def main():
    # Simulation parameters
    L = 32            # Lattice size: 32 x 32
    m = 1.0           # Mass parameter
    lam = 0.1         # Coupling constant for phi^4 interaction
    delta = 0.5       # Maximum change for the Metropolis proposal
    num_sweeps = 1000 # Number of full lattice sweeps
    record_interval = 10  # Record configuration every 10 sweeps
    
    # Initial configuration: random (Gaussian distributed) field values
    phi0 = np.random.randn(L, L)
    print("Starting phi^4 simulation on a 2D lattice...")
    
    # Run the simulation
    configurations, actions = metropolis_simulation(phi0, m, lam, delta, num_sweeps, record_interval)
    
    # Plot the final field configuration as a heatmap
    final_phi = configurations[-1]
    plt.figure(figsize=(6, 5))
    plt.imshow(final_phi, cmap='RdBu_r', origin='lower')
    plt.title("Final Field Configuration in phi^4 Theory")
    plt.colorbar(label=r'Field value $\phi$')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    # Plot the history of the action
    plt.figure(figsize=(6, 4))
    plt.plot(actions, marker='o', linestyle='-', color='black')
    plt.xlabel("Recorded Sweep (every {} sweeps)".format(record_interval))
    plt.ylabel("Action")
    plt.title("Action vs. Monte Carlo Sweeps")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
