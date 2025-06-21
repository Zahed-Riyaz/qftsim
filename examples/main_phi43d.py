import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from qftsim.simulate_phi43d import metropolis_simulation_3d, compute_action_3d

def animate_slice(configurations):
    """
    Animate a 2D slice (the middle plane in z) of the 3D configuration.
    """
    fig, ax = plt.subplots()
    L = configurations[0].shape[0]
    mid = L // 2
    im = ax.imshow(configurations[0][:, :, mid],
                   cmap='RdBu_r', origin='lower', vmin=-3, vmax=3)
    ax.set_title("Middle Slice (z = {})".format(mid))
    fig.colorbar(im, ax=ax)
    
    def update(frame):
        im.set_array(frame[:, :, mid])
        ax.set_title("Middle Slice (z = {})".format(mid))
        return [im]
    
    anim = animation.FuncAnimation(fig, update, frames=configurations,
                                   interval=200, blit=True)
    plt.show()

def animate_scatter(configurations):
    """
    Animate a 3D scatter plot of the entire lattice.
    The field values are represented by color.
    """
    from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    L = configurations[0].shape[0]
    # Create a grid of lattice coordinates
    X, Y, Z = np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing='ij')
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    
    sc = ax.scatter(X, Y, Z, c=configurations[0].flatten(),
                    cmap='RdBu_r', vmin=-3, vmax=3)
    ax.set_title("3D Scatter of phi^4 Field")
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, L)
    
    def update(frame):
        ax.cla()  # clear the axes
        sc = ax.scatter(X, Y, Z, c=frame.flatten(),
                        cmap='RdBu_r', vmin=-3, vmax=3)
        ax.set_title("3D Scatter of phi^4 Field")
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_zlim(0, L)
        return sc,
    
    anim = animation.FuncAnimation(fig, update, frames=configurations,
                                   interval=200, blit=False)
    plt.show()

def main():
    # Simulation parameters
    L = 16              # Lattice size: 16 x 16 x 16
    m = 1.0             # Mass parameter
    lam = 0.1           # Coupling constant for phi^4 interaction
    delta = 0.5         # Maximum proposal change
    num_sweeps = 100    # Total number of lattice sweeps (increase for better statistics)
    record_interval = 5 # Record configuration every 5 sweeps
    
    print("Running 3D phi^4 Monte Carlo simulation...")
    
    # Initial configuration: random (Gaussian distributed) field values
    phi0 = np.random.randn(L, L, L)
    
    configurations, actions = metropolis_simulation_3d(phi0, m, lam, delta,
                                                       num_sweeps, record_interval)
    
    print("Final action =", actions[-1])
    
    # Animate a 2D slice (middle plane in z)
    animate_slice(configurations)
    
    # Animate a 3D scatter plot of the full lattice configuration
    animate_scatter(configurations)
    
    # Plot the evolution of the action during the simulation
    plt.figure(figsize=(6, 4))
    plt.plot(actions, marker='o', linestyle='-', color='black')
    plt.xlabel("Recorded Sweep (every {} sweeps)".format(record_interval))
    plt.ylabel("Action")
    plt.title("Action vs. Monte Carlo Sweeps")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
