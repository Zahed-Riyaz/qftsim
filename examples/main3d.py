import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from qftsim.simulations3d import metropolis_simulation_3d

def animate_slice(configurations):
    """
    Animate the middle slice (in z) of the 3D configuration using imshow.
    """
    fig, ax = plt.subplots()
    L = configurations[0].shape[0]
    mid = L // 2
    # Set vmin/vmax to fix the color scale (adjust as needed)
    im = ax.imshow(configurations[0][:, :, mid], cmap='viridis', origin='lower', vmin=-3, vmax=3)
    ax.set_title("Middle Slice of 3D Field")
    fig.colorbar(im, ax=ax)
    
    def update(frame):
        im.set_array(frame[:, :, mid])
        ax.set_title("Middle Slice of 3D Field")
        return [im]
    
    anim = animation.FuncAnimation(fig, update, frames=configurations, interval=200, blit=True)
    plt.show()

def animate_scatter(configurations):
    """
    Animate a 3D scatter plot of the full 3D field configuration.
    (For clarity the lattice is kept small; adjust lattice_size if desired.)
    """
    from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    L = configurations[0].shape[0]
    # Create grid coordinates
    X, Y, Z = np.meshgrid(np.arange(L), np.arange(L), np.arange(L), indexing='ij')
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    
    # Initial scatter plot
    sc = ax.scatter(X, Y, Z, c=configurations[0].flatten(), cmap='viridis', vmin=-3, vmax=3)
    ax.set_title("3D Scatter of Field Configuration")
    ax.set_xlim(0, L)
    ax.set_ylim(0, L)
    ax.set_zlim(0, L)
    
    def update(frame):
        ax.cla()  # clear the axes
        sc = ax.scatter(X, Y, Z, c=frame.flatten(), cmap='viridis', vmin=-3, vmax=3)
        ax.set_title("3D Scatter of Field Configuration")
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_zlim(0, L)
        return sc,
    
    anim = animation.FuncAnimation(fig, update, frames=configurations, interval=200, blit=False)
    plt.show()

def main():
    # Simulation parameters
    lattice_size = 10    # 10x10x10 lattice (adjust as needed)
    num_sweeps = 50      # total sweeps over the lattice
    mass = 1.0           # mass parameter
    coupling = 0.1       # coupling constant
    delta = 0.5          # maximum update change in the Metropolis algorithm
    record_interval = 1  # record configuration every sweep
    
    print("Running 3D Monte Carlo simulation...")
    configurations, actions = metropolis_simulation_3d(lattice_size, num_sweeps, mass, coupling, delta, record_interval)
    
    print("Animating a 2D slice (middle z-plane) of the 3D field configuration...")
    animate_slice(configurations)
    
    print("Animating a 3D scatter plot of the field configuration...")
    animate_scatter(configurations)

if __name__ == '__main__':
    main()
