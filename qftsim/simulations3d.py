# simulations3d.py
import numpy as np

def compute_action_3d(configuration, mass, coupling):
    """
    Compute the full Euclidean action for a 3D lattice configuration.

    The action is defined as:
      S = sum_{i,j,k} [ 0.5*m^2*phi^2 + coupling*phi^4 ]
          + 0.5 * sum_{<neighbors>} (phi_i - phi_j)^2
    where the sum over neighbors is taken only once (here we sum only
    over the positive directions to avoid double counting).
    
    configuration: 3D numpy array of shape (L,L,L).
    mass: mass parameter (float).
    coupling: coupling constant (float).
    
    Returns:
      S: float, the total action.
    """
    L = configuration.shape[0]
    S = 0.0
    for i in range(L):
        for j in range(L):
            for k in range(L):
                phi = configuration[i, j, k]
                S += 0.5 * mass**2 * phi**2 + coupling * phi**4
                # kinetic term: count only in +x, +y, and +z directions
                for di, dj, dk in [(1,0,0), (0,1,0), (0,0,1)]:
                    ip = (i + di) % L
                    jp = (j + dj) % L
                    kp = (k + dk) % L
                    S += 0.5 * (phi - configuration[ip, jp, kp])**2
    return S

def metropolis_update_3d(configuration, mass, coupling, delta):
    """
    Perform a single Metropolis sweep update for the 3D lattice.
    
    For each lattice site at (i,j,k), propose a change 
      phi_new = phi_old + random_uniform(-delta, delta),
    and accept or reject the update with the probability
      min(1, exp(-ΔS)),
    where ΔS is computed using the contributions from the potential
    term and the six nearest-neighbor kinetic links.
    
    configuration: 3D numpy array of shape (L,L,L).
    mass: mass parameter (float).
    coupling: coupling constant (float).
    delta: maximum change magnitude (float).
    
    Returns:
      new_config: updated configuration as a numpy array.
    """
    L = configuration.shape[0]
    new_config = configuration.copy()
    for i in range(L):
        for j in range(L):
            for k in range(L):
                old_val = new_config[i, j, k]
                new_val = old_val + np.random.uniform(-delta, delta)
                # Compute change in potential at (i,j,k)
                delta_S = 0.5 * mass**2 * (new_val**2 - old_val**2) \
                          + coupling * (new_val**4 - old_val**4)
                # Compute change in kinetic contributions from all 6 neighbors.
                for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                    ip = (i + di) % L
                    jp = (j + dj) % L
                    kp = (k + dk) % L
                    neighbor_val = new_config[ip, jp, kp]
                    delta_S += 0.5 * ( (new_val - neighbor_val)**2 - (old_val - neighbor_val)**2 )
                if delta_S < 0 or np.random.rand() < np.exp(-delta_S):
                    new_config[i, j, k] = new_val
    return new_config

def metropolis_simulation_3d(lattice_size, num_sweeps, mass, coupling, delta, record_interval=1):
    """
    Run a Metropolis Monte Carlo simulation on a 3D lattice.
    
    lattice_size: number of lattice sites in each direction (int).
    num_sweeps: number of full sweeps over the lattice (int).
    mass: mass parameter (float).
    coupling: coupling constant (float).
    delta: maximum update change (float).
    record_interval: record configuration every this many sweeps.
    
    Returns:
      configurations: list of 3D numpy arrays (field configurations).
      actions: list of action values corresponding to the recorded configurations.
    """
    configuration = np.random.randn(lattice_size, lattice_size, lattice_size)
    configurations = []
    actions = []
    
    configurations.append(configuration.copy())
    actions.append(compute_action_3d(configuration, mass, coupling))
    
    for sweep in range(1, num_sweeps + 1):
        configuration = metropolis_update_3d(configuration, mass, coupling, delta)
        if sweep % record_interval == 0:
            configurations.append(configuration.copy())
            actions.append(compute_action_3d(configuration, mass, coupling))
    
    return configurations, actions
