import numpy as np

def compute_action_3d(phi, m, lam):
    """
    Compute the Euclidean action for a 3D lattice configuration in phi^4 theory.

    The action is defined as:
      S = sum_{i,j,k} [ 0.5*m^2*phi(i,j,k)^2 + lam*phi(i,j,k)^4 ]
          + 0.5 * sum_{i,j,k} { [phi(i,j,k) - phi(i+1,j,k)]^2 +
                                [phi(i,j,k) - phi(i,j+1,k)]^2 +
                                [phi(i,j,k) - phi(i,j,k+1)]^2 }
    with periodic boundary conditions.

    Parameters:
      phi: 3D numpy array of field values (shape LxLxL).
      m: mass parameter (float).
      lam: coupling constant (lambda, float).

    Returns:
      S: total action (float).
    """
    # Potential term: sum over all lattice sites
    potential = np.sum(0.5 * m**2 * phi**2 + lam * phi**4)
    
    # Kinetic term: using forward differences along x, y, and z directions
    kinetic_x = 0.5 * np.sum((phi - np.roll(phi, -1, axis=0))**2)
    kinetic_y = 0.5 * np.sum((phi - np.roll(phi, -1, axis=1))**2)
    kinetic_z = 0.5 * np.sum((phi - np.roll(phi, -1, axis=2))**2)
    
    return potential + kinetic_x + kinetic_y + kinetic_z

def metropolis_update_3d(phi, m, lam, delta):
    """
    Perform one full Metropolis sweep update for the 3D lattice phi^4 theory.
    
    For each lattice site (i, j, k), propose a new value:
      phi_new = phi_old + U(-delta, delta),
    and compute the local change in the action ΔS due to the update. The change
    is computed from the potential at the site and the kinetic contributions of the
    bonds linking the site with its 6 nearest neighbors.
    
    Parameters:
      phi: 3D numpy array (current configuration).
      m: mass parameter (float).
      lam: coupling constant (float).
      delta: maximum magnitude for the proposal (float).
      
    Returns:
      new_phi: updated 3D configuration as a numpy array.
    """
    L = phi.shape[0]
    new_phi = phi.copy()
    
    # Loop over all lattice sites
    for i in range(L):
        for j in range(L):
            for k in range(L):
                old_val = new_phi[i, j, k]
                proposed_val = old_val + np.random.uniform(-delta, delta)
                
                # Potential energy change at site (i,j,k)
                old_pot = 0.5 * m**2 * old_val**2 + lam * old_val**4
                new_pot = 0.5 * m**2 * proposed_val**2 + lam * proposed_val**4
                delta_S = new_pot - old_pot
                
                # Kinetic energy change: contributions from all 6 nearest neighbors.
                # For neighbor in +x direction:
                neighbor = new_phi[(i+1) % L, j, k]
                old_kin = 0.5 * (old_val - neighbor)**2
                new_kin = 0.5 * (proposed_val - neighbor)**2
                delta_S += (new_kin - old_kin)
                
                # For neighbor in -x direction:
                neighbor = new_phi[(i-1) % L, j, k]
                old_kin = 0.5 * (neighbor - old_val)**2
                new_kin = 0.5 * (neighbor - proposed_val)**2
                delta_S += (new_kin - old_kin)
                
                # For neighbor in +y direction:
                neighbor = new_phi[i, (j+1) % L, k]
                old_kin = 0.5 * (old_val - neighbor)**2
                new_kin = 0.5 * (proposed_val - neighbor)**2
                delta_S += (new_kin - old_kin)
                
                # For neighbor in -y direction:
                neighbor = new_phi[i, (j-1) % L, k]
                old_kin = 0.5 * (neighbor - old_val)**2
                new_kin = 0.5 * (neighbor - proposed_val)**2
                delta_S += (new_kin - old_kin)
                
                # For neighbor in +z direction:
                neighbor = new_phi[i, j, (k+1) % L]
                old_kin = 0.5 * (old_val - neighbor)**2
                new_kin = 0.5 * (proposed_val - neighbor)**2
                delta_S += (new_kin - old_kin)
                
                # For neighbor in -z direction:
                neighbor = new_phi[i, j, (k-1) % L]
                old_kin = 0.5 * (neighbor - old_val)**2
                new_kin = 0.5 * (neighbor - proposed_val)**2
                delta_S += (new_kin - old_kin)
                
                # Metropolis acceptance criterion
                if delta_S < 0 or np.random.rand() < np.exp(-delta_S):
                    new_phi[i, j, k] = proposed_val
                    
    return new_phi

def metropolis_simulation_3d(phi0, m, lam, delta, num_sweeps, record_interval=1):
    """
    Run the full Metropolis Monte Carlo simulation for 3D phi^4 theory.
    
    Parameters:
      phi0: initial 3D numpy array of shape (L, L, L).
      m: mass parameter.
      lam: coupling constant.
      delta: maximum proposal update.
      num_sweeps: number of full lattice sweeps.
      record_interval: record the configuration every 'record_interval' sweeps.
      
    Returns:
      configurations: list of recorded 3D numpy array configurations.
      actions: list of action values corresponding to the recorded configurations.
    """
    configurations = []
    actions = []
    
    phi = phi0.copy()
    configurations.append(phi.copy())
    actions.append(compute_action_3d(phi, m, lam))
    
    for sweep in range(1, num_sweeps + 1):
        phi = metropolis_update_3d(phi, m, lam, delta)
        if sweep % record_interval == 0:
            configurations.append(phi.copy())
            actions.append(compute_action_3d(phi, m, lam))
            
    return configurations, actions
