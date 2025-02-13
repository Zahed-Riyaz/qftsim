import numpy as np

def compute_action(phi, m, lam):
    """
    Compute the Euclidean action for a 2D lattice configuration of phi^4 theory.
    
    Parameters:
      phi: 2D numpy array of field values on the lattice.
      m: mass parameter (float).
      lam: coupling constant (lambda, float).
      
    Returns:
      S: the total action (float).
    """
    L = phi.shape[0]
    # Potential term: sum_x [ 0.5*m^2*phi^2 + lam*phi^4 ]
    potential = np.sum(0.5 * m**2 * phi**2 + lam * phi**4)
    
    # Kinetic term: sum_x [0.5*(phi(x) - phi(x+e))^2] summed over positive directions to avoid double counting.
    # Here we take differences in the x-direction and y-direction.
    kinetic_x = 0.5 * np.sum((phi - np.roll(phi, -1, axis=0))**2)
    kinetic_y = 0.5 * np.sum((phi - np.roll(phi, -1, axis=1))**2)
    
    return potential + kinetic_x + kinetic_y

def metropolis_update(phi, m, lam, delta):
    """
    Perform one full Metropolis sweep over the 2D lattice for phi^4 theory.
    
    At each lattice site, a new value is proposed by adding a random number from a uniform distribution
    in the range [-delta, delta]. The local change in the action is computed and the move is accepted
    with probability min(1, exp(-ΔS)).
    
    Parameters:
      phi: 2D numpy array of current field values.
      m: mass parameter (float).
      lam: coupling constant (lambda, float).
      delta: maximum magnitude of change for the proposal.
      
    Returns:
      new_phi: updated 2D numpy array of field values.
    """
    L = phi.shape[0]
    new_phi = phi.copy()
    
    # Loop over lattice sites
    for i in range(L):
        for j in range(L):
            old_val = new_phi[i, j]
            proposed_val = old_val + np.random.uniform(-delta, delta)
            
            # Compute the potential contribution at site (i,j)
            old_pot = 0.5 * m**2 * old_val**2 + lam * old_val**4
            new_pot = 0.5 * m**2 * proposed_val**2 + lam * proposed_val**4
            
            # Kinetic contribution: only bonds attached to site (i,j) change.
            # Neighbors (with periodic boundary conditions):
            left  = new_phi[i, (j-1) % L]
            right = new_phi[i, (j+1) % L]
            up    = new_phi[(i-1) % L, j]
            down  = new_phi[(i+1) % L, j]
            
            old_kinetic = 0.5 * ( (old_val - left)**2 + (old_val - right)**2 +
                                  (old_val - up)**2 + (old_val - down)**2 )
            new_kinetic = 0.5 * ( (proposed_val - left)**2 + (proposed_val - right)**2 +
                                  (proposed_val - up)**2 + (proposed_val - down)**2 )
            
            # Local action change
            delta_S = (new_pot + new_kinetic) - (old_pot + old_kinetic)
            
            # Metropolis acceptance
            if delta_S < 0 or np.random.rand() < np.exp(-delta_S):
                new_phi[i, j] = proposed_val
                
    return new_phi

def metropolis_simulation(phi0, m, lam, delta, num_sweeps, record_interval=1):
    """
    Run a full Metropolis Monte Carlo simulation for phi^4 theory on a 2D lattice.
    
    Parameters:
      phi0: initial 2D numpy array of field values.
      m: mass parameter (float).
      lam: coupling constant (lambda, float).
      delta: maximum proposal change.
      num_sweeps: total number of full lattice sweeps.
      record_interval: record configuration every 'record_interval' sweeps.
      
    Returns:
      configurations: a list of 2D numpy arrays (recorded field configurations).
      actions: a list of action values corresponding to the recorded configurations.
    """
    configurations = []
    actions = []
    
    phi = phi0.copy()
    configurations.append(phi.copy())
    actions.append(compute_action(phi, m, lam))
    
    for sweep in range(1, num_sweeps+1):
        phi = metropolis_update(phi, m, lam, delta)
        if sweep % record_interval == 0:
            configurations.append(phi.copy())
            actions.append(compute_action(phi, m, lam))
            
    return configurations, actions
