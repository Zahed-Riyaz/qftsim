# simulations.py
import numpy as np

def compute_action(configuration, mass, coupling):
    """
    Compute the Euclidean action for a 1D lattice configuration of a scalar field.
    
    S = sum_x [ 0.5*(phi[x+1]-phi[x])^2 + 0.5*m^2*phi[x]^2 + coupling*phi[x]^4 ]
    (with periodic boundary conditions).
    
    configuration: numpy array of field values.
    mass: mass parameter (float).
    coupling: coupling constant (float).
    
    Returns:
      action: float, the total action for the configuration.
    """
    kinetic = 0.5 * np.sum((np.roll(configuration, -1) - configuration)**2)
    potential = 0.5 * mass**2 * np.sum(configuration**2) + coupling * np.sum(configuration**4)
    return kinetic + potential

def metropolis_update(configuration, mass, coupling, delta):
    """
    Perform a single Metropolis sweep update on the lattice configuration.
    
    configuration: numpy array of field values.
    mass: mass parameter (float).
    coupling: coupling constant (float).
    delta: maximum change for the proposed update.
    
    Returns:
      new_configuration: updated configuration as a numpy array.
    """
    L = len(configuration)
    new_config = configuration.copy()
    for i in range(L):
        old_val = new_config[i]
        # Propose a new value: add a random number from a uniform distribution in [-delta, delta]
        proposed_change = np.random.uniform(-delta, delta)
        new_val = old_val + proposed_change
        
        # Only the terms involving site i (and its nearest neighbors) change.
        i_minus = (i - 1) % L
        i_plus = (i + 1) % L
        
        # Compute the local action contributions before and after the update.
        old_kinetic = 0.5 * ((new_config[i] - new_config[i_minus])**2 + (new_config[i_plus] - new_config[i])**2)
        old_potential = 0.5 * mass**2 * old_val**2 + coupling * old_val**4
        
        new_kinetic = 0.5 * ((new_val - new_config[i_minus])**2 + (new_config[i_plus] - new_val)**2)
        new_potential = 0.5 * mass**2 * new_val**2 + coupling * new_val**4
        
        delta_S = (new_kinetic + new_potential) - (old_kinetic + old_potential)
        
        # Metropolis acceptance: accept if delta_S is negative, or with probability exp(-delta_S) if positive.
        if delta_S < 0 or np.random.rand() < np.exp(-delta_S):
            new_config[i] = new_val
        # Otherwise, reject the change and leave new_config[i] unchanged.
    return new_config

def metropolis_simulation(lattice_size, num_sweeps, mass, coupling, delta, record_interval=1):
    """
    Perform a Metropolis Monte Carlo simulation for a 1D lattice scalar field.
    
    lattice_size: number of lattice sites (int).
    num_sweeps: number of full sweeps over the lattice (int).
    mass: mass parameter (float).
    coupling: coupling constant (float).
    delta: maximum update change (float).
    record_interval: record configuration every 'record_interval' sweeps.
    
    Returns:
      configurations: list of numpy arrays representing the field configurations.
      actions: list of action values corresponding to each recorded configuration.
    """
    configuration = np.random.randn(lattice_size)  # initial random configuration
    configurations = []
    actions = []
    
    # Record the initial configuration
    configurations.append(configuration.copy())
    actions.append(compute_action(configuration, mass, coupling))
    
    for sweep in range(1, num_sweeps + 1):
        configuration = metropolis_update(configuration, mass, coupling, delta)
        if sweep % record_interval == 0:
            configurations.append(configuration.copy())
            actions.append(compute_action(configuration, mass, coupling))
    
    return configurations, actions

# (Optional) Retain your simple update rule from before.
def example_update(configuration):
    noise = np.random.normal(scale=0.1, size=configuration.shape)
    return configuration + noise

def monte_carlo_simulation(lattice_size, num_steps, update_rule):
    """
    A simple Monte Carlo simulation skeleton using a generic update_rule.
    
    lattice_size: the number of lattice sites (1D for simplicity).
    num_steps: number of Monte Carlo steps.
    update_rule: a function defining how to update the field configuration.
    
    Returns:
      configurations: list of configurations recorded during the simulation.
    """
    configuration = np.random.randn(lattice_size)
    configurations = [configuration.copy()]

    for step in range(num_steps):
        configuration = update_rule(configuration)
        configurations.append(configuration.copy())

    return configurations
