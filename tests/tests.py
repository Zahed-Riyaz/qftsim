
import numpy as np
import pytest

# Import functions from your package.
# Adjust the import paths as needed depending on your package structure.
from qftsim.simulate_phi4 import compute_action, metropolis_update, metropolis_simulation
from qftsim.simulate_phi43d import compute_action_3d, metropolis_update_3d, metropolis_simulation_3d

def test_compute_action_constant():
    """
    For a constant 2D configuration, the kinetic term should vanish.
    Thus, the action should equal the number of sites times the potential at that constant value.
    """
    L = 10
    m = 1.0
    lam = 0.1
    constant_value = 2.0
    phi = np.full((L, L), constant_value)
    
    # Expected potential: Sum_x [ 0.5 * m^2 * phi^2 + lam * phi^4 ]
    expected_potential = L * L * (0.5 * m**2 * constant_value**2 + lam * constant_value**4)
    
    computed_action = compute_action(phi, m, lam)
    np.testing.assert_allclose(computed_action, expected_potential, rtol=1e-6,
                               err_msg="2D action for constant field does not match expected value.")

def test_compute_action_3d_constant():
    """
    For a constant 3D configuration, the kinetic term should vanish.
    Thus, the action should equal the number of sites times the potential at that constant value.
    """
    L = 8
    m = 1.0
    lam = 0.1
    constant_value = 1.5
    phi = np.full((L, L, L), constant_value)
    
    expected_potential = L**3 * (0.5 * m**2 * constant_value**2 + lam * constant_value**4)
    
    computed_action = compute_action_3d(phi, m, lam)
    np.testing.assert_allclose(computed_action, expected_potential, rtol=1e-6,
                               err_msg="3D action for constant field does not match expected value.")

def test_metropolis_update_phi4():
    """
    Test that the 2D Metropolis update:
      - Returns a configuration of the same shape.
      - Does not introduce any NaN values.
    """
    L = 16
    m = 1.0
    lam = 0.1
    delta = 0.5
    phi = np.random.randn(L, L)
    new_phi = metropolis_update(phi, m, lam, delta)
    
    assert new_phi.shape == phi.shape, "Updated 2D configuration shape mismatch."
    assert not np.isnan(new_phi).any(), "NaN values found in updated 2D configuration."

def test_metropolis_update_phi4_3d():
    """
    Test that the 3D Metropolis update:
      - Returns a configuration of the same shape.
      - Does not introduce any NaN values.
    """
    L = 8
    m = 1.0
    lam = 0.1
    delta = 0.5
    phi = np.random.randn(L, L, L)
    new_phi = metropolis_update_3d(phi, m, lam, delta)
    
    assert new_phi.shape == phi.shape, "Updated 3D configuration shape mismatch."
    assert not np.isnan(new_phi).any(), "NaN values found in updated 3D configuration."

def test_metropolis_simulation_phi4():
    """
    Test that the 2D phi^4 simulation:
      - Returns a nonempty list of configurations.
      - Each configuration has the expected shape.
      - The actions list has the same length as the configurations list.
    """
    L = 16
    m = 1.0
    lam = 0.1
    delta = 0.5
    num_sweeps = 10
    record_interval = 2
    phi0 = np.random.randn(L, L)
    
    configurations, actions = metropolis_simulation(phi0, m, lam, delta, num_sweeps, record_interval)
    
    assert len(configurations) > 0, "No configurations returned by the 2D simulation."
    for config in configurations:
        assert config.shape == (L, L), "A 2D configuration does not match the expected shape."
    assert len(actions) == len(configurations), "Mismatch between number of configurations and actions."

def test_metropolis_simulation_phi4_3d():
    """
    Test that the 3D phi^4 simulation:
      - Returns a nonempty list of configurations.
      - Each configuration has the expected shape.
      - The actions list has the same length as the configurations list.
    """
    L = 8
    m = 1.0
    lam = 0.1
    delta = 0.5
    num_sweeps = 5
    record_interval = 1
    phi0 = np.random.randn(L, L, L)
    
    configurations, actions = metropolis_simulation_3d(phi0, m, lam, delta, num_sweeps, record_interval)
    
    assert len(configurations) > 0, "No configurations returned by the 3D simulation."
    for config in configurations:
        assert config.shape == (L, L, L), "A 3D configuration does not match the expected shape."
    assert len(actions) == len(configurations), "Mismatch between number of configurations and actions."
