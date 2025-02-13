# fields.py
import numpy as np
import sympy as sp

class Field:
    """
    Base class for a field.
    """
    def __init__(self, name, mass):
        self.name = name
        self.mass = mass  # Mass of the field (could be symbolic)

    def propagator(self, momentum):
        """
        Returns the propagator in momentum space.
        This is a placeholder to be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this!")

class ScalarField(Field):
    """
    A scalar field with a simple propagator.
    """
    def propagator(self, momentum):
        # For a free scalar field, the propagator is 1/(p^2 - m^2 + iε)
        p2 = momentum.dot(momentum)
        epsilon = 1e-10  # A small number to mimic the iε prescription
        return 1.0 / (p2 - self.mass**2 + 1j * epsilon)
