# Lagrangians.py
import sympy as sp

class Lagrangian:
    """
    Class to represent a Lagrangian.
    """
    def __init__(self, fields, terms):
        """
        fields: a list of Field objects.
        terms: a list of symbolic expressions representing terms in the Lagrangian.
        """
        self.fields = fields
        self.terms = terms

    def add_term(self, term):
        self.terms.append(term)

    def display(self):
        lagrangian = sum(self.terms)
        sp.init_printing()
        sp.pprint(lagrangian)
        return lagrangian

# Example: A simple scalar field Lagrangian (symbolic, for illustration)
def scalar_field_lagrangian(mass, coupling):
    phi = sp.symbols('phi')
    # Note: sp.diff(phi, 'x') is only symbolic; a proper lattice derivative would be a finite difference.
    kinetic = 0.5 * sp.diff(phi, 'x')**2  
    potential = 0.5 * mass**2 * phi**2 + coupling * phi**4
    return Lagrangian(fields=["phi"], terms=[kinetic, potential])
