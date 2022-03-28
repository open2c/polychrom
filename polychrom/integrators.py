r"""
Custom integrators to use with polychrom
----------------------------------------

Here, we develop Brownian integrators capable of integrating
the overdamped Langevin equation with active, correlated noise.

"""
from simtk import unit
import openmm as mm
from openmmtools import utils
from openmmtools.integrators import PrettyPrintableIntegrator

class ActiveBrownianIntegrator(utils.RestorableOpenMMObject, PrettyPrintableIntegrator, mm.CustomIntegrator):
    """ Brownian integrator with Diffusion coefficient that varies along the chain.

    Parameters
    ----------
    timestep : float or simtk.unit.Quantity
        time step in units of femtoseconds
    collision_rate : float or simtk.unit.Quantity
        friction coefficient governing collisions with solvent, in units of inverse picoseconds
    particleD: (N, 3) array-like
        diffusion coefficients of N monomers in units of kT/(collision_rate * mass)

    """
    def __init__(self, timestep, collision_rate, particleD):
        super(ActiveBrownianIntegrator, self).__init__(timestep * unit.femtosecond)
        self.addGlobalVariable("zeta", collision_rate * (1 / unit.picosecond))
        self.addPerDofVariable("D", 0)
        self.addPerDofVariable("x1", 0)
        self.setPerDofVariableByName("D", particleD)
        self.addUpdateContextState()
        self.addComputePerDof("x1", "x + (f/(zeta*m))*dt + sqrt(2*D*dt)*gaussian")
        self.addComputePerDof("v", "(x1 - x)/dt")
        self.addComputePerDof("x", "x1")
        self.addConstrainVelocities()
        self.addConstrainPositions()
