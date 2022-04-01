r"""
Custom integrators to use with polychrom
----------------------------------------

Here, we develop Brownian integrators capable of integrating
the overdamped Langevin equation with active, correlated noise.

"""
import numpy as np
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

class CorrelatedNoiseIntegrator(utils.RestorableOpenMMObject, PrettyPrintableIntegrator, mm.CustomIntegrator):
    """ Brownian motion integrator with correlated noise.
    
    rhos = desired C(i, j)
    """
    def __init__(self, timestep, collision_rate, particleD, rhos):
        super(CorrelatedNoiseIntegrator, self).__init__(timestep * unit.femtosecond)
        rhos = np.sign(rhos) * np.sqrt(np.abs(rhos))
        k, N = rhos.shape #each rho is +rho, -rho, or 0
        #global variables
        self.addGlobalVariable("zeta", collision_rate *(1 / unit.picosecond))
        self.addGlobalVariable("k", k)
        #per dof variables
        self.addPerDofVariable("noise", 0)
        self.addPerDofVariable("D", 0)
        self.addPerDofVariable("sigma", 0)
        self.addGlobalVariable("ghostx", 0) #the ghost random variable used to correlate noise
        self.addGlobalVariable("ghosty", 0)
        self.addGlobalVariable("ghostz", 0)
        self.addPerDofVariable("maskx", 0)
        self.addPerDofVariable("masky", 0)
        self.addPerDofVariable("maskz", 0)
        self.addPerDofVariable("xx", 0) #standard normal random variable
        
        #set variables
        self.setPerDofVariableByName("D", particleD)
        mask = np.zeros((N, 3))
        mask[:, 0] = 1
        self.setPerDofVariableByName("maskx", mask)
        mask = np.zeros((N, 3))
        mask[:, 1] = 1
        self.setPerDofVariableByName("masky", mask)
        mask = np.zeros((N, 3))
        mask[:, 2] = 1
        self.setPerDofVariableByName("maskz", mask)
        self.addComputePerDof("sigma", "sqrt(2*D*dt)/sqrt(k)")
        #reset noise to zero at start of time step
        self.addComputePerDof("noise", "noise - noise")
        
        #compute noise per DOF
        for i in range(k):
            rho = np.ones((N, 3)) #per dof variables need to be this shape
            rho[:, 0] = rhos[i, :]
            rho[:, 1] = rhos[i, :]
            rho[:, 2] = rhos[i, :]
            #TODO: check if rho changes upon each iteration of this for loop
            self.addPerDofVariable(f"rho{i}", 0)
            self.setPerDofVariableByName(f"rho{i}", rho)
            #create a ghost random variable -- one per spatial dimension
            self.addComputeGlobal("ghostx", "gaussian")
            self.addComputeGlobal("ghosty", "gaussian")
            self.addComputeGlobal("ghostz", "gaussian")
            self.addComputePerDof("xx", "ghostx*maskx + ghosty*masky + ghostz*maskz") 
            #different standard normal random variable
            self.addComputePerDof("noise", f"noise + sigma * (rho{i} * xx + sqrt(1 - rho{i}*rho{i})*gaussian)")
              
        #Euler Marayama
        self.addPerDofVariable("x1", 0)
        self.addUpdateContextState()
        self.addComputePerDof("x1", "x + (f/(zeta*m))*dt + noise")
        self.addComputePerDof("v", "(x1 - x)/dt")
        self.addComputePerDof("x", "x1")
        #self.addConstrainVelocities()
        self.addConstrainPositions()
