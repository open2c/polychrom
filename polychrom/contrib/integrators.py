r"""
Custom integrators to use with polychrom
----------------------------------------

Here, we develop Brownian integrators capable of integrating
the overdamped Langevin equation for a polymer driven by correlated active forces.
The forces can vary in magnitude along the chain, such that each monomer has a different effective temperature :math:`T_i` or scalar activity :math:`A_i \propto k_B T_i`. In addition, the forces acting on distinct monomers can be correlated in direction. In that case, the equation of motion for the ith monomer of the polymer is
    
.. math::
    \frac{dx(i, t)}{dt} =  \frac{1}{\xi}\vec{f}_{d} + \vec{\eta}_i(t)
    
where :math:`\vec{f}_d` represents all deterministic forces, :math:`\xi` is the friction coefficient, 
and :math:`\vec{\eta}_i(t)` is a mean-zero 
Gaussian random velocity field with `:math:`\langle \eta_{ik} \eta_{jl} \rangle = 
2\sqrt{D_i}\sqrt{D_j}C_{ij}\delta_{kl}`. :math:`k, l` index the spatial components of the noise
vector, and :math:`i, j` index the monomers. Here, :math:`D_i = k_B T_i / \xi` is the diffusion coefficient of the ith monomer, and :math:`C_{ij} \in [-1, 1]` is the Pearson correlation matrix.

The ActiveBrownianIntegrator considers the case where :math:`C_{ij} = \delta_{ij}`, i.e. the random forces acting on distinct monomers are independent, but may vary in magnitude.

The CorrelatedNoiseIntegrator considers the case where :math:`C_{ij}` is non-diagonal.

For examples on how to run simulations with these custom integrators, see examples/customIntegrators.

Notes
-----
The default integrator used in polychrom is a Langevin integrator with variable time stepping. This does not take the overdamped limit of the Langevin equation, i.e. all monomers have inertia and show ballistic dynamics on short time scales. The custom integrators below, in contrast, are Brownian integrators, since there this is a nonequilibrium system and there is no guarantee that underdamped and overdamped dynamics will lead to the same nonequilibrium steady state. However,

- Brownian integrators are A LOT slower than Langevin integrators (i.e. require many more time steps. For example, a chain with 1000 monomers needs about 10^7 time steps to equilibrate).
- Since monomers do not have inertia, their "velocity" is a meaningless quantity. Thus all monomer velocities should be initialized to zero using the Simulation class's `set_velocities()` function. Specify the integrator type as "brownian" so that the simulation code does not raise an error if the polymer's kinetic energy exceeds a threshold. 
- Use a larger collision rate (2.0 works) for brownian integrators since dynamics are overdamped. (Langevin integrators use 0.1 or below).

"""
import numpy as np
import openmm as mm
from openmmtools import utils
from openmmtools.integrators import PrettyPrintableIntegrator
from simtk import unit


class ActiveBrownianIntegrator(utils.RestorableOpenMMObject, PrettyPrintableIntegrator, mm.CustomIntegrator):
    """Brownian dynamics integrator with monomer Diffusion coefficient that varies along the chain.

    Parameters
    ----------
    timestep : float or simtk.unit.Quantity
        time step in units of femtoseconds
    collision_rate : float or simtk.unit.Quantity
        friction coefficient governing collisions with solvent, in units of inverse picoseconds
    particleD: (N, 3) array-like
        diffusion coefficients of N monomers in x,y,z directions in units of kT/(collision_rate * mass)

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
    """Brownian dynamics integrator with correlated active noise.

    To define the correlations, we define a set of k attributes that specify the monomer's identity, such
    as charge, methylation status, etc. For each attribute, all monomers are assigned type 1, type -1, or
    type 0. The integrator defines a procedure by which type 1 monomers are correlated with one another
    with correlation coefficient :math:`\rho`, type -1 monomers are correlated with one another with
    coefficient :math:`\rho`, but type 1 and type -1 monomers are anticorrelated with coefficient :math:`-
    \rho`. Type 0 monomers do not experience correlated fluctuations.

    Parameters
    ----------
    timestep : float or simtk.unit.Quantity
        time step in units of femtoseconds
    collision_rate : float or simtk.unit.Quantity
        friction coefficient governing collisions with solvent, in units of inverse picoseconds
    particleD: (N, 3) array-like
        diffusion coefficients of N monomers in units of kT/(collision_rate * mass)
    rhos: (k, N) array-like
        kth row contains rho, 0, or -rho to assign monomers of type 1, type 0, or type -1 for the
        kth feature, where rho is the associated correlation coefficient
    """

    def __init__(self, timestep, collision_rate, particleD, rhos):
        super(CorrelatedNoiseIntegrator, self).__init__(timestep * unit.femtosecond)
        rhos = np.sign(rhos) * np.sqrt(np.abs(rhos))
        k, N = rhos.shape  # each rho is +rho, -rho, or 0
        # global variables
        self.addGlobalVariable("zeta", collision_rate * (1 / unit.picosecond))
        self.addGlobalVariable("k", k)  # number of features
        # per dof variables
        self.addPerDofVariable("noise", 0)  # noise term in Langevin equation
        self.addPerDofVariable("D", 0)  # monomer diffusion coefficient
        self.addPerDofVariable("sigma", 0)  # standard deviation of noise
        self.addGlobalVariable("ghostx", 0)  # the ghost random variable used to correlate noise
        self.addGlobalVariable("ghosty", 0)
        self.addGlobalVariable("ghostz", 0)
        self.addPerDofVariable("maskx", 0)  # mask y and z directions
        self.addPerDofVariable("masky", 0)  # mask x and z directions
        self.addPerDofVariable("maskz", 0)  # mask x and y directions
        self.addPerDofVariable("xx", 0)  # standard normal random variable

        # set variables
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
        # reset noise to zero at start of time step
        self.addComputePerDof("noise", "noise - noise")

        # compute noise per DOF
        for i in range(k):
            rho = np.ones((N, 3))  # per dof variables need to be this shape
            rho[:, 0] = rhos[i, :]
            rho[:, 1] = rhos[i, :]
            rho[:, 2] = rhos[i, :]
            self.addPerDofVariable(f"rho{i}", 0)
            self.setPerDofVariableByName(f"rho{i}", rho)
            # create a ghost random variable -- one per spatial dimension
            self.addComputeGlobal("ghostx", "gaussian")
            self.addComputeGlobal("ghosty", "gaussian")
            self.addComputeGlobal("ghostz", "gaussian")
            self.addComputePerDof("xx", "ghostx*maskx + ghosty*masky + ghostz*maskz")
            # draw per dof noise that is correlated with the ghost random variable
            self.addComputePerDof(
                "noise",
                f"noise + sigma * (rho{i} * xx + sqrt(1 - rho{i}*rho{i})*gaussian)",
            )

        # Euler Marayama
        self.addPerDofVariable("x1", 0)
        self.addUpdateContextState()
        self.addComputePerDof("x1", "x + (f/(zeta*m))*dt + noise")
        self.addComputePerDof("v", "(x1 - x)/dt")
        self.addComputePerDof("x", "x1")
        self.addConstrainVelocities()
        self.addConstrainPositions()
