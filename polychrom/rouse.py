""" 
Simulation parameters
---------------------

This module provides some intuition for the units that are used in polychrome.
For example, the mass of a particle is automatically set to 100 amu and 
the collision rate is set depending on the integrator. For Brownian integrators,
setting the collision rate to 2.0 works well, whereas for Langevin integrators,
it is best to set the collision rate to 0.001-0.01. For loop extrusion simulations,
it is typically set to 0.1. These values have been determined empirically by group
members. For Brownian dynamics, for example, using a smaller collision rate leads
to integration failures. 

In any case, the monomer diffusion coefficient naturally follows as
:math:`D=k_B T / m \zeta`, where :math:`\zeta` is the collision rate. So once the
mass and collision rate are set, there is no control over the choice of D unless
you use a custom Brownian integrator (see integrators module) that directly takes
in D as a parameter. However, even then, the units of D are in terms of 
:math:`k_B T / m \zeta`.

The other arbitrary parameters are the monomer radius, set to sim.conlen = 1 nm
by default, and the bondWiggleDistance. The monomer radius sets the length scale
at which repulsive self-avoidance occurs. the bondWiggleDistance, :math:`x`, sets
the stiffness, :math:`k`,  of the springs connecting adjacent monomers, such that
:math:`k = 2k_B T / x^2`. For a Rouse chain, :math:`k = 3 k_B T / y^2`, where
:math:`y` is the standard deviation of the bond extension. Every Rouse chain 
is secretely a wormlike chain at shorter length scales; thus, :math:`y` should
be set to the end-to-end distance of the underlying WLC, which is defined as 
:math:`y = \sqrt{L_0 b}`, where :math:`L_0` is the length of DNA per monomer and
:math:`b` is the Kuhn length. These relations imply that the bondWiggleDistance
should be set to :math:`x = \sqrt{2L_0 b / 3}`. 

Note that all length scales in
polychrome are in terms of sim.conlen = 1 nm. So :math:`L_0` and :math:`b` should
also be in nanometers. It is not obvious how to convert from nanometers of
chromatin to basepairs. One way of doing it is using the formula
n_basepairs = (n_nm / 0.34 nm/bp) * (1 + 146/<L>), where <L> is the average linker
length in the cell. For example, for human T cells, <L> = 50 bp; so b=40nm of 
cumulative linker length would translate to 469 bp of chromatin, where we account
for the buried DNA in nucleosomes. We can also invert this relation to get
n_nm = n_basepairs/(1 + 146/<L>) * 0.34 nm/bp. So if we would like each monomer
to represent 2 kilobases, this would translate to 174 nm of cumulative linker length.
Using these values as an example, the resulting bondWiggleDistance would be
sqrt(2Lb/3) = 68 nm. We then divide by the size of a monomer to understand what the
bondWiggleDistance would be in terms of sim.conlen. So if we posit that the diameter
of a bead is equal to 34 nm, the bondWiggleDistance would be close to 2.0. This is 
a much larger number than 0.1, which is what is used as default in polychrome! 

Generally, the more coarse-grained the simulation is, the more flexible the springs
should be and the larger the bondWiggleDistance should be, since there is more DNA
per bead.

For a self avoiding polymer there are 2 main dimensionless numbers to keep in mind.
One is Ddt/b^2 and the other is a/b, where a is the radius of a monomer and b is the
Kuhn length. Recall that D and dt are set arbitrarily based on computational convenience,
and a is set to 1 nm by default. Thus, even if length scales and time scales can be
rescaled at the end of the simulation to match experimental data, these ratios should
be decided on beforehand and preserved. Most people set the rest length of the spring
to be the diameter of the monomer = 1 nm. 
"""


import numpy as np
from simtk import unit


class SimulationParams(object):
    """
    This class provides guidance on how to choose polychrome parameters based
    on principles of the Rouse polymer model.
    """
    def __init__(self, N=1000, timestep=170, collision_rate=2.0,
            mass=100, temperature=300, length_scale=1.0):
        """
        Parameters
        ----------

        N : int
            number of particles (default 1000)
        timestep : float
            timestep in femtoseconds (default 170)
        collision_rate : float
            collision rate in inverse picoseconds (default 2.0)
        mass : float
            Particle mass (default 100 amu)
        temperature : float
            temperature in kelvins, (defaults to 300 K)
        length_scale : float
            monomer radius and rest length of springs. Defaults to 1 nm.
        """
        self.N = N
        self.timestep = timestep * unit.femtosecond
        self.collision_rate = collision_rate * (1 / unit.picosecond)
        self.length_scale = length_scale * unit.nanometer
        self.temperature = temperature * unit.kelvin
        self.mass = mass * unit.amu
        

    def get_D_from_sim_params(self):
        """ Usually the mass, temperature, conlen, and collision_rate of polychrome
        simulations are unchanged and set to ensure the simulation does not blow up.
        This functions returns the monomer diffusion coefficient in SI units
        implied by these arbitary choices."""

        #convert amu to kilograms -- cannot use this using in_units_of for some reason
        mass_in_kg = self.mass._value * 1.66*10**(-27) * unit.kilogram
        kB = unit.BOLTZMANN_CONSTANT_kB
        kT = kB * self.temperature
        D = kT / (mass_in_kg * collision_rate)
        return D.in_units_of(unit.meter ** 2 / unit.second)

    def get_D_from_measured_Dapp(self, Dapp=0.01 * unit.micrometer ** 2 / unit.second ** 0.5,
            b = 40 * unit.nanometer):
        """ Measurements of the subdiffusive motion of chromosomal loci indicate that
        MSD = D_app t^{1/2}, where Dapp is approximately 0.01 um^2 / s^{1/2}. Here, we
        convert D_app to a monomer diffusion coefficient using a given Kuhn length b. """
        
        D = np.pi * Dapp**2 / (12 * b**2) #in m^2 / second
        return D.in_units_of(unit.meter ** 2 / unit.second)

    def get_rouse_time(self):
        """ Compute expected number of timesteps that corresponds to a rouse time for this polymer. """
        D = self.get_D_from_sim_params()
        rouse_time = (self.N * self.length_scale.in_units_of(unit.meter))**2 / (3 * np.pi**2 * D)
        ntimesteps = np.rint(rouse_time / self.timestep)
        return ntimesteps

    def guess_bondWiggleDistance(L0, b, mean_linker_length, a=None)
        """ Return bond wiggle distance based on the amount of DNA per bead (L0), the
        Kuhn length (b) in basepairs, and the mean linker length in basepairs, and the
        expected radius of a monomer in nanometers (a)."""
        L0_nm = L0 / (1 + 146/mean_linker_length) * 0.34
        b_nm = b / (1 + 146/mean_linker_length) * 0.34
        if a is None:
            a = b_nm
        return np.sqrt(2 * L0_nm * b_nm / 3) / a
