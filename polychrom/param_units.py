""" 
Simulation parameters and the Rouse model
-----------------------------------------

This module provides some intuition for the units that are used in polychrom as
well as some guidance on how to interpret polychrom parameters through the lens
of the Rouse model. Of course, chromatin is not necessarily a Rouse polymer, nor
is a simulated self-avoiding polymer. However, since the Rouse model has been used
to fit experimental data on chromatin dynamics, it provides some intuition for how
to guess polychrom parameters from first principles.

For example, the mass of a particle is automatically set to 100 amu and 
the collision rate is set depending on the integrator. For Brownian integrators,
setting the collision rate to 2.0 works well, whereas for Langevin integrators,
it is best to set the collision rate to 0.001-0.01. For loop extrusion simulations,
it is typically set to 0.1. These values have been determined empirically by group
members. For Brownian dynamics, for example, using a smaller collision rate leads
to integration failures. By default the temperature is set to 300K. 

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
:math:`y` is the standard deviation of the bond extension. The Rouse model of DNA
is secretely a wormlike chain at shorter length scales; thus, :math:`y` should
be set to the end-to-end distance of the underlying WLC, which is defined as 
:math:`y = \sqrt{L_0 b}`, where :math:`L_0` is the length of DNA per monomer and
:math:`b` is the Kuhn length. These relations imply that the bondWiggleDistance
should be set to :math:`x = \sqrt{2L_0 b / 3}`. 

Note that all length scales in
polychrom are in terms of sim.conlen = 1 nm. So :math:`L_0` and :math:`b` should
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
a much larger number than 0.1, which is what is used as default in polychrom! 

Generally, the more coarse-grained the simulation is, the more flexible the springs
should be and the larger the bondWiggleDistance should be, since there is more DNA
per bead. Of course, if bending forces are added then the simulator should choose
:math:`L_0` to be below the persistence length of chromatin (< kilobase). On the flip
side, each monomer should represent more than a kilobase to justify the ommission of
bending rigidity in polychrom simulations of chromatin.

For a self avoiding polymer there are 2 main dimensionless numbers to keep in mind.
One is Ddt/b^2 and the other is a/b, where a is the radius of a monomer and b is the
Kuhn length. Recall that D and dt (timestep) are set arbitrarily based on 
computational convenience, and a is set to 1 nm by default. 
Thus, even if length scales and time scales can be
rescaled at the end of the simulation to match experimental data, these ratios should
be decided on beforehand and preserved. Most people set the rest length of the spring
to be the diameter of the monomer = 1 nm. 
"""


import numpy as np
from simtk import unit


class SimulationParams(object):
    """
    The methods in this class provide ways of calculating physically meaningful
    quantities in the Rouse model fromn polychrom parameter values and vice versa.
    The parameters in the constructor are usually set for computational convenience
    and do not have any physical meaning, except the number of monomers, N.

    Notes
    -----
    This is only one way to guess initial polychrom simulation parameters and is
    exclusively based on the Rouse model, which may or may not apply to chromatin.

    To do
    -----
    - add more functions to guess parameters based on desired dimensionless ratios
    or experimental measurements.
    - create additional classes based on other polymer models (not the Rouse model)
    """

    def __init__(
        self,
        N=1000,
        timestep=170,
        collision_rate=2.0,
        mass=100,
        temperature=300,
        length_scale=1.0,
    ):
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
        """Usually the mass, temperature, conlen, and collision_rate of polychrome
        simulations are unchanged and set to ensure the simulation does not blow up.
        This functions returns the monomer diffusion coefficient in SI units
        implied by these arbitary choices."""

        # convert amu to kilograms -- cannot use this using in_units_of for some reason
        mass_in_kg = self.mass._value * 1.66 * 10 ** (-27) * unit.kilogram
        kB = unit.BOLTZMANN_CONSTANT_kB
        kT = kB * self.temperature
        D = kT / (mass_in_kg * self.collision_rate)
        return D.in_units_of(unit.meter**2 / unit.second)

    def get_D_from_measured_Dapp(
        self,
        Dapp=0.01 * unit.micrometer**2 / unit.second**0.5,
        b=40 * unit.nanometer,
    ):
        """Extract the monomer diffusion coefficient from a given Kuhn length, b, and a measurement of the anomolus diffusion
        coefficient D_app, where MSD = D_app t^{1/2}. Some measurements of the subdiffusive motion of chromosomal loci indicate that
        Dapp is approximately 0.01 um^2 / s^{1/2}. Note, this assumes MSDs scale as :math:`t^{1/2}`. A different polymer model is
        needed if a different scaling with time is observed."""

        if not isinstance(Dapp, unit.Quantity):
            raise ValueError("Dapp should be a simtk.Quantity object")
        if not isinstance(b, unit.Quantity):
            raise ValueError("b should be a simtk.Quantity object")
        D = np.pi * Dapp**2 / (12 * b**2)  # in m^2 / second
        return D.in_units_of(unit.meter**2 / unit.second)

    def get_rouse_time(self, b_nm=None):
        """Compute expected number of timesteps that corresponds to a rouse time for this polymer.
        This is the expected number of Brownian Dynamics timesteps required to equilibrate the
        entire length of the polymer. The Rouse time is :math:`N^2 b^2 / (3 \pi^2 D)`, where N
        is the number of Kuhn lengths and b is the Kuhn length.

        Parameters
        ----------
        b_nm : float
            Kuhn length in nanometers.
        """
        D = self.get_D_from_sim_params()  # in m^2 / s
        if b_nm is None:
            b_nm = self.length_scale._value
        # N is the number of particles with diameter `self.length_scale` nm
        # Nhat is number of Kuhn lengths
        Nhat = (self.N * self.length_scale._value) / b_nm
        b = b_nm * unit.nanometer
        # in units of seconds
        rouse_time = (Nhat * b.in_units_of(unit.meter)) ** 2 / (3 * np.pi**2 * D)
        ntimesteps = np.ceil(rouse_time / self.timestep)
        return ntimesteps

    def guess_bondWiggleDistance(L0, b, mean_linker_length, a=None):
        """Return bond wiggle distance based on the amount of DNA per bead (L0), the
        Kuhn length (b) in basepairs, and the mean linker length in basepairs, and the
        expected radius of a monomer in nanometers (a)."""
        L0_nm = L0 / (1 + 146 / mean_linker_length) * 0.34
        b_nm = b / (1 + 146 / mean_linker_length) * 0.34
        if a is None:
            a = b_nm
        return np.sqrt(2 * L0_nm * b_nm / 3) / a
