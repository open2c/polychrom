"""
Creating a simulation: Simulation class
=======================================

Both initialization and running the simulation is done by interacting with an instance 
of :py:class:`polychrom.simulation.Simulation` class.  

Overall parameters
------------------

Overall technical parameters of a simulation are generally initialized in the constructor of the 
Simulation class. :py:meth:`polychrom.simulation.Simulation.__init__` . This includes 

**Techcnical parameters not affecting the output of simulations**

* Platform (cuda (usually), opencl, or CPU (slow)) 
* GPU index
* reporter (where to save results): see :py:mod`polychrom.hdf5_reporter`

**Parameters affecting the simulation**

* number of particles
* integrator (we usually use variable Langevin) + error tolerance of integrator
* collision rate 
* Whether to use periodic boundary conditions (PBC)
* timestep (if using non-variable integrator)

**Parameters that are changed rarely, but may be useful**

* particle mass, temperature and length scale 
* kinetic energy at which to raise an error 
* OpenMM precision
* Rounding before saving (default is to 0.01) 

Starting conformation is loaded using :meth:`polychrom.simulation.Simulation.set_data` method. 
Many tools for creating starting conformations are in :mod:`polychrom.starting_conformations`

Adding forces 
-------------

**Forces** define the main aspects of a given simulation. Polymer connectivity, confinement, crosslinks, tethering monomers, etc. 
are all defined as different forces acting on the particles. 

Typicall used forces are listed in :py:mod:`polychrom.forces` module. Forces out of there can be added using :py:meth:`polychrom.simulation.Simulation.add_force` method. 

Forces and their parameters are an essential part of nearly any polymer simulations. Some forces have just a few paramters (e.g. spherical confinement just needs a radius), while other forces may have lots of parameters and can define complex structures. For example, harmonidBondForce with a specially-created bond list was used to create a backbone-plectoneme conformation in Caulobacter simulations (Le et al, Science 2013). Same harmonic bonds that change over time are used to simulate loop extrusion as in (Fudenberg, 2016). 

Some forces need to be added together. Those include forces defining polymer connectivity. Those forces are combined 
into **forcekits**. Forcekits are defined in :py:mod:`polychrom.forcekits` module. The only example 
of a forcekit for now is defining polymer connectivity using bonds, polymer stiffness, and inter-monomer interaction 
("nonbonded force"). 

Some forces were written for openmm-polymer library and were not fully ported/tested into the polychrom library. 
Those forces reside in :py:mod:`polychrom.legacy.forces` module. Some of them can be used as is, and some of them 
would need to be copied to your code and potentially conformed to the new style of defining forces. This includes 
accepting simulation object as a parameter, and having a ``.name`` attribute. 


Defining your own forces
------------------------

Each force in :py:mod:`polychrom.forces` is a simple function that wraps creation of an openmm force object. 
Users can create new forces in the script defining their simulation and add them using add_force method. 
Good examples of forces are in :py:mod:`polychrom.forces` - all but harmonic bond force use custom forces, 
and provide explanations of why particular energy function was chosen. Description of the module :py:mod:`polychrom.forces` 
has some important information about adding new forces. 



Running a simulation 
--------------------

To run a simulation, you call :py:meth:`polychrom.simulation.Simulation.doBlock` method in a loop. 
Unless specified otherwise, this would save a conformation into a defined reporter. Terminating a 
simulation is not necessary; however, terminating a reporter using reporter.dump_data() is needed for 
the hdf5 reporter. This all can be viewed in the example script. 

"""


from __future__ import absolute_import, division, print_function
import numpy as np
import sys
import os
import time
import tempfile
import logging
import warnings
import hoomd
import gsd.hoomd
import copy
from collections.abc import Iterable

try:
    import openmm
except Exception:
    import simtk.openmm as openmm
import simtk.unit

from polychrom import forces


logging.basicConfig(level=logging.INFO)

# updated manually every now and then
VER_LATEST = "7.7"
VER_DATE = "2022-03-13"

if hasattr(openmm, "__version__"):
    ver_cur = openmm.__version__
    if ver_cur < VER_LATEST:
        warnings.warn(
            f"\n WARNING: you have OpenMM {ver_cur}; {VER_LATEST} is the latest as of {VER_DATE}, "
            "Upgrade is recommended."
        )
        print("to upgrade openmm, run --->  conda install -c conda-forge openmm")
        print("Ideally in a new conda environment")


class IntegrationFailError(Exception):
    pass


class EKExceedsError(Exception):
    pass


class Simulation(object):
    """
    This is a base class for creating a Simulation and interacting with it. All
    general simulation parameters are defined in the constructor.
    Forces are defined in :py:mod:`polychrom.forces` module, and are added
    using :py:meth:`polychrom.simulation.Simulation.add_force` method.
    """

    def __init__(self, **kwargs):
        """
        All numbers here are floats. Units specified in a parameter.

        Parameters
        ----------

        N : int
            number of particles

        error_tol : float, optional
            Error tolerance parameter for variableLangevin integrator
            Values of around 0.01 are reasonable for a "nice" simulation
            (i.e. simulation with soft forces etc).
            Simulations with strong forces may need 0.001 or less
            OpenMM manual recommends 0.001, but our forces tend to be "softer" than theirs

        timestep : number
            timestep in femtoseconds. Mandatory for non-variable integrators.
            Ignored for variableLangevin integrator. Value of 70-80 are appropriate

        collision_rate : number
            collision rate in inverse picoseconds. values of 0.01 or 0.05 are often used.
            Consult with lab members on values.

            In brief, equilibrium simulations likely do not care about the exact dynamics
            you're using, and therefore can be simulated in a "ballistic" dynamics with
            col_rate of around 0.001-0.01.

            Dynamical simulations and active simulations may be more sensitive to col_rate,
            though this is still under discussion/investigation.

            Johannes converged on using 0.1 for loop extrusion simulations, just to be safe.

        PBCbox : (float,float,float) or False; default:False
            Controls periodic boundary conditions
            If PBCbox is False, do not use periodic boundary conditions
            If intending to use PBC, then set PBCbox to (x,y,z) where x,y,z are dimensions
            of the bounding box for PBC

        GPU : GPU index as a string ("0" for first, "1" for second etc.)
            Machines with 1 GPU automatically select their GPU.

        integrator : "langevin", "variableLangevin", "verlet", "variableVerlet",
                     "brownian", optional Integrator to use
                     (see Openmm class reference)

        mass : number or np.array
            Particle mass (default 100 amu)

        temperature : simtk.units.quantity(units.kelvin), optional
            Temperature of the simulation. Devault value is 300 K.

        verbose : bool, optional
            If True, prints a lot of stuff in the command line.

        length_scale : float, optional
            The geometric scaling factor of the system.
            By default, length_scale=1.0 and harmonic bonds and repulsive
            forces have the scale of 1 nm.

        max_Ek: float, optional
            raise error if kinetic energy in (kT/particle) exceeds this value

        platform : string, optional
            Platform to use:
            CUDA (preferred fast GPU platform)
            OpenCL (maybe slower GPU platofrm, does not need CUDA installed)
            CPU (medium speed parallelized CPU platform)
            reference (slow CPU platform for debug)

        verbose : bool, optional
            Shout out loud about every change.

        precision: str, optional (not recommended to change)
            mixed is optimal for most situations.
            If you are using double precision, it will be slower by a factor of 10 or so.

        save_decimals: int or False, optional
            Round to this number of decimals before saving. ``False`` is no rounding.
            Default is 2. It gives maximum error of 0.005, which is nearly always harmless
            but saves up to 40% of storage space (0.6 of the original)
            Using one decimal is safe most of the time, and reduces storage to 40% of int32.
            NOTE that using periodic boundary conditions will make storage advantage less.

        reporters: list, optional
            List of reporters to use in the simulation.

        """
        default_args = {
            "platform": "CUDA",
            "GPU": "0",
            "integrator": "variablelangevin",
            "temperature": 300,
            "PBCbox": False,
            "length_scale": 1.0,
            "mass": 100,
            "reporters": [],
            "max_Ek": 10,
            "precision": "mixed",
            "save_decimals": 2,
            "verbose": False,
            "backend": "openmm",
        }
        valid_names = list(default_args.keys()) + [
            "N",
            "error_tol",
            "collision_rate",
            "timestep",
        ]
        for i in kwargs.keys():
            if i not in valid_names:
                raise ValueError(
                    "incorrect argument provided: {0}. Allowed are {1}".format(i, valid_names)
                )

        if None in kwargs.values():
            raise ValueError(
                "None is not allowed in arguments due to HDF5 incompatiliblity. Use False instead."
            )
        default_args.update(kwargs)
        kwargs = default_args
        self.kwargs = kwargs

        platform = kwargs["platform"]
        self.GPU = kwargs["GPU"]  # setting default GPU
        self.backend = kwargs["backend"]

        properties = {}
        if self.GPU.lower() != "default":
            if platform.lower() in ["cuda", "opencl"]:
                properties["DeviceIndex"] = str(self.GPU)
                properties["Precision"] = kwargs["precision"]
        self.properties = properties
        if self.backend == "openmm":
            if platform.lower() == "opencl":
                platform_object = openmm.Platform.getPlatformByName("OpenCL")
            elif platform.lower() == "reference":
                platform_object = openmm.Platform.getPlatformByName("Reference")
            elif platform.lower() == "cuda":
                platform_object = openmm.Platform.getPlatformByName("CUDA")
            elif platform.lower() == "cpu":
                platform_object = openmm.Platform.getPlatformByName("CPU")
            else:
                raise RuntimeError("Undefined platform: {0}".format(platform))
        else:
            if platform.lower() == "cuda":
                platform_object = hoomd.Simulation(
                    device=hoomd.device.GPU(), seed=np.random.randint(100000)
                )
            elif platform.lower() == "cpu":
                platform_object = hoomd.Simulation(
                    device=hoomd.device.CPU(), seed=np.random.randint(100000)
                )
            else:
                raise RuntimeError("Undefined platform: {0}".format(platform))
        self.platform = platform_object

        self.temperature = kwargs["temperature"]

        self.collisionRate = kwargs["collision_rate"] * (1 / simtk.unit.picosecond)

        self.integrator_type = kwargs["integrator"]

        self.kB = simtk.unit.BOLTZMANN_CONSTANT_kB * simtk.unit.AVOGADRO_CONSTANT_NA
        self.kT = self.kB * self.temperature * simtk.unit.kelvin  # thermal energy
        if isinstance(self.integrator_type, str):
            self.integrator_type = str(self.integrator_type)
            if self.integrator_type.lower() == "dpd":
                if self.backend == "openmm":
                    raise NotImplementedError("DPD is not implemented in OpenMM")
                else:
                    self.nl = hoomd.md.nlist.Cell(0.4)
                    self.dpd = hoomd.md.pair.DPD(
                        default_r_cut=1.0,
                        nlist=self.nl,
                        kT=self.kT.value_in_unit(simtk.unit.kilojoule_per_mole),
                    )
                    self.dpd.params.default = dict(
                        A=0,
                        gamma=self.kwargs["mass"] * kwargs["collision_rate"],
                    )

                    nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
                    self.integrator = hoomd.md.Integrator(
                        dt=kwargs["timestep"] * 1e-3, methods=[nve], forces=[self.dpd]
                    )

            elif self.integrator_type.lower() == "langevin":
                if self.backend == "openmm":
                    self.integrator = openmm.LangevinIntegrator(
                        self.temperature,
                        kwargs["collision_rate"] * (1 / simtk.unit.picosecond),
                        kwargs["timestep"] * simtk.unit.femtosecond,
                    )
                else:
                    nvt = hoomd.md.methods.NVT(
                        kT=self.kT.value_in_unit(simtk.unit.kilojoule_per_mole),
                        filter=hoomd.filter.All(),
                        tau=1 / kwargs["collision_rate"],
                    )
                    self.integrator = hoomd.md.Integrator(
                        dt=kwargs["timestep"] * 1e-3, methods=[nvt]
                    )
            elif self.integrator_type.lower() == "variablelangevin":
                if self.backend == "openmm":
                    self.integrator = openmm.VariableLangevinIntegrator(
                        self.temperature,
                        kwargs["collision_rate"] * (1 / simtk.unit.picosecond),
                        kwargs["error_tol"],
                    )
                else:
                    raise NotImplementedError(
                        f"VariableLangevin not implemented for backend: {self.backend}"
                    )

            elif self.integrator_type.lower() == "langevinmiddle":
                if self.backend == "openmm":
                    self.integrator = openmm.LangevinMiddleIntegrator(
                        self.temperature,
                        kwargs["collision_rate"] * (1 / simtk.unit.picosecond),
                        kwargs["timestep"] * simtk.unit.femtosecond,
                    )
                else:
                    raise NotImplementedError(
                        f"LangevinMiddle not implemented for backend: {self.backend}"
                    )
            elif self.integrator_type.lower() == "verlet":
                if self.backend == "hoomd":
                    nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
                    self.integrator = hoomd.md.Integrator(
                        dt=kwargs["timestep"] * 1e-3, methods=[nve]
                    )
                else:
                    self.integrator = openmm.VerletIntegrator(
                        kwargs["timestep"] * simtk.unit.femtosecond
                    )
            elif self.integrator_type.lower() == "variableverlet":
                if self.backend == "openmm":
                    self.integrator = openmm.VariableVerletIntegrator(kwargs["error_tol"])
                else:
                    raise NotImplementedError(
                        f"Variableverlet not implemented for backend: {self.backend}"
                    )

            elif self.integrator_type.lower() == "brownian":
                if self.backend == "openmm":
                    self.integrator = openmm.BrownianIntegrator(
                        self.temperature,
                        kwargs["collision_rate"] * (1 / simtk.unit.picosecond),
                        kwargs["timestep"] * simtk.unit.femtosecond,
                    )
                else:
                    brown = hoomd.md.methods.Brownian(
                        kT=self.kT.value_in_unit(simtk.unit.kilojoule_per_mole),
                        filter=hoomd.filter.All(),
                        default_gamma=self.kwargs["mass"]
                        * kwargs["collision_rate"],  # amu/picosecond
                    )
                    self.integrator = hoomd.md.Integrator(
                        dt=kwargs["timestep"] * 1e-3, methods=[brown]
                    )
        else:
            if self.backend == "openmm":
                logging.info("Using the provided integrator object")
                self.integrator = self.integrator_type
                self.integrator_type = "UserDefined"
                kwargs["integrator"] = "user_defined"
            else:
                raise NotImplementedError(
                    f"provided integrator objects are not implemented for backend: {self.backend}"
                )
        if self.backend == "hoomd":
            self.thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(
                filter=hoomd.filter.All()
            )
            self.platform.operations.computes.append(self.thermodynamic_properties)

        self.N = kwargs["N"]

        self.verbose = kwargs["verbose"]
        self.reporters = kwargs["reporters"]
        self.forces_applied = False
        self.length_scale = kwargs["length_scale"]
        self.eK_critical = kwargs["max_Ek"]  # Max allowed kinetic energy

        self.step = 0
        self.block = 0
        self.time = 0

        self.nm = simtk.unit.nanometer

        # All masses are the same,
        # unless individual mass multipliers are specified in self.load()
        self.conlen = 1.0 * simtk.unit.nanometer * self.length_scale

        self.kbondScalingFactor = float(
            (2 * self.kT / self.conlen**2)
            / (simtk.unit.kilojoule_per_mole / simtk.unit.nanometer**2)
        )
        if self.backend == "openmm":
            self.system = openmm.System()
        else:
            snapshot = gsd.hoomd.Frame()
            snapshot.particles.N = self.N
            snapshot.particles.typeid = np.array([0] * self.N)
            snapshot.particles.types = ["A"]
            self.system = snapshot
            self.platform.operations.integrator = self.integrator
            self.untouched_force_dict = {}

        # adding PBC
        self.PBC = False
        if kwargs["PBCbox"]:
            self.PBC = True
            PBCbox = np.array(kwargs["PBCbox"])
            if self.backend == "openmm":
                self.system.setDefaultPeriodicBoxVectors(
                    [float(PBCbox[0]), 0.0, 0.0],
                    [0.0, float(PBCbox[1]), 0.0],
                    [0.0, 0.0, float(PBCbox[2])],
                )
            else:
                self.system.configuration.box = [
                    float(PBCbox[0]),
                    float(PBCbox[1]),
                    float(PBCbox[2]),
                    0,
                    0,
                    0,
                ]

        self.force_dict = {}  # Dictionary to store forces

        # saving arguments - not trying to save reporters because they are not serializable
        kwCopy = {i: j for i, j in kwargs.items() if i != "reporters"}
        for reporter in self.reporters:
            reporter.report("initArgs", kwCopy)

    def get_data(self):
        "Returns an Nx3 array of positions"
        return np.asarray(self.data / simtk.unit.nanometer, dtype=np.float32)

    def get_scaled_data(self):
        """Returns data, scaled back to PBC box"""
        if not self.PBC:
            return self.get_data()
        alldata = self.get_data()
        boxsize = np.array(self.kwargs["PBCbox"])
        mults = np.floor(alldata / boxsize[None, :])
        toRet = alldata - mults * boxsize[None, :]
        assert toRet.min() >= 0
        return toRet

    def set_data(self, data, center=False, random_offset=1e-5, report=True):
        """Sets particle positions

        Parameters
        ----------

        data : Nx3 array-like
            Array of positions

        center : bool or "zero", optional
            Move center of mass to zero before starting the simulation
            if center == "zero", then center the data such as all positions are positive and start at zero

        random_offset: float or None
            add random offset to each particle
            Recommended for integer starting conformations and in general

        report : bool, optional
            If set to False, will not report this action to reporters.

        """

        data = np.asarray(data, dtype="float")
        if len(data) != self.N:
            raise ValueError(f"length of data, {len(data)} does not match N, {self.N}")

        if data.shape[1] != 3:
            raise ValueError(
                "Data is not shaped correctly. Needs (N,3), provided: {0}".format(data.shape)
            )
        if np.isnan(data).any():
            raise ValueError("Data contains NANs")

        if random_offset:
            data = data + (np.random.random(data.shape) * 2 - 1) * random_offset

        if center is True:
            av = np.mean(data, axis=0)
            data -= av
        elif center == "zero":
            minvalue = np.min(data, axis=0)
            data -= minvalue
        if self.backend == "openmm":
            self.data = simtk.unit.Quantity(data, simtk.unit.nanometer)
        else:
            self.data = data
        if report:
            for reporter in self.reporters:
                reporter.report(
                    "starting_conformation",
                    {"pos": data, "time": self.time, "block": self.block},
                )

        if hasattr(self, "context"):
            self.init_positions()

    def set_velocities(self, v):
        """Set initial velocities of particles.

        Parameters
        ----------
        v : (N, 3) array-like
            initial x, y, z velocities of the N particles
        """

        v = np.asarray(v, dtype="float")
        if len(v) != self.N:
            raise ValueError(f"length of velocity array, {len(v)} does not match N, {self.N}")

        if v.shape[1] != 3:
            raise ValueError(
                "Data is not shaped correctly. Needs (N,3), provided: {0}".format(v.shape)
            )
        if np.isnan(v).any():
            raise ValueError("Data contains NANs")
        self.velocities = simtk.unit.Quantity(v, simtk.unit.nanometer / simtk.unit.picosecond)
        if hasattr(self, "context"):
            self.init_velocities()

    def RG(self):
        """
        Returns
        -------

        Gyration ratius in units of length (bondlength).
        """
        data = self.get_scaled_data()
        data = data - np.mean(data, axis=0)[None, :]
        return np.sqrt(np.sum(np.var(np.array(data), 0)))

    def dist(self, i, j):
        """
        Calculates distance between particles i and j.

        Added for convenience, and not for production code. Not for use in large for-loops.
        """
        data = self.get_data()
        dif = data[i] - data[j]
        return np.sqrt(sum(dif**2))

    def add_force(self, force):
        """
        Adds a force or a forcekit to the system.
        """
        if isinstance(force, Iterable):
            for f in force:
                self.add_force(f)
        else:
            if force.name is None:
                pass
            else:
                if force.name in self.force_dict:
                    raise ValueError(
                        "A force named {} was added to the system twice!".format(force.name)
                    )
                if self.backend == "openmm":
                    forces._prepend_force_name_to_params(force)
                self.force_dict[force.name] = force
                if self.backend == "hoomd":
                    self.untouched_force_dict[force.name] = copy.deepcopy(force)

        if self.forces_applied:
            raise RuntimeError("Cannot add force after the context has been created")

    def _apply_forces(self):
        """
        Adds all particles and forces to the system.
        Then applies all the forces in the forcedict.
        Forces should not be modified after that, unless you do it carefully
        (see openmm reference).

        This method is called automatically when you run energy minimization,
        or run your first block. On rare occasions, you would need to run it manually,
        but then you probably know what you're doing.

        One example when this method is used is a loop extrusion code (extrusion_3d.ipynb).
        In that case, you restart a simulation, but don't do energy minimization.
        However, before doing the first block, you just need to advance the integrator.
        This requires manually creating context/etc which would be normally done by
        the do_block method.
        """

        if self.forces_applied:
            return

        self.masses = np.zeros(self.N, dtype=float) + self.kwargs["mass"]
        if self.backend == "openmm":
            for mass in self.masses:
                self.system.addParticle(mass)
        else:
            self.system.particles.mass = self.masses

        for i in list(self.force_dict.keys()):  # Adding forces
            force = self.force_dict[i]

            if hasattr(force, "CutoffNonPeriodic") and hasattr(force, "CutoffPeriodic"):
                if self.backend == "hoomd":
                    raise NotImplementedError("Cutoffstuffs not implemented for hoomd yet")
                else:
                    if self.PBC:
                        force.setNonbondedMethod(force.CutoffPeriodic)
                        logging.info("Using periodic boundary conditions")
                    else:
                        force.setNonbondedMethod(force.CutoffNonPeriodic)
            if self.backend == "openmm":
                logging.info(
                    "adding force {} {}".format(i, self.system.addForce(self.force_dict[i]))
                )
            else:
                logging.info(
                    "adding force {} {}".format(
                        i, self.integrator.forces.append(self.force_dict[i])
                    )
                )

        for reporter in self.reporters:
            reporter.report(
                "applied_forces",
                {i: j.__getstate__() for i, j in self.force_dict.items()},
            )
        if self.backend == "openmm":

            self.context = openmm.Context(
                self.system, self.integrator, self.platform, self.properties
            )
        self.init_positions()
        if self.backend == "hoomd":
            if hasattr(self, "velocities"):
                self.system.particles.velocity = self.velocities
            self.platform.create_state_from_snapshot(self.system)
            self.context = self.platform.state
            self.platform.operations.integrator = self.integrator
            self.platform.run(0)
            eP = (
                self.thermodynamic_properties.potential_energy
                / self.N
                / self.kT.value_in_unit(simtk.unit.kilojoule_per_mole)
            )

            logging.info("Particles loaded. Potential energy is %lf" % eP)

        self.init_velocities()
        self.forces_applied = True

    def initialize(self, **kwargs):
        """Initialize, particles, velocities for the first time.
        Only need to use this function if your system has no forces (free Brownian particles).
        Otherwise _apply_force() will execute these lines to add particles to the system,
        initialize their positions/velocities, initialize the context.
        """

        self.masses = np.zeros(self.N, dtype=float) + self.kwargs["mass"]
        for mass in self.masses:
            self.system.addParticle(mass)
        self.context = openmm.Context(self.system, self.integrator, self.platform, self.properties)
        self.init_positions()
        self.init_velocities(**kwargs)

    def init_velocities(self, temperature="current"):
        """Initializes particles velocities

        Parameters
        ----------
        temperature: temperature to set velocities (default: temerature of the simulation)
        v: (N,) array-like
            Vector of initial velocities for the N particles. If None, velocities are chosen
            from a Boltzmann distribution at a given temperature.
        """
        try:
            self.context
        except:
            raise ValueError("No context, cannot set velocs.Initialize context before that")

        if hasattr(self, "velocities"):
            if self.backend == "openmm":
                self.context.setVelocities(self.velocities)
            else:
                pass  # already set in hoomd
            return

        if temperature == "current":
            temperature = self.temperature
        if self.backend == "openmm":
            self.context.setVelocitiesToTemperature(temperature)
        else:
            self.context.thermalize_particle_momenta(
                filter=hoomd.filter.All(), kT=self.kT.value_in_unit(simtk.unit.kilojoule_per_mole)
            )

    def init_positions(self):
        """Sends particle coordinates to OpenMM system.
        If system has exploded, this is
         used in the code to reset coordinates."""
        if self.backend == "openmm":
            try:
                self.context
            except:
                raise ValueError(
                    "No context, cannot set positions. Initialize context before that"
                )
            self.context.setPositions(self.data)
            eP = self.context.getState(getEnergy=True).getPotentialEnergy() / self.N / self.kT
            logging.info("Particles loaded. Potential energy is %lf" % eP)

        else:
            self.system.particles.position = self.data

    def reinitialize(self):
        """Reinitializes the OpenMM context object.
        This should be called if low-level parameters,
        such as parameters of forces, have changed
        """

        self.context.reinitialize()
        self.init_positions()
        self.init_velocities()

    def local_energy_minimization(self, tolerance=0.3, maxIterations=0, random_offset=0.02):
        """
        A wrapper to the build-in OpenMM Local Energy Minimization

        See caveat below

        Parameters
        ----------

        tolerance: float
            It is something like a value of force below which
            the minimizer is trying to minimize energy to.
            see openmm documentation for description

            Value of 0.3 seems to be fine for most normal forces.

        maxIterations: int
            Maximum # of iterations for minimization to do.
            default: 0 means there is no limit

            This is relevant especially if your simulation does not have a
            well-defined energy minimum (e.g. you want to simulate a collapse of a chain
            in some potential). In that case, if you don't limit energy minimization,
            it will attempt to do a whole simulation for you. In that case, setting
            a limit to the # of iterations will just stop energy minimization manually when
            it reaches this # of iterations.

        random_offset: float
            A random offset to introduce after energy minimization.
            Should ideally make your forces have realistic values.

            For example, if your stiffest force is polymer bond force
            with "wiggle_dist" of 0.05, setting this to 0.02 will make
            separation between monomers realistic, and therefore will
            make force values realistic.

            See why do we need it in the caveat below.


        Caveat
        ------

        If using variable langevin integrator after minimization, a big error may
        happen in the first timestep. The reason is that enregy minimization
        makes all the forces basically 0. Variable langevin integrator measures
        the forces and assumes that they are all small - so it makes the timestep
        very large, and at the first timestep it overshoots completely and energy goes up a lot.

        The workaround for now is to randomize positions after energy minimization

        """

        logging.info("Performing local energy minimization")

        self._apply_forces()

        if self.backend == "openmm":
            self.state = self.context.getState(getPositions=False, getEnergy=True)
            eK = self.state.getKineticEnergy() / self.N / self.kT
            eP = self.state.getPotentialEnergy() / self.N / self.kT
            locTime = self.state.getTime()
        else:
            eK = (
                self.thermodynamic_properties.kinetic_energy
                / self.N
                / self.kT.value_in_unit(simtk.unit.kilojoule_per_mole)
            )
            eP = (
                self.thermodynamic_properties.potential_energy
                / self.N
                / self.kT.value_in_unit(simtk.unit.kilojoule_per_mole)
            )
            locTime = self.platform.walltime
        logging.info("before minimization eK={0}, eP={1}, time={2}".format(eK, eP, locTime))
        if self.backend == "openmm":
            openmm.LocalEnergyMinimizer.minimize(self.context, tolerance, maxIterations)
        else:
            fire = hoomd.md.minimize.FIRE(
                dt=self.kwargs["timestep"] * 1e-3 * 1e-2,
                energy_tol=5e-2 * self.kT.value_in_unit(simtk.unit.kilojoule_per_mole),
                force_tol=5e-2 * self.kT.value_in_unit(simtk.unit.kilojoule_per_mole),
                angmom_tol=5e-2 * self.kT.value_in_unit(simtk.unit.kilojoule_per_mole),
                forces=[copy.deepcopy(i) for i in list(self.untouched_force_dict.values())],
                methods=[hoomd.md.methods.NVE(filter=hoomd.filter.All())],
            )
            self.platform.operations.integrator = fire

            while not (fire.converged):
                self.platform.run(100)
                print(
                    f"kin temp = {self.thermodynamic_properties.kinetic_temperature}, E_P/N ="
                    f" {self.thermodynamic_properties.potential_energy   / self.N/ self.kT.value_in_unit(simtk.unit.kilojoule_per_mole)}"
                )
                # gsd_optimized_writer.write(sim.state, gsd_optimized_writer.filename)

            self.platform.operations.integrator = self.integrator

        if self.backend == "openmm":
            self.state = self.context.getState(getPositions=True, getEnergy=True)
            eK = self.state.getKineticEnergy() / self.N / self.kT
            eP = self.state.getPotentialEnergy() / self.N / self.kT
            coords = self.state.getPositions(asNumpy=True)
            locTime = self.state.getTime()
        else:

            self.init_velocities()

            eK = (
                self.thermodynamic_properties.kinetic_energy
                / self.N
                / self.kT.value_in_unit(simtk.unit.kilojoule_per_mole)
            )
            eP = (
                self.thermodynamic_properties.potential_energy
                / self.N
                / self.kT.value_in_unit(simtk.unit.kilojoule_per_mole)
            )
            locTime = self.platform.walltime
            coords = self.platform.state.get_snapshot().particles.position

        self.data = coords
        self.set_data(self.get_data(), random_offset=random_offset, report=False)
        for reporter in self.reporters:
            reporter.report(
                "energy_minimization",
                {"pos": self.get_data(), "time": self.time, "block": self.block},
            )

        logging.info("after minimization eK={0}, eP={1}, time={2}".format(eK, eP, locTime))

    def do_block(
        self,
        steps=None,
        check_functions=[],
        get_velocities=False,
        save=True,
        save_extras={},
    ):
        """performs one block of simulations, doing steps timesteps,
        or steps_per_block if not specified.

        Parameters
        ----------

        steps : int or None
            Number of timesteps to perform.
        increment : bool, optional
            If true, will not increment self.block and self.steps counters
        """

        if not self.forces_applied:
            if self.verbose:
                logging.info("applying forces")
                sys.stdout.flush()
            self._apply_forces()
            self.forces_applied = True

        a = time.time()
        if self.backend == "openmm":
            self.integrator.step(steps)  # integrate!
            self.state = self.context.getState(
                getPositions=True, getVelocities=get_velocities, getEnergy=True
            )
        else:
            self.platform.run(steps)
            self.state = self.platform.state.get_snapshot()

        b = time.time()
        if self.backend == "openmm":
            coords = self.state.getPositions(asNumpy=True)
            newcoords = coords / simtk.unit.nanometer
            self.time = self.state.getTime() / simtk.unit.picosecond
            # calculate energies in KT/particle
            eK = self.state.getKineticEnergy() / self.N / self.kT
            eP = self.state.getPotentialEnergy() / self.N / self.kT
        else:

            newcoords = self.state.particles.position
            coords = newcoords
            self.time = self.platform.timestep
            # calculate energies in KT/particle
            eK = (
                self.thermodynamic_properties.kinetic_energy
                / self.N
                / self.kT.value_in_unit(simtk.unit.kilojoule_per_mole)
            )
            eP = (
                self.thermodynamic_properties.potential_energy
                / self.N
                / self.kT.value_in_unit(simtk.unit.kilojoule_per_mole)
            )

        newcoords = np.array(newcoords, dtype=np.float32)
        if self.kwargs["save_decimals"] is not False:
            newcoords = np.round(newcoords, self.kwargs["save_decimals"])

        curtime = self.time  # self.state.getTime() / simtk.unit.picosecond

        msg = "block %4s " % int(self.block)
        msg += "pos[1]=[%.1lf %.1lf %.1lf] " % tuple(newcoords[0])

        check_fail = False
        for check_function in check_functions:
            if not check_function(newcoords):
                check_fail = True

        if np.isnan(newcoords).any():
            raise IntegrationFailError("Coordinates are NANs")
        if eK > self.eK_critical and self.integrator_type.lower() != "brownian":
            raise EKExceedsError("Ek={1} exceeds {0}".format(self.eK_critical, eK))
        if (np.isnan(eK)) or (np.isnan(eP)):
            raise IntegrationFailError("Energy is NAN)")
        if check_fail:
            raise IntegrationFailError("Custom checks failed")

        dif = np.sqrt(np.mean(np.sum((newcoords - self.get_data()) ** 2, axis=1)))
        msg += "dr=%.2lf " % (dif,)
        self.data = coords
        msg += "t=%2.1lfps " % self.time  # (self.state.getTime() / simtk.unit.picosecond)
        msg += "kin=%.2lf pot=%.2lf " % (eK, eP)
        msg += "Rg=%.3lf " % self.RG()
        msg += "SPS=%.0lf " % (steps / (float(b - a)))

        if (
            self.integrator_type.lower() == "variablelangevin"
            or self.integrator_type.lower() == "variableverlet"
        ) and self.backend == "openmm":
            dt = self.integrator.getStepSize()
            msg += "dt=%.1lffs " % (dt / simtk.unit.femtosecond)
            mass = self.system.getParticleMass(0)
            dx = simtk.unit.sqrt(2.0 * eK * self.kT / mass) * dt
            msg += "dx=%.2lfpm " % (dx / simtk.unit.nanometer * 1000.0)

        logging.info(msg)

        result = {
            "pos": newcoords,
            "potentialEnergy": eP,
            "kineticEnergy": eK,
            "time": curtime,
            "block": self.block,
        }
        if get_velocities:
            if self.backend == "openmm":
                result["vel"] = self.state.getVelocities() / (
                    simtk.unit.nanometer / simtk.unit.picosecond
                )
            else:
                result["vel"] = self.state.particles.velocity
        result.update(save_extras)
        if save:
            for reporter in self.reporters:
                reporter.report("data", result)

        self.block += 1
        self.step += steps

        return result

    def print_stats(self):
        """Prints detailed statistics of a system.
        Will be run every 50 steps
        """
        state = self.context.getState(getPositions=True, getVelocities=True, getEnergy=True)

        eP = state.getPotentialEnergy()
        pos = np.array(state.getPositions() / simtk.unit.nanometer)
        bonds = np.sqrt(np.sum(np.diff(pos, axis=0) ** 2, axis=1))
        sbonds = np.sort(bonds)
        vel = state.getVelocities()
        mass = self.system.getParticleMass(1)
        vkT = np.array(vel / simtk.unit.sqrt(self.kT / mass), dtype=float)
        self.velocs = vkT
        EkPerParticle = 0.5 * np.sum(vkT**2, axis=1)

        cm = np.mean(pos, axis=0)
        centredPos = pos - cm[None, :]
        dists = np.sqrt(np.sum(centredPos**2, axis=1))
        per95 = np.percentile(dists, 95)
        den = (0.95 * self.N) / ((4.0 * np.pi * per95**3) / 3)
        per5 = np.percentile(dists, 5)
        den5 = (0.05 * self.N) / ((4.0 * np.pi * per5**3) / 3)
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        minmedmax = lambda x: (x.min(), np.median(x), x.mean(), x.max())

        print("\n Statistics: number of particles: %d\n" % (self.N,))
        print("Statistics for particle position")
        print("     mean position is: ", np.mean(pos, axis=0), "  Rg = ", self.RG())
        print("     median bond size is ", np.median(bonds))
        print(
            "     three shortest/longest (<10)/ bonds are ",
            sbonds[:3],
            "  ",
            sbonds[sbonds < 10][-3:],
        )
        if (sbonds > 10).sum() > 0:
            print("longest 10 bonds are", sbonds[-10:])

        print("     95 percentile of distance to center is:   ", per95)
        print("     density of closest 95% monomers is:   ", den)
        print("     density of the 5% closest to CoM monomers is:   ", den5)
        print("     min/median/mean/max coordinates are: ")
        print("     x: %.2lf, %.2lf, %.2lf, %.2lf" % minmedmax(x))
        print("     y: %.2lf, %.2lf, %.2lf, %.2lf" % minmedmax(y))
        print("     z: %.2lf, %.2lf, %.2lf, %.2lf" % minmedmax(z))
        print()
        print("Statistics for velocities:")
        print("     mean kinetic energy is: ", np.mean(EkPerParticle), "should be:", 1.5)
        print("     fastest particles are (in kT): ", np.sort(EkPerParticle)[-5:])

        print()
        print("Statistics for the system:")
        print("     Forces are: ", list(self.force_dict.keys()))
        print()
        print("Potential Energy Ep = ", eP / self.N / self.kT)

    def show(self, shifts=[0.0, 0.2, 0.4, 0.6, 0.8], scale="auto"):
        """shows system in rasmol by drawing spheres
        draws 4 spheres in between any two points (5 * N spheres total)
        """

        # if you want to change positions of the spheres along each segment,
        # change these numbers: e.g. [0,.1, .2 ...  .9] will draw 10 spheres,
        # and this will look better

        data = self.get_data()
        if len(data[0]) != 3:
            data = np.transpose(data)
        if len(data[0]) != 3:
            logging.error("wrong data!")
            return
        # determining the 95 percentile distance between particles,
        if scale == "auto":
            meandist = np.percentile(np.sqrt(np.sum(np.diff(data, axis=0) ** 2, axis=1)), 95)
            # rescaling the data, so that bonds are of the order of 1.
            # This is because rasmol spheres are of the fixed diameter.
            data /= meandist
        else:
            data /= scale

        if self.N > 1000:  # system is sufficiently large
            count = 0
            for _ in range(100):
                a, b = np.random.randint(0, self.N, 2)
                dist = np.sqrt(np.sum((data[a] - data[b]) ** 2))
                if dist < 1.3:
                    count += 1
            if count > 100:
                raise RuntimeError(
                    "Too many particles are close together. This will cause rasmol to choke"
                )

        rascript = tempfile.NamedTemporaryFile()
        # writing the rasmol script. Spacefill controls radius of the sphere.
        rascript.write(
            b"""wireframe off
        color temperature
        spacefill 100
        background white
        """
        )
        rascript.flush()

        # creating the array, linearly chanhing from -225 to 225
        # to serve as an array of colors
        colors = np.array([int((j * 450.0) / (len(data))) - 225 for j in range(len(data))])

        # creating spheres along the trajectory
        newData = np.zeros((len(data) * len(shifts) - (len(shifts) - 1), 4))
        for i in range(len(shifts)):
            newData[i : -1 : len(shifts), :3] = data[:-1] * shifts[i] + data[1:] * (1 - shifts[i])
            newData[i : -1 : len(shifts), 3] = colors[:-1]
        newData[-1, :3] = data[-1]
        newData[-1, 3] = colors[-1]

        towrite = tempfile.NamedTemporaryFile()
        towrite.write(("{:d}\n\n".format(int(len(newData))).encode("utf-8")))

        # number of atoms and a blank line after is a requirement of rasmol
        for i in newData:
            towrite.write(
                ("CA\t{:f}\t{:f}\t{:f}\t{:d}\n".format(i[0], i[1], i[2], int(i[3]))).encode(
                    "utf-8"
                )
            )

        towrite.flush()
        "TODO: rewrite using subprocess.popen"

        if os.name == "posix":  # if linux
            os.system("rasmol -xyz %s -script %s" % (towrite.name, rascript.name))
        else:  # if windows
            os.system("C:/RasWin/raswin.exe -xyz %s -script %s" % (towrite.name, rascript.name))

        rascript.close()
        towrite.close()
