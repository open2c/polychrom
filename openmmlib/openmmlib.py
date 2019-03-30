"""
Openmm-lib - a wrapper around Openmm to use with polymer simulations
====================================================================

Summary
-------

This is a wrapper above a GPU-assisted molecular dynamics package Openmm.

You can find extensive description of openmm classes here:
https://simtk.org/api_docs/openmm/api10/annotated.html

The main library is located in openmmlib.Simulation

A file polymerScalings.py has some utilities to calculate Rg(s) and Pc(s) for
polymer conformations in a fast and efficient way.

A file contactmaps.py has code to quickly calculate contacts within a polymer structure,
and organize them as contactmaps. It is used by polymerScalings.

A file pymol_show has utilities to visualize polymer conformations using pymol


Input/Output file format
------------------------

Polymer configuration is represented as a Nx3 numpy array of coordinates.
Start/end of chains/rings are not directly specified in the file,
and have to be added through method :py:func:`setLayout <Simulation.setLayout>`

Input file may have the simplistic format, described in txtToJoblib.py (first line with
number of particles, then N lines with three floats corresponding to x,y,z coordinates each).
Input file can also be any of the output files.

Output file format is a dictionary, saved with joblib.dump.
Nx3 data array is stored under the key "data".
The rest of the dictionary consists of metadata, describing details of the simulation.
This metadata is for informative purpose only, and is ignored by the code.


Implemented forces
------------------

All forces of the system are added by the code to the self.ForceDict dictionary.
After the force is added, the class (or user) can get it out of the
self.ForceDict and modify the force.
Once the simulation is started, all forces are automatically applied
and cannot be modified.

Two types of bond forces are harmonic bond force
and FENE-type bond as described in Grosberg papers.
Individual bonds can be added using :py:func:`addBond <Simulation.addBond>`,
while polymer bonds can be added using
:py:func:`addHarmonicPolymerBonds <Simulation.addHarmonicPolymerBonds>`, etc.

Write about polynomial repulsive forces 

Stiffness force can be harmonic, or the "special" Grosberg force, kept
for compatibility with the systems used in Grosberg forces

External forces include spherical confinement, cylindrical confinement,
attraction to "lamina"- surface of a cylinder, gravity, etc.

Information, printed on current step
------------------------------------

A sample line of information printed on each step looks like this:

minim  bl=5 (i)  pos[1]=[99.2 52.1 52.4] dr=0.54
kin=3.65 pot=3.44 Rg=107.312 SPS=144:

Let's go over it step by step.

minim - simulation name (sim -default, minim - energy minimization.
Other name can be provided in self.name).

bl=5 - name of a current block

(i) indicate that velocity reinitialization was done at this step.
You will simultaneously see that Ek is more than 2.4

pos[1] is a position of a first monomer

dr is sqrt(mean square displacement) of monomers, i.e.
how much did a monomer shift on average.

kin=3.65 pot=3.44  - energies: kinetic, potential

Rg=107.312 - current radius of gyration (size of the system)

SPS - steps per second


Functions
---------
Functions depend on each other, and have to be applied in certain groups

1.


:py:func:`load <Simulation.load>`  ---    Mandatory

:py:func:`setup <Simulation.setup>`  ---   Mandatory

:py:func:`setLayout <Simulation.setLayout>` --- Mandatory, after self.load()

:py:func:`saveFolder <Simulation.saveFolder>`  ---  Optional
(default is folder with the code)

2.

self.add___Force()  --- Use any set of forces

Then go all the addBond, and other modifications and tweaks of forces.

3.

Before running actual simulation, it is advised to resolve all possible
conflict by doing :py:func:`energyMinimization <Simulation.energyMinimization>`

4.


:py:func:`doBlock <Simulation.doBlock>`  --- the actual simulation

:py:func:`save <Simulation.save>` --- saves conformation


Frequently-used settings - where to specify them?
-------------------------------------------------

Select GPU ("0" or "1") - :py:func:`setup <Simulation.setup>`

Select chain/ring - :py:func:`setLayout <Simulation.setLayout>`

Select timestep or collision rate - :py:class:`Simulation`


-------------------------------------------------------------------------------

"""
# Licensed under the MIT license:
# http://www.opensource.org/licenses/mit-license.php

from __future__ import absolute_import, division, print_function
import numpy
import numpy as np
import pickle
import sys
import os
import time
import joblib
import tempfile
import warnings
from six import string_types

os.environ["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "") + ":/usr/local/cuda/lib64"

import simtk.openmm as openmm
import simtk.unit as units

nm = units.meter * 1e-9
fs = units.second * 1e-15
ps = units.second * 1e-12

class integrationFailError(Exception):
    pass

class eKExceedsError(Exception):
    pass 

class Simulation():
    """Base class for openmm simulations

    """
    def __init__(
        self, timestep=80, thermostat=0.001, temperature=300 * units.kelvin,
        verbose=False,
        name="sim",
        length_scale=1.0,
        mass_scale=1.0, 
        maxEk = 10 , 
        ):  # name to print out
        """

        Parameters
        ----------

        timestep : number
            timestep in femtoseconds. Default value is good.

        thermostat : number
            collision rate in inverse picoseconds. Default value is ok...

        temperature : simtk.units.quantity(units.kelvin), optional
            Temperature of the simulation. Devault value is 300 K.

        verbose : bool, optional
            If True, prints a lot of stuff in the command line.


        name : string, optional
            Name to be printed out as a first line of each block.
            Use it if you run simulations one after another
            and want to see what's going on.

        length_scale : float, optional
            The geometric scaling factor of the system.
            By default, length_scale=1.0 and harmonic bonds and repulsive
            forces have the scale of 1 nm.

        mass_scale : float, optional
            The scaling factor of the mass of the system.
           
        maxEk: float, optional
            raise error if kinetic energy in (kT/particle) exceeds this value 



        """

        self.name = name
        self.timestep = timestep * fs
        self.collisionRate = thermostat * (1 / ps)
        self.temperature = temperature
        self.verbose = verbose
        self.loaded = False  # check if the data is loaded
        self.forcesApplied = False
        self.folder = "."
        self.metadata = {}
        self.length_scale = length_scale
        self.mass_scale = mass_scale
        self.eKcritical = maxEk  # Max allowed kinetic energy
        self.nm = nm


    def setup(self, platform="CUDA", PBC=False, PBCbox=None, GPU="default",
              integrator="langevin", errorTol=None, precision="mixed"):
        """Sets up the important low-level parameters of the platform.
        Mandatory to run.

        Parameters
        ----------

        platform : string, optional
            Platform to use

        PBC : bool, optional
            Use periodic boundary conditions, default:False

        PBCbox : (float,float,float), optional
            Define size of the bounding box for PBC

        GPU : "0" or "1", optional
            Switch to another GPU. Mostly unnecessary.
            Machines with 1 GPU automatically select right GPU.
            Machines with 2 GPUs select GPU that is less used.

        integrator : "langevin", "variableLangevin", "verlet", "variableVerlet",
                     "brownian", optional Integrator to use
                     (see Openmm class reference)

        verbose : bool, optional
            Shout out loud about every change.

        errorTol : float, optional
            Error tolerance parameter for variableLangevin integrator
            Values of 0.03-0.1 are reasonable for "nice" simulation
            Simulations with strong forces may need 0.01 or less
        
        precision: str, optional (not recommended to change)
            mixed is optimal for most situations. 
            If you are using double precision, it will be slower by a factor of 10 or so. 
        

        """

        self.step = 0
        if PBC == True:
            self.metadata["PBC"] = True

        precision = precision.lower()
        if precision not in ["mixed", "single", "double"]:
            raise ValueError("Presision must be mixed, single or double")


        self.kB = units.BOLTZMANN_CONSTANT_kB * \
            units.AVOGADRO_CONSTANT_NA  # Boltzmann constant
        self.kT = self.kB * self.temperature  # thermal energy
        self.mass = 100.0 * units.amu * self.mass_scale
        # All masses are the same,
        # unless individual mass multipliers are specified in self.load()
        self.bondsForException = []
        self.mm = openmm
        self.conlen = 1. * nm * self.length_scale
        self.system = self.mm.System()
        self.PBC = PBC

        if self.PBC == True:  # if periodic boundary conditions
            if PBCbox is None:  # Automatically setting up PBC box
                data = self.getData()
                data -= numpy.min(data, axis=0)

                datasize = 1.1 * (2 + (numpy.max(self.getData(), axis=0) - \
                                       numpy.min(self.getData(), axis=0)))
                # size of the system plus some overhead

                self.SolventGridSize = (datasize / 1.1) - 2
                print("density is ", self.N / (datasize[0]
                    * datasize[1] * datasize[2]))
            else:
                PBCbox = numpy.array(PBCbox)
                datasize = PBCbox

            self.metadata["PBCbox"] = PBCbox
            self.system.setDefaultPeriodicBoxVectors([datasize[0], 0.,
                0.], [0., datasize[1], 0.], [0., 0., datasize[2]])
            self.BoxSizeReal = datasize

        self.GPU = str(GPU)  # setting default GPU
        properties = {}
        if self.GPU.lower() != "default":
            properties["DeviceIndex"] = str(GPU)
            properties["Precision"] = precision
        self.properties = properties

        if platform.lower() == "opencl":
            platformObject = self.mm.Platform.getPlatformByName('OpenCL')

        elif platform.lower() == "reference":
            platformObject = self.mm.Platform.getPlatformByName('Reference')

        elif platform.lower() == "cuda":
            platformObject = self.mm.Platform.getPlatformByName('CUDA')

        elif platform.lower() == "cpu":
            platformObject = self.mm.Platform.getPlatformByName('CPU')


        else:
            self.exit("\n!!!!!!!!!!unknown platform!!!!!!!!!!!!!!!\n")
        self.platform = platformObject

        self.forceDict = {}  # Dictionary to store forces

        self.integrator_type = integrator
        if isinstance(integrator, string_types):
            integrator = str(integrator)
            if integrator.lower() == "langevin":
                self.integrator = self.mm.LangevinIntegrator(self.temperature,
                    self.collisionRate, self.timestep)
            elif integrator.lower() == "variablelangevin":
                self.integrator = self.mm.VariableLangevinIntegrator(self.temperature,
                    self.collisionRate, errorTol)
            elif integrator.lower() == "verlet":
                self.integrator = self.mm.VariableVerletIntegrator(self.timestep)
            elif integrator.lower() == "variableverlet":
                self.integrator = self.mm.VariableVerletIntegrator(errorTol)

            elif integrator.lower() == 'brownian':
                self.integrator = self.mm.BrownianIntegrator(self.temperature,
                    self.collisionRate, self.timestep)
            else:
                print ('please select from "langevin", "variablelangevin", '
                       '"verlet", "variableVerlet", '
                       '"brownian" or provide an integrator object')
        else:
            self.integrator = integrator
            self.integrator_type = "UserDefined"

    def saveFolder(self, folder):
        """
        sets the folder where to save data.

        Parameters
        ----------
            folder : string
                folder to save the data

        """
        if os.path.exists(folder) == False:
            os.mkdir(folder)
        self.folder = folder

    def _exitProgram(self, line):
        print(line)
        print("--------------> Bye <---------------")
        exit()


    def setChains(self, chains=[(0, None, 0)]):
        """
        Sets configuration of the chains in the system. This information is
        later used by the chain-forming methods, e.g. addHarmonicPolymerBonds()
        and addStiffness().
        This method supersedes the less general getLayout().

        Parameters
        ----------

        chains: list of tuples
            The list of chains in format [(start, end, isRing)]. The particle
            range should be semi-open, i.e. a chain (0,3,0) links
            the particles 0, 1 and 2. If bool(isRing) is True than the first
            and the last particles of the chain are linked into a ring.
            The default value links all particles of the system into one chain.
        """

        if not hasattr(self, "N"):
            raise ValueError("Load the chain first, or provide chain length")

        self.chains = [i for i in chains]  # copy
        for i in range(len(self.chains)):
            start, end, isRing = self.chains[i]
            end = self.N if (end is None) else end
            self.chains[i] = (start, end, isRing)

    def getChains(self):
        "returns configuration of chains"
        return self.chains



    def load(self, filename,  # Input filename, or input data array
             center=False,  # Shift center of mass to zero?
             masses=None,
             ):
        """loads data from file.
        Accepts text files, joblib files or pure data as Nx3 or 3xN array
        
        Parameters
        ----------

        filename : joblib file, or text file name, or Nx3 or 3xN numpy array
            Input filename or array with data

        center : bool or "zero", optional
            Move center of mass to zero before starting the simulation
            if center == "zero", then center the data such as all positions are positive and start at zero

        masses : array
            Masses of each atom, measured in self.mass (default: 100 AMU,
            but could be modified by self.mass_scale)
        """

        if type(filename) == str:
            data = polymerutils.load(filename)
            
        else:
            data = filename

        data = numpy.asarray(data, float)

        if len(data) == 3:
            data = numpy.transpose(data)
        if len(data[0]) != 3:
            self._exitProgram("strange data file")
            
        if numpy.isnan(data).any():
            self._exitProgram("\n!!!!!!!!!!file contains NANS!!!!!!!!!\n")

        if center is True:
            av = numpy.mean(data, 0)
            data -= av

        if center == "zero":
            minvalue = numpy.min(data, 0)
            data -= minvalue

        self.setData(data)
        self.randomizeData()

        if self.verbose == True:
            print("center of mass is", numpy.mean(self.data, 0))
            print("Radius of gyration is,", self.RG())

        if masses == None:
            self.masses = [1. for _ in range(self.N)]
        else:
            self.masses = masses

        if not hasattr(self, "chains"):
            self.setChains()

    def save(self, filename=None, mode="joblib"):
        """Saves conformation plus some metadata.
        Metadata is not interpreted by this library, and is for your reference

        If data is saved to the .vtf format,
        the same filename should be specified each time.
        .vtf format is then viewable by VMD.

        Parameters:
            mode : str
                "h5dict" : use build-in h5dict storage
                "joblib" : use joblib storage
                "txt" : use text file storage

            filename : str or None
                Filename not needed for h5dict storage
                (use initStorage command) for joblib and txt,
                if filename not provided, it is created automatically.

        """
        mode = mode.lower()


        if filename is None:
            filename = "block%d.dat" % self.step
            filename = os.path.join(self.folder, filename)

        if mode == "joblib":
            self.metadata["data"] = self.getData()
            self.metadata["timestep"] = repr(self.timestep / fs)
            self.metadata["Collision rate"] = repr(self.collisionRate / ps)
            joblib.dump(self.metadata, filename=filename, compress=3)

        elif (mode == "txt") :
            polymerutils.save(data, filename, mode=mode)
        else:
            raise ValueError("Unknown mode : %s, use  joblib or txt" % mode)

            
    def getData(self):
        "Returns an Nx3 array of positions"
        return numpy.asarray(self.data / nm, dtype=np.float32)

    def getScaledData(self):
        """Returns data, scaled back to PBC box """
        if self.PBC != True:
            return self.getData()
        alldata = self.getData()
        boxsize = numpy.array(self.BoxSizeReal)
        mults = numpy.floor(alldata / boxsize[None, :])
        toRet = alldata - mults * boxsize[None, :]
        assert toRet.min() >= 0
        return toRet

    def addUnits(self, coords):
        coords = np.asarray(coords, dtype="float")
        coords = units.Quantity(coords, nm)
        return coords

    def setData(self, data):
        """Sets particle positions

        Parameters
        ----------

        data : Nx3 array-line
            Array of positions with distance ~1 between connected atoms.
        """
        data = numpy.asarray(data, dtype="float")
        self.data = units.Quantity(data, nm)
        self.N = len(self.data)
        if hasattr(self, "context"):
            self.initPositions()

    def randomizeData(self):
        """
        Runs automatically to offset data  (helps if data is integer based)
        """
        data = self.getData()
        data = data + numpy.random.randn(*data.shape) * 0.0001
        self.setData(data)

    def RG(self):
        """
        Returns
        -------

        Gyration ratius in units of length (bondlength).
        """
        data = self.getScaledData()
        data = data - np.mean(data, axis=0)[None,:]
        return numpy.sqrt(numpy.sum(numpy.var(numpy.array(data), 0)))

    def RMAX(self, percentile=None):
        """
        Returns
        -------
        Distance to the furthest from the origin particle.

        """
        data = self.getScaledData()
        dists = numpy.sqrt(numpy.sum((numpy.array(data)) ** 2, 1))
        if percentile == None:
            return numpy.max(dists)
        else:
            return numpy.percentile(dists, percentile)

    def dist(self, i, j):
        """
        Calculates distance between particles i and j
        """
        data = self.getData()
        dif = data[i] - data[j]
        return numpy.sqrt(sum(dif ** 2))


    def _initHarmonicBondForce(self):
        "Internal, inits harmonic forse for polymer and non-polymer bonds"
        if "HarmonicBondForce" not in list(self.forceDict.keys()):
            self.forceDict["HarmonicBondForce"] = self.mm.HarmonicBondForce()
        self.bondType = "Harmonic"


    def _initAbsBondForce(self):
        "inits abs(x) FENE bond force"
        if "AbsBondForce" not in list(self.forceDict.keys()):
            force = "(1. / ABSwiggle) * ABSunivK * "\
            "(sqrt((r-ABSr0 * ABSconlen)* "\
            " (r - ABSr0 * ABSconlen) + ABSa * ABSa) - ABSa)"

            bondforceAbs = self.mm.CustomBondForce(force)
            bondforceAbs.addPerBondParameter("ABSwiggle")
            bondforceAbs.addPerBondParameter("ABSr0")
            bondforceAbs.addGlobalParameter("ABSunivK", self.kT / self.conlen)
            bondforceAbs.addGlobalParameter("ABSa", 0.02 * self.conlen)
            bondforceAbs.addGlobalParameter("ABSconlen", self.conlen)
            self.forceDict["AbsBondForce"] = bondforceAbs



    def addBond(self,
                i, j,  # particles connected by bond
                bondWiggleDistance=0.2,
                # Flexibility of the bond,
                # measured in distance at which energy equals kT
                distance=None,  # Equilibrium length of the bond, default = self.length_scale
                bondType=None,  # Harmonic, Grosberg, ABS
                verbose=None):  # Set this to False or True to override self.verbose for this function
                # and don't want to contaminate output by 10000 messages
        """Adds bond between two particles, allows to specify parameters

        Parameters
        ----------

        i,j : int
            Particle numbers

        bondWiggleDistance : float
            Average displacement from the equilibrium bond distance

        bondType : "Harmonic" or "abs"
            Type of bond. Distance and bondWiggleDistance can be
            specified for harmonic bonds only

        verbose : bool
            Set this to False if you're in verbose mode and don't want to
            print "bond added" message

        """

        if not hasattr(self, "bondLengths"):
            self.bondLengths = []

        if verbose is None:
            verbose = self.verbose
        if (i >= self.N) or (j >= self.N):
            raise ValueError("\nCannot add bond with monomers %d,%d that"\
            "are beyound the polymer length %d" % (i, j, self.N))
        
        bondSize = float(bondWiggleDistance)
        
        if distance is None:
            distance = self.length_scale
        else:
            distance = self.length_scale * distance
        distance = float(distance)

        if not hasattr(self, "kbondScalingFactor"):  # caching kbondScalingFactor - performance improvement... 
            self.kbondScalingFactor = float((2 * self.kT / (self.conlen) ** 2) / (units.kilojoule_per_mole / nm ** 2))
        kbondScalingFactor = self.kbondScalingFactor  #... not to calculate it eevry time we add bond 
        # this will be an integer, so we don't have to deal with slow simtk.units every time we add a bond 

        if bondType is None:
            bondType = self.bondType

        if bondType.lower() == "harmonic":
            self._initHarmonicBondForce()
            kbond = kbondScalingFactor / (bondSize ** 2)  # using kbondScalingFactor because force accepts parameters with units
            self.forceDict["HarmonicBondForce"].addBond(int(i), int(j), float(distance), float(kbond))
            self.bondLengths.append([int(i), int(j), float(distance), float(bondSize)])
        elif bondType.lower() == "abs":
            self._initAbsBondForce()
            self.forceDict["AbsBondForce"].addBond(int(i), int(
                j), [float(bondWiggleDistance), float(distance)])  # force is initialized to accept floats already
            self.bondLengths.append([int(i), int(j), float(distance), float(bondSize)])
        else:
            self._exitProgram("Bond type not known")
        if verbose == True:
            print("%s bond added between %d,%d, wiggle %lf dist %lf" % (
                bondType, i, j, float(bondWiggleDistance), float(distance)))

    def addHarmonicPolymerBonds(self,
                                wiggleDist=0.05,
                                bondLength=1.0,
                                exceptBonds=True):
        """Adds harmonic bonds connecting polymer chains

        Parameters
        ----------

        wiggleDist : float
            Average displacement from the equilibrium bond distance
        bondLength : float
            The length of the bond
        exceptBonds : bool
            If True then do not calculate non-bonded forces between the
            particles connected by a bond. True by default.
        """


        for start, end, isRing in self.chains:
            for j in range(start, end - 1):
                self.addBond(j, j + 1, wiggleDist,
                    distance=bondLength,
                    bondType="Harmonic", verbose=False)
                if exceptBonds:
                    self.bondsForException.append((j, j + 1))

            if isRing:
                self.addBond(start, end - 1, wiggleDist,
                    distance=bondLength, bondType="Harmonic")
                if exceptBonds:
                    self.bondsForException.append((start, end - 1))
                if self.verbose == True:
                    print("ring bond added", start, end - 1)

        self.metadata["HarmonicPolymerBonds"] = repr(
            {"wiggleDist": wiggleDist, 'bondLength':bondLength})


    def addStiffness(self, k=1.5):
        """Adds harmonic angle bonds. k specifies energy in kT at one radian
        If k is an array, it has to be of the length N.
        Xth value then specifies stiffness of the angle centered at
        monomer number X.
        Values for ends of the chain will be simply ignored.

        Parameters
        ----------

        k : float or list of length N
            Stiffness of the bond.
            If list, then determines stiffness of the bond at monomer i.
            Potential is k * alpha^2 * 0.5 * kT
        """
        try:
            k[0]
        except:
            k = numpy.zeros(self.N, float) + k
        stiffForce = self.mm.CustomAngleForce(
            "kT*angK * (theta - 3.141592) * (theta - 3.141592) * (0.5)")
        self.forceDict["AngleForce"] = stiffForce
        for start, end, isRing in self.chains:
            for j in range(start + 1, end - 1):
                stiffForce.addAngle(j - 1, j, j + 1, [float(k[j])])
            if isRing:
                stiffForce.addAngle(int(end - 2),int( end - 1), int(start), [float(k[end - 1])  ])
                stiffForce.addAngle(int(end - 1),int( start), int(start + 1), [float(k[start])  ])

        stiffForce.addGlobalParameter("kT", self.kT)
        stiffForce.addPerAngleParameter("angK")
        self.metadata["AngleForce"] = repr({"stiffness": k})



    def addPolynomialRepulsiveForce(self, trunc=3.0, radiusMult=1.):
        """This is a simple polynomial repulsive potential. It has the value
        of `trunc` at zero, stays flat until 0.6-0.7 and then drops to zero
        together with its first derivative at r=1.0.

        Parameters
        ----------

        trunc : float
            the energy value around r=0

        """
        radius = self.conlen * radiusMult
        self.metadata["PolynomialRepulsiveForce"] = repr({"trunc": trunc})
        nbCutOffDist = radius
        repul_energy = (
            "rsc12 * (rsc2 - 1.0) * REPe / REPemin + REPe;"
            "rsc12 = rsc4 * rsc4 * rsc4;"
            "rsc4 = rsc2 * rsc2;"
            "rsc2 = rsc * rsc;"
            "rsc = r / REPsigma * REPrmin;")
        self.forceDict["Nonbonded"] = self.mm.CustomNonbondedForce(
            repul_energy)
        repulforceGr = self.forceDict["Nonbonded"]

        repulforceGr.addGlobalParameter('REPe', trunc * self.kT)
        repulforceGr.addGlobalParameter('REPsigma', radius)
        # Coefficients for x^8*(x*x-1)
        # repulforceGr.addGlobalParameter('REPemin', 256.0 / 3125.0)
        # repulforceGr.addGlobalParameter('REPrmin', 2.0 / np.sqrt(5.0))
        # Coefficients for x^12*(x*x-1)
        repulforceGr.addGlobalParameter('REPemin', 46656.0 / 823543.0)
        repulforceGr.addGlobalParameter('REPrmin', np.sqrt(6.0 / 7.0))
        for _ in range(self.N):
            repulforceGr.addParticle(())

        repulforceGr.setCutoffDistance(nbCutOffDist)

    def addSmoothSquareWellForce(self,
        repulsionEnergy=3.0, repulsionRadius=1.,
        attractionEnergy=0.5, attractionRadius=2.0,
        ):
        """
        This is a simple and fast polynomial force that looks like a smoothed
        version of the square-well potential. The energy equals `repulsionEnergy`
        around r=0, stays flat until 0.6-0.7, then drops to zero together
        with its first derivative at r=1.0. After that it drop down to
        `attractionEnergy` and gets back to zero at r=`attractionRadius`.

        The energy function is based on polynomials of 12th power. Both the
        function and its first derivative is continuous everywhere within its
        domain and they both get to zero at the boundary.

        Parameters
        ----------

        repulsionEnergy: float
            the heigth of the repulsive part of the potential.
            E(0) = `repulsionEnergy`
        repulsionRadius: float
            the radius of the repulsive part of the potential.
            E(`repulsionRadius`) = 0,
            E'(`repulsionRadius`) = 0
        attractionEnergy: float
            the depth of the attractive part of the potential.
            E(`repulsionRadius`/2 + `attractionRadius`/2) = `attractionEnergy`
        attractionEnergy: float
            the maximal range of the attractive part of the potential.

        """
        nbCutOffDist = self.conlen * attractionRadius
        self.metadata["PolynomialAttractiveForce"] = repr({"trunc": repulsionEnergy})
        energy = (
            "step(REPsigma - r) * Erep + step(r - REPsigma) * Eattr;"
            ""
            "Erep = rsc12 * (rsc2 - 1.0) * REPe / emin12 + REPe;"
            "rsc12 = rsc4 * rsc4 * rsc4;"
            "rsc4 = rsc2 * rsc2;"
            "rsc2 = rsc * rsc;"
            "rsc = r / REPsigma * rmin12;"
            ""
            "Eattr = - rshft12 * (rshft2 - 1.0) * ATTRe / emin12 - ATTRe;"
            "rshft12 = rshft4 * rshft4 * rshft4;"
            "rshft4 = rshft2 * rshft2;"
            "rshft2 = rshft * rshft;"
            "rshft = (r - REPsigma - ATTRdelta) / ATTRdelta * rmin12"

            )
        self.forceDict["Nonbonded"] = self.mm.CustomNonbondedForce(
            energy)
        repulforceGr = self.forceDict["Nonbonded"]

        repulforceGr.addGlobalParameter('REPe', repulsionEnergy * self.kT)
        repulforceGr.addGlobalParameter('REPsigma', repulsionRadius * self.conlen)

        repulforceGr.addGlobalParameter('ATTRe', attractionEnergy * self.kT)
        repulforceGr.addGlobalParameter('ATTRdelta',
            self.conlen * (attractionRadius - repulsionRadius) / 2.0)
        # Coefficients for the minimum of x^12*(x*x-1)
        repulforceGr.addGlobalParameter('emin12', 46656.0 / 823543.0)
        repulforceGr.addGlobalParameter('rmin12', np.sqrt(6.0 / 7.0))

        for _ in range(self.N):
            repulforceGr.addParticle(())

        repulforceGr.setCutoffDistance(nbCutOffDist)

    def addSelectiveSSWForce(self,
        stickyParticlesIdxs,
        extraHardParticlesIdxs,
        repulsionEnergy=3.0,
        repulsionRadius=1.,
        attractionEnergy=3.0,
        attractionRadius=1.5,
        selectiveRepulsionEnergy=20.0,
        selectiveAttractionEnergy=1.0):
        """
        This is a simple and fast polynomial force that looks like a smoothed
        version of the square-well potential. The energy equals `repulsionEnergy`
        around r=0, stays flat until 0.6-0.7, then drops to zero together
        with its first derivative at r=1.0. After that it drop down to
        `attractionEnergy` and gets back to zero at r=`attractionRadius`.

        The energy function is based on polynomials of 12th power. Both the
        function and its first derivative is continuous everywhere within its
        domain and they both get to zero at the boundary.

        This is a tunable version of SSW:
        a) You can specify the set of "sticky" particles. The sticky particles
        are attracted only to other sticky particles.
        b) You can select a subset of particles and make them "extra hard".
        
        This force was used two-ways. First was to make a small subset of particles very sticky. 
        In that case, it is advantageous to make the sticky particles and their neighbours
        "extra hard" and thus prevent the system from collapsing.
        
        Another useage is to induce phase separation by making all B monomers sticky. In that case, 
        extraHard particles may not be needed at all, because the system would not collapse on itself. 
       

        Parameters
        ----------

        stickyParticlesIdxs: list of int
            the list of indices of the "sticky" particles. The sticky particles
            are attracted to each other with extra `selectiveAttractionEnergy`
        extraHardParticlesIdxs : list of int
            the list of indices of the "extra hard" particles. The extra hard
            particles repel all other particles with extra
            `selectiveRepulsionEnergy`
        repulsionEnergy: float
            the heigth of the repulsive part of the potential.
            E(0) = `repulsionEnergy`
        repulsionRadius: float
            the radius of the repulsive part of the potential.
            E(`repulsionRadius`) = 0,
            E'(`repulsionRadius`) = 0
        attractionEnergy: float
            the depth of the attractive part of the potential.
            E(`repulsionRadius`/2 + `attractionRadius`/2) = `attractionEnergy`
        attractionRadius: float
            the maximal range of the attractive part of the potential.
        selectiveRepulsionEnergy: float
            the EXTRA repulsion energy applied to the "extra hard" particles
        selectiveAttractionEnergy: float
            the EXTRA attraction energy applied to the "sticky" particles
        """

        energy = (
            "step(REPsigma - r) * Erep + step(r - REPsigma) * Eattr;"
            ""
            "Erep = rsc12 * (rsc2 - 1.0) * REPeTot / emin12 + REPeTot;"  # + ESlide;"
            "REPeTot = REPe + (ExtraHard1 + ExtraHard2) * REPeAdd;"
            "rsc12 = rsc4 * rsc4 * rsc4;"
            "rsc4 = rsc2 * rsc2;"
            "rsc2 = rsc * rsc;"
            "rsc = r / REPsigma * rmin12;"
            ""
            "Eattr = - rshft12 * (rshft2 - 1.0) * ATTReTot / emin12 - ATTReTot;"
            "ATTReTot = ATTRe + min(Sticky1, Sticky2) * ATTReAdd;"
            "rshft12 = rshft4 * rshft4 * rshft4;"
            "rshft4 = rshft2 * rshft2;"
            "rshft2 = rshft * rshft;"
            "rshft = (r - REPsigma - ATTRdelta) / ATTRdelta * rmin12;"
            ""
            )

        if selectiveRepulsionEnergy == float('inf'):
            energy += (
            "REPeAdd = 4 * ((REPsigma / (2.0^(1.0/6.0)) / r)^12 - (REPsigma / (2.0^(1.0/6.0)) / r)^6) + 1;"
            )

        repulforceGr = self.mm.CustomNonbondedForce(energy)

        repulforceGr.setCutoffDistance(attractionRadius * self.conlen)

        self.metadata["PolynomialAttractiveForce"] = {"trunc": repulsionEnergy}

        repulforceGr.addGlobalParameter('REPe', repulsionEnergy * self.kT)
        if selectiveRepulsionEnergy != float('inf'):
            repulforceGr.addGlobalParameter('REPeAdd', selectiveRepulsionEnergy * self.kT)
        repulforceGr.addGlobalParameter('REPsigma', repulsionRadius * self.conlen)

        repulforceGr.addGlobalParameter('ATTRe', attractionEnergy * self.kT)
        repulforceGr.addGlobalParameter('ATTReAdd', selectiveAttractionEnergy * self.kT)
        repulforceGr.addGlobalParameter('ATTRdelta',
            self.conlen * (attractionRadius - repulsionRadius) / 2.0)

        # Coefficients for x^12*(x*x-1)
        repulforceGr.addGlobalParameter('emin12', 46656.0 / 823543.0)
        repulforceGr.addGlobalParameter('rmin12', np.sqrt(6.0 / 7.0))

        repulforceGr.addPerParticleParameter("Sticky")
        repulforceGr.addPerParticleParameter("ExtraHard")
        counts = np.bincount(stickyParticlesIdxs, minlength=self.N)

        for i in range(self.N):
            repulforceGr.addParticle(
                (float(counts[i]),
                 float(i in extraHardParticlesIdxs)))

        self.forceDict["Nonbonded"] = repulforceGr



    def addCylindricalConfinement(self, r, bottom=None, k=0.1, top=9999):
        "As it says."

        if bottom == True:
            warnings.warn(DeprecationWarning(
                "Use bottom=0 instead of bottom = True! "))
            bottom = 0

        self.metadata["CylindricalConfinement"] = repr({"r": r,
            "bottom": bottom, "k": k, "top": top})

        if bottom is not None:
            extforce2 = self.mm.CustomExternalForce(
                "step(r-CYLaa) * CYLkb * (sqrt((r-CYLaa)*(r-CYLaa) + CYLt*CYLt) - CYLt)"
                "+ step(-z + CYLbot) * CYLkb * (sqrt((z - CYLbot)^2 + CYLt^2) - CYLt) "
                "+ step(z - CYLtop) * CYLkb * (sqrt((z - CYLtop)^2 + CYLt^2) - CYLt);"
                "r = sqrt(x^2 + y^2 + CYLtt^2)")
        else:
            extforce2 = self.mm.CustomExternalForce(
                "step(r-CYLaa) * CYLkb * (sqrt((r-CYLaa)*(r-CYLaa) + CYLt*CYLt) - CYLt);"
                "r = sqrt(x^2 + y^2 + CYLtt^2)")

        self.forceDict["CylindricalConfinement"] = extforce2
        for i in range(self.N):
            extforce2.addParticle(i, [])
        extforce2.addGlobalParameter("CYLkb", k * self.kT / nm)
        extforce2.addGlobalParameter("CYLtop", top * self.conlen)
        if bottom is not None:
            extforce2.addGlobalParameter("CYLbot", bottom * self.conlen)
        extforce2.addGlobalParameter("CYLkt", self.kT)
        extforce2.addGlobalParameter("CYLweired", nm)
        extforce2.addGlobalParameter("CYLaa", (r - 1. / k) * nm)
        extforce2.addGlobalParameter("CYLt", (1. / (10 * k)) * nm)
        extforce2.addGlobalParameter("CYLtt", 0.01 * nm)

    def addSphericalConfinement(self,
                r="density",  # radius... by default uses certain density
                k=5.,  # How steep the walls are
                density=.3):  # target density, measured in particles
                                # per cubic nanometer (bond size is 1 nm)
        """Constrain particles to be within a sphere.
        With no parameters creates sphere with density .3

        Parameters
        ----------
        r : float or "density", optional
            Radius of confining sphere. If "density" requires density,
            or assumes density = .3
        k : float, optional
            Steepness of the confining potential, in kT/nm
        density : float, optional, <1
            Density for autodetection of confining radius.
            Density is calculated in particles per nm^3,
            i.e. at density 1 each sphere has a 1x1x1 cube.
        """
        self.metadata["SphericalConfinement"] = repr({"r": r, "k": k,
            "density": density})

        spherForce = self.mm.CustomExternalForce(
            "step(r-SPHaa) * SPHkb * (sqrt((r-SPHaa)*(r-SPHaa) + SPHt*SPHt) - SPHt) "
            ";r = sqrt(x^2 + y^2 + z^2 + SPHtt^2)")
        self.forceDict["SphericalConfinement"] = spherForce

        for i in range(self.N):
            spherForce.addParticle(i, [])
        if r == "density":
            r = (3 * self.N / (4 * 3.141592 * density)) ** (1 / 3.)

        self.sphericalConfinementRadius = r
        if self.verbose == True:
            print("Spherical confinement with radius = %lf" % r)
        # assigning parameters of the force
        spherForce.addGlobalParameter("SPHkb", k * self.kT / nm)
        spherForce.addGlobalParameter("SPHaa", (r - 1. / k) * nm)
        spherForce.addGlobalParameter("SPHt", (1. / k) * nm / 10.)
        spherForce.addGlobalParameter("SPHtt", 0.01 * nm)
        return r




    def tetherParticles(self, particles, k=30, positions="current"):
        """tethers particles in the 'particles' array.
        Increase k to tether them stronger, but watch the system!

        Parameters
        ----------

        particles : list of ints
            List of particles to be tethered (fixed in space)
        k : int, optional
            Steepness of the tethering potential.
            Values >30 will require decreasing potential,
            but will make tethering rock solid.
        """
        self.metadata["TetheredParticles"] = repr({"particles": particles, "k": k})
        if "Tethering Force" not in self.forceDict:
            tetherForce = self.mm.CustomExternalForce(
              " TETHkb * ((x - TETHx0)^2 + (y - TETHy0)^2 + (z - TETHz0)^2)")
            self.forceDict["Tethering Force"] = tetherForce
        else:
            tetherForce = self.forceDict["Tethering Force"]

        # assigning parameters of the force
        tetherForce.addGlobalParameter("TETHkb", k * self.kT / nm)
        tetherForce.addPerParticleParameter("TETHx0")
        tetherForce.addPerParticleParameter("TETHy0")
        tetherForce.addPerParticleParameter("TETHz0")
        if positions == "current":
            positions = [self.data[i] for i in particles]
        else:
            positions = self.addUnits(positions)

        for i, pos in zip(particles, positions):  # adding all the particles on which force acts
            i = int(i)
            tetherForce.addParticle(i, list(pos))
            if self.verbose == True:
                print("particle %d tethered! " % i)

    def addPullForce(self, particles, forces):
        """
        adds force pulling on each particle
        particles: list of particle indices
        forces: list of forces [[f0x,f0y,f0z],[f1x,f1y,f1z], ...]
        if there are fewer forces than particles forces are padded with forces[-1]
        """
        import itertools
        pullForce = self.mm.CustomExternalForce(
            "PULLx * x + PULLy * y + PULLz * z")
        pullForce.addPerParticleParameter("PULLx")
        pullForce.addPerParticleParameter("PULLy")
        pullForce.addPerParticleParameter("PULLz")
        for num, force in itertools.zip_longest(particles, forces, fillvalue=forces[-1]):
            force = [float(i) * (self.kT / self.conlen) for i in force]
            pullForce.addParticle(num, force)
        self.forceDict["PullForce"] = pullForce

        
    def addCenterOfMassRemover(self):
        remover = self.mm.CMMotionRemover(10)
        self.forceDict["CoM_Remover"] = remover
        
    def addAndersenThermostat(self):
        andersenThermo = self.mm.AndersenThermostat(
            self.temperature, self.collisionRate)
        self.forceDict["AndersenThermostat"] = andersenThermo
        
    def _loadParticles(self):
        if not hasattr(self, "system"):
            return
        if not self.loaded:
            for mass in self.masses:
                self.system.addParticle(self.mass * mass)
            if self.verbose == True:
                print("%d particles loaded" % self.N)
            self.loaded = True
        
        
    def _applyForces(self):
        """Adds all particles to the system.
        Then applies all the forces in the forcedict.
        Forces should not be modified after that, unless you do it carefully
        (see openmm reference)."""

        if self.forcesApplied == True:
            return
        self._loadParticles()

        exc = self.bondsForException
        print("Number of exceptions:", len(exc))

        if len(exc) > 0:
            exc = numpy.array(exc)
            exc = numpy.sort(exc, axis=1)
            exc = [tuple(i) for i in exc]
            exc = list(set(exc))  # only unique pairs are left

        for i in list(self.forceDict.keys()):  # Adding exceptions
            force = self.forceDict[i]
            if hasattr(force, "addException"):
                print('Add exceptions for {0} force'.format(i))
                for pair in exc:
                    force.addException(int(pair[0]),
                        int(pair[1]), 0, 0, 0, True)
            elif hasattr(force, "addExclusion"):
                print('Add exclusions for {0} force'.format(i))
                for pair in exc:
                    # force.addExclusion(*pair)
                    force.addExclusion(int(pair[0]), int(pair[1]))

            if hasattr(force, "CutoffNonPeriodic") and hasattr(
                                                    force, "CutoffPeriodic"):
                if self.PBC:
                    force.setNonbondedMethod(force.CutoffPeriodic)
                    print("Using periodic boundary conditions!!!!")
                else:
                    force.setNonbondedMethod(force.CutoffNonPeriodic)
            print("adding force ", i, self.system.addForce(self.forceDict[i]))

        self.context = self.mm.Context(self.system, self.integrator, self.platform, self.properties)
        self.initPositions()
        self.initVelocities()
        self.forcesApplied = True

    def initVelocities(self, mult=1):
        """Initializes particles velocities

        Parameters
        ----------
        mult : float, optional
            Multiply velosities by this. Is good for a cold/hot start.
        """
        try:
            self.context
        except:
            raise ValueError("No context, cannot set velocs."\
                             "Initialize context before that")

        sigma = units.sqrt(self.kT / self.system.getParticleMass(
            1))  # calculating mean velocity
        velocs = units.Quantity(mult * numpy.random.normal(
            size=(self.N, 3)), units.meter) * (sigma / units.meter)
        # Guide to simtk.unit: 1. Always use units.quantity.
        # 2. Avoid dimensionless shit.
        # 3. If you have to, create fake units, as done here with meters
        self.context.setVelocities(velocs)

    def initPositions(self):
        """Sends particle coordinates to OpenMM system.
        If system has exploded, this is
         used in the code to reset coordinates. """

        print("Positions... ")
        try:
            self.context
        except:
            raise ValueError("No context, cannot set positions."\
                             " Initialize context before that")

        self.context.setPositions(self.data)
        print(" loaded!")
        state = self.context.getState(getPositions=True, getEnergy=True)
            # get state of a system: positions, energies
        eP = state.getPotentialEnergy() / self.N / self.kT
        print("potential energy is %lf" % eP)

    def reinitialize(self, mult=1):
        """Reinitializes the OpenMM context object.
        This should be called if low-level parameters,
        such as forces, have changed.

        Parameters
        ----------
        mult : float, optional
            mult to be passed to
             :py:func:'initVelocities <Simulation.initVelocities>'
        """
        self.context.reinitialize()
        self.initPositions()
        self.initVelocities(mult)

    def localEnergyMinimization(self, tolerance=0.3, maxIterations=0):
        "A wrapper to the build-in OpenMM Local Energy Minimization"
        print("Performing local energy minimization")

        self._applyForces()
        oldName = self.name
        self.name = "minim"

        self.state = self.context.getState(getPositions=False,
                                           getEnergy=True)
        eK = (self.state.getKineticEnergy() / self.N / self.kT)
        eP = self.state.getPotentialEnergy() / self.N / self.kT
        locTime = self.state.getTime()
        print("before minimization eK={0}, eP={1}, time={2}".format(eK, eP, locTime))

        self.mm.LocalEnergyMinimizer.minimize(
            self.context, tolerance, maxIterations)

        self.state = self.context.getState(getPositions=False,
                                           getEnergy=True)
        eK = (self.state.getKineticEnergy() / self.N / self.kT)
        eP = self.state.getPotentialEnergy() / self.N / self.kT
        locTime = self.state.getTime()
        print("after minimization eK={0}, eP={1}, time={2}".format(eK, eP, locTime))

        self.name = oldName



    def doBlock(self, steps=None, increment=True,  reinitialize=True, maxIter=0, checkFunctions=[]):
        """performs one block of simulations, doing steps timesteps,
        or steps_per_block if not specified.

        Parameters
        ----------

        steps : int or None
            Number of timesteps to perform.
        increment : bool, optional
            If true, will not increment self.steps counter
        """

        if self.forcesApplied == False:
            if self.verbose:
                print("applying forces")
                sys.stdout.flush()
            self._applyForces()
            self.forcesApplied = True
            
        if increment == True:
            self.step += 1

        if (increment == True) and ((self.step % 50) == 25):
            self.printStats()

        print("bl=%d" % (self.step), end=' ')
        sys.stdout.flush()
        if self.verbose:
            print()
            sys.stdout.flush()

        a = time.time()
        self.integrator.step(steps)  # integrate!

        # get state of a system: positions, energies
        self.state = self.context.getState(getPositions=True,
                                           getEnergy=True)

        b = time.time()
        coords = self.state.getPositions(asNumpy=True)
        newcoords = coords / nm

        # calculate energies in KT/particle
        eK = (self.state.getKineticEnergy() / self.N / self.kT)
        eP = self.state.getPotentialEnergy() / self.N / self.kT

        print("pos[1]=[%.1lf %.1lf %.1lf]" % tuple(newcoords[0]), end=' ')


        checkFail = False
        for checkFunction in checkFunctions:
            if not checkFunction(newcoords):
                checkFail = True

        if ((numpy.isnan(newcoords).any()):
            raise integrationFailError("Coordinates are NANs")
        if (eK > self.eKcritical):
            raise eKExceedsError("Ek exceeds {0}".format(eKcritical))
        if  (numpy.isnan(eK)) or (numpy.isnan(eP))):
            raise integrationFailError("Energy is NAN)")
        if checkFail:
            raise integrationFailError("Custom checks failed")

        dif = numpy.sqrt(numpy.mean(numpy.sum((newcoords -
            self.getData()) ** 2, axis=1)))
        print("dr=%.2lf" % (dif,), end=' ')
        self.data = coords
        print("t=%2.1lfps" % (self.state.getTime() / ps), end=' ')
        print("kin=%.2lf pot=%.2lf" % (eK,
            eP), "Rg=%.3lf" % self.RG(), end=' ')
        print("SPS=%.0lf" % (steps / (float(b - a))), end=' ')

        if (self.integrator_type.lower() == 'variablelangevin'
            or self.integrator_type.lower() == 'variableverlet'):
            dt = self.integrator.getStepSize()
            print('dt=%.1lffs' % (dt / fs), end=' ')
            mass = self.system.getParticleMass(1)
            dx = (units.sqrt(2.0 * eK * self.kT / mass) * dt)
            print('dx=%.2lfpm' % (dx / nm * 1000.0), end=' ')

        print("")

        return {"Ep":eP, "Ek":eK}


    def printStats(self):
        """Prints detailed statistics of a system.
        Will be run every 50 steps
        """
        state = self.context.getState(getPositions=True,
            getVelocities=True, getEnergy=True)

        eP = state.getPotentialEnergy()
        pos = numpy.array(state.getPositions() / nm)
        bonds = numpy.sqrt(numpy.sum(numpy.diff(pos, axis=0) ** 2, axis=1))
        sbonds = numpy.sort(bonds)
        vel = state.getVelocities()
        mass = self.system.getParticleMass(1)
        vkT = numpy.array(vel / units.sqrt(self.kT / mass), dtype=float)
        self.velocs = vkT
        EkPerParticle = 0.5 * numpy.sum(vkT ** 2, axis=1)

        cm = numpy.mean(pos, axis=0)
        centredPos = pos - cm[None, :]
        dists = numpy.sqrt(numpy.sum(centredPos ** 2, axis=1))
        per95 = numpy.percentile(dists, 95)
        den = (0.95 * self.N) / ((4. * numpy.pi * per95 ** 3) / 3)
        per5 = numpy.percentile(dists, 5)
        den5 = (0.05 * self.N) / ((4. * numpy.pi * per5 ** 3) / 3)
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        minmedmax = lambda x: (x.min(), numpy.median(x), x.mean(), x.max())

        print()
        print("Statistics for the simulation %s, number of particles: %d, "\
        " number of chains: %d" % (
            self.name, self.N, len(self.chains)))
        print()
        print("Statistics for particle position")
        print("     mean position is: ", numpy.mean(
            pos, axis=0), "  Rg = ", self.RG())
        print("     median bond size is ", numpy.median(bonds))
        print("     three shortest/longest (<10)/ bonds are ", sbonds[
            :3], "  ", sbonds[sbonds < 10][-3:])
        if (sbonds > 10).sum() > 0:
            print("longest 10 bonds are", sbonds[-10:])

        print("     95 percentile of distance to center is:   ", per95)
        print("     density of closest 95% monomers is:   ", den)
        print("     density of the core monomers is:   ", den5)
        print("     min/median/mean/max coordinates are: ")
        print("     x: %.2lf, %.2lf, %.2lf, %.2lf" % minmedmax(x))
        print("     y: %.2lf, %.2lf, %.2lf, %.2lf" % minmedmax(y))
        print("     z: %.2lf, %.2lf, %.2lf, %.2lf" % minmedmax(z))
        print()
        print("Statistics for velocities:")
        print("     mean kinetic energy is: ", numpy.mean(
            EkPerParticle), "should be:", 1.5)
        print("     fastest particles are (in kT): ", numpy.sort(
            EkPerParticle)[-5:])

        print()
        print("Statistics for the system:")
        print("     Forces are: ", list(self.forceDict.keys()))
        print("     Number of exceptions:  ", len(self.bondsForException))
        print()
        print("Potential Energy Ep = ", eP / self.N / self.kT)

    def show(self, shifts=[0., 0.2, 0.4, 0.6, 0.8], scale="auto"):
        """shows system in rasmol by drawing spheres
        draws 4 spheres in between any two points (5 * N spheres total)
        """

        # if you want to change positions of the spheres along each segment,
        # change these numbers: e.g. [0,.1, .2 ...  .9] will draw 10 spheres,
        # and this will look better

        data = self.getData()
        if len(data[0]) != 3:
            data = numpy.transpose(data)
        if len(data[0]) != 3:
            print("wrong data!")
            return
        # determining the 95 percentile distance between particles,
        if scale == "auto":
            meandist = numpy.percentile(numpy.sqrt(
                numpy.sum(numpy.diff(data, axis=0) ** 2, axis=1)), 95)
            # rescaling the data, so that bonds are of the order of 1.
            # This is because rasmol spheres are of the fixed diameter.
            data /= meandist
        else:
            data /= scale

        if self.N > 1000:  # system is sufficiently large
            count = 0
            for _ in range(100):
                a, b = numpy.random.randint(0, self.N, 2)
                dist = numpy.sqrt(numpy.sum((data[a] - data[b]) ** 2))
                if dist < 1.3:
                    count += 1
            if count > 100:
                raise RuntimeError(
                    "Too many particles are close together. "\
                    "This will cause rasmol to choke")

        rascript = tempfile.NamedTemporaryFile()
        # writing the rasmol script. Spacefill controls radius of the sphere.
        rascript.write(b"""wireframe off
        color temperature
        spacefill 100
        background white
        """)
        rascript.flush()

        # creating the array, linearly chanhing from -225 to 225
        # to serve as an array of colors
        colors = numpy.array([int((j * 450.) / (len(data))) -
            225 for j in range(len(data))])

        # creating spheres along the trajectory
        newData = numpy.zeros(
            (len(data) * len(shifts) - (len(shifts) - 1), 4))
        for i in range(len(shifts)):
            newData[i:-1:len(shifts), :3] = data[:-1] * shifts[
                i] + data[1:] * (1 - shifts[i])
            newData[i:-1:len(shifts), 3] = colors[:-1]
        newData[-1, :3] = data[-1]
        newData[-1, 3] = colors[-1]

        towrite = tempfile.NamedTemporaryFile()
        towrite.write( ((  ("{:d}\n\n".format(int(len(newData))).encode('utf-8'))   )))

        # number of atoms and a blank line after is a requirement of rasmol
        for i in newData:
            towrite.write(   ("CA\t{:f}\t{:f}\t{:f}\t{:d}\n".format(i[0],i[1],i[2],int(i[3]) )).encode('utf-8')     )

        towrite.flush()
        "TODO: rewrite using subprocess.popen"

        if os.name == "posix":  # if linux
            os.system("rasmol -xyz %s -script %s" % (
                towrite.name, rascript.name))
        else:  # if windows
            os.system("C:/RasWin/raswin.exe -xyz %s -script %s" % (
                                        towrite.name, rascript.name))

        rascript.close()
        towrite.close()

