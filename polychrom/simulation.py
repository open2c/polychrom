# Licensed under the MIT license:
# http://www.opensource.org/licenses/mit-license.php

from __future__ import absolute_import, division, print_function
import numpy as np
import sys
import os
import time
import tempfile
import warnings
import logging

from six import string_types

from collections.abc import Iterable

import simtk.openmm as openmm
import simtk.unit as units

from . import forces

nm = units.meter * 1e-9
fs = units.second * 1e-15
ps = units.second * 1e-12

logging.basicConfig(level=logging.INFO)

class integrationFailError(Exception):
    pass


class eKExceedsError(Exception):
    pass 


class Simulation():

    def __init__(self, **kwargs):
        """Base class for openmm simulations

        All parameters here are floats. Units specified in a parameter. 

        Parameters
        ----------
        
        N : int
            number of particles 
        
        error_tol : float, optional
            Error tolerance parameter for variableLangevin integrator
            Values of 0.03-0.1 are reasonable for "nice" simulation
            Simulations with strong forces may need 0.01 or less


        timestep : number
            timestep in femtoseconds. Mandatory for non-variable integrators. Value of 70-80 are appropriate

        collision_rate : number
            collision rate in inverse picoseconds. values of 0.01 or 0.05 are often used. 
            Consult with lab members on values. 

        PBC : bool, optional
            Use periodic boundary conditions, default:False

        PBCbox : (float,float,float), optional
            Define size of the bounding box for PBC

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


        name : string, optional
            Name to be printed out as a first line of each block.
            Use it if you run simulations one after another
            and want to see what's going on.

        length_scale : float, optional
            The geometric scaling factor of the system.
            By default, length_scale=1.0 and harmonic bonds and repulsive
            forces have the scale of 1 nm.

        maxEk: float, optional
            raise error if kinetic energy in (kT/particle) exceeds this value 

        platform : string, optional
            Platform to use

        verbose : bool, optional
            Shout out loud about every change.

        
        precision: str, optional (not recommended to change)
            mixed is optimal for most situations. 
            If you are using double precision, it will be slower by a factor of 10 or so. 
        


        """
        defaultArgs = {"platform":"CUDA", 
                       "GPU":"0",
                       "integrator":"variablelangevin", 
                       "temperature":300,
                       "PBC":False,
                       "length_scale":1.0,
                       "mass":100, 
                       "reporters":[],
                       "maxEk":10 , 
                       "precision":"mixed", 
                       "verbose":False, 
                       "name":"sim"}
        valid_names = list(defaultArgs.keys()) + ["N", "error_tol", "collision_rate", "timestep"]
        for i in kwargs.keys():
            if i not in valid_names:
                raise ValueError("incorrect argument provided: {0}. Allowed are {1}".format(i, valid_names))
        defaultArgs.update(kwargs)
        kwargs = defaultArgs
        self.kwargs = kwargs

        platform = kwargs["platform"]
        self.GPU = kwargs["GPU"]  # setting default GPU

        properties = {}
        if self.GPU.lower() != "default":
            if platform.lower() in ["cuda", "opencl"]:
                properties["DeviceIndex"] = str(self.GPU)
                properties["Precision"] = kwargs["precision"]
        self.properties = properties

        if platform.lower() == "opencl":
            platformObject = openmm.Platform.getPlatformByName('OpenCL')
        elif platform.lower() == "reference":
            platformObject = openmm.Platform.getPlatformByName('Reference')
        elif platform.lower() == "cuda":
            platformObject = openmm.Platform.getPlatformByName('CUDA')
        elif platform.lower() == "cpu":
            platformObject = openmm.Platform.getPlatformByName('CPU')
        else:
            raise RuntimeError("Undefined platform: {0}".format(platform))
        self.platform = platformObject
        
        self.temperature = kwargs["temperature"]

        self.collisionRate = kwargs["collision_rate"] * (1 / ps)

        self.integrator_type = kwargs["integrator"]                
        if isinstance(self.integrator_type, string_types):
            self.integrator_type = str(self.integrator_type)
            if self.integrator_type.lower() == "langevin":
                self.integrator = openmm.LangevinIntegrator(self.temperature,
                    kwargs["collision_rate"] * (1 / ps), kwargs["timestep"]* fs)
            elif self.integrator_type.lower() == "variablelangevin":
                self.integrator = openmm.VariableLangevinIntegrator(self.temperature,
                    kwargs["collision_rate"] * (1 / ps), kwargs["error_tol"])
            elif self.integrator_type.lower() == "verlet":
                self.integrator = openmm.VariableVerletIntegrator(kwargs["timestep"]* fs)
            elif self.integrator_type.lower() == "variableverlet":
                self.integrator = openmm.VariableVerletIntegrator(kwargs["error_tol"])

            elif self.integrator_type.lower() == 'brownian':
                self.integrator = openmm.BrownianIntegrator(self.temperature,
                   kwarg["collision_rate"] * (1 / ps), kwargs["timestep"])
            else:
                logging.info(
                    'Using the provided integrator object'
                       #'please select from "langevin", "variablelangevin", '
                       #'"verlet", "variableVerlet", '
                       #'"brownian" or provide an integrator object'
                )
                self.integrator = integrator
        else:
            self.integrator = self.integrator_type
            self.integrator_type = "UserDefined"
            kwargs["integrator"] = "user_defined"
        
        self.N = kwargs["N"]

        self.verbose = kwargs["verbose"]
        self.reporters = kwargs["reporters"]
        self.forcesApplied = False
        self.length_scale = kwargs["length_scale"]
        self.eKcritical = kwargs["maxEk"]  # Max allowed kinetic energy

        self.step = 0
        self.block = 0
        self.time = 0 

        self.nm = nm
        self.kB = units.BOLTZMANN_CONSTANT_kB * \
            units.AVOGADRO_CONSTANT_NA  # Boltzmann constant
        self.kT = self.kB * self.temperature * units.kelvin  # thermal energy        
        
        # All masses are the same,
        # unless individual mass multipliers are specified in self.load()
        self.conlen = 1. * nm * self.length_scale
        
        self.kbondScalingFactor = float(
                (2 * self.kT / (self.conlen) ** 2) 
                / (units.kilojoule_per_mole / nm ** 2))

        self.system = openmm.System()

        self.PBC = kwargs["PBC"]

        if self.PBC == True:  # if periodic boundary conditions
            PBCbox = np.array(kwargs["PBCbox"])            
            self.system.setDefaultPeriodicBoxVectors(
                [PBCbox[0], 0., 0.], 
                [0., PBCbox[1], 0.], 
                [0., 0., PBCbox[2]])

        self.forceDict = {}  # Dictionary to store forces
            
            
    def getData(self):
        "Returns an Nx3 array of positions"
        return np.asarray(self.data / nm, dtype=np.float32)

    
    def getScaledData(self):
        """Returns data, scaled back to PBC box """
        if self.PBC != True:
            return self.getData()
        alldata = self.getData()
        boxsize = np.array(self.kwargs["PBCbox"])
        mults = np.floor(alldata / boxsize[None, :])
        toRet = alldata - mults * boxsize[None, :]
        assert toRet.min() >= 0
        return toRet

    
    def setData(self, data, center=False, random_offset = 1e-5):
        """Sets particle positions

        Parameters
        ----------

        data : Nx3 array-line
            Array of positions 

        center : bool or "zero", optional
            Move center of mass to zero before starting the simulation
            if center == "zero", then center the data such as all positions are positive and start at zero
            
        random_offset: float or None
            add random offset to each particle
            Recommended for integer starting conformations and in general 

         """

        
        data = np.asarray(data, dtype="float")
        if len(data) != self.N: 
            raise ValueError(f"length of data, {len(data)} does not match N, {self.N}")

        if data.shape[1] != 3:
            raise ValueError("Data is not shaped correctly. Needs (N,3), provided: {0}".format(data.shape))
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
        
        self.data = units.Quantity(data, nm)
        for reporter in self.reporters:
            reporter.report("load_data", {"data":data, "time":self.time, "block":self.block})
        
        if hasattr(self, "context"):
            self.initPositions()        
        

    def RG(self):
        """
        Returns
        -------

        Gyration ratius in units of length (bondlength).
        """
        data = self.getScaledData()
        data = data - np.mean(data, axis=0)[None,:]
        return np.sqrt(np.sum(np.var(np.array(data), 0)))    

    
    def dist(self, i, j):
        """
        Calculates distance between particles i and j
        """
        data = self.getData()
        dif = data[i] - data[j]
        return np.sqrt(sum(dif ** 2))
        

    def addForce(self, force):
        if isinstance(force, Iterable):
            for f in force:
                self.addForce(f)
        else:
            if force.name in self.forceDict:
                raise ValueError(
                    'A force named {} was added to the system twice!'.format(force.name))
            forces._prepend_force_name_to_params(force)
            self.forceDict[force.name] = force
        

    def _applyForces(self):
        """Adds all particles to the system.
        Then applies all the forces in the forcedict.
        Forces should not be modified after that, unless you do it carefully
        (see openmm reference)."""

        if self.forcesApplied == True:
            return
        
        self.masses = np.zeros(self.N, dtype=float) + self.kwargs["mass"]
        for mass in self.masses:
            self.system.addParticle(mass)


        for i in list(self.forceDict.keys()):  # Adding forces
            force = self.forceDict[i]
                    
            if hasattr(force, "CutoffNonPeriodic") and hasattr(force, "CutoffPeriodic"):
                if self.PBC:
                    force.setNonbondedMethod(force.CutoffPeriodic)
                    logging.info("Using periodic boundary conditions")
                else:
                    force.setNonbondedMethod(force.CutoffNonPeriodic)
                    
            logging.info("adding force {} {}".format( 
                i, self.system.addForce(self.forceDict[i])))
            
        for reporter in self.reporters:
            reporter.report("applied_forces", {i:j.__getstate__() for i,j in self.forceDict.items()})

        self.context = openmm.Context(self.system, self.integrator, self.platform, self.properties)
        self.initPositions()
        self.initVelocities()
        self.forcesApplied = True

        
    def initVelocities(self,  temperature="current"):
        """Initializes particles velocities

        Parameters
        ----------
        temperature: temperature to set velocities (default: temerature of the simulation)        
        """
        try:
            self.context
        except:
            raise ValueError("No context, cannot set velocs."\
                             "Initialize context before that")
            
        if temperature == "current":
            temperature=self.temperature        
            
        self.context.setVelocitiesToTemperature(temperature)
    
    def initPositions(self):
        """Sends particle coordinates to OpenMM system.
        If system has exploded, this is
         used in the code to reset coordinates. """

        try:
            self.context
        except:
            raise ValueError("No context, cannot set positions."\
                             " Initialize context before that")

        self.context.setPositions(self.data)        
        eP = self.context.getState(getEnergy=True).getPotentialEnergy() / self.N / self.kT
        logging.info("Particles loaded. Potential energy is %lf" % eP)

        
    def reinitialize(self):
        """Reinitializes the OpenMM context object.
        This should be called if low-level parameters,
        such as forces, have changed"""
        
        self.context.reinitialize()
        self.initPositions()
        self.initVelocities()
        

    def localEnergyMinimization(self, tolerance=0.3, maxIterations=0):
        "A wrapper to the build-in OpenMM Local Energy Minimization"

        logging.info("Performing local energy minimization")

        self._applyForces()

        self.state = self.context.getState(getPositions=False,
                                           getEnergy=True)
        eK = (self.state.getKineticEnergy() / self.N / self.kT)
        eP = self.state.getPotentialEnergy() / self.N / self.kT
        locTime = self.state.getTime()
        logging.info("before minimization eK={0}, eP={1}, time={2}".format(eK, eP, locTime))

        openmm.LocalEnergyMinimizer.minimize(self.context, tolerance, maxIterations)

        self.state = self.context.getState(getPositions=False, getEnergy=True)
        eK = (self.state.getKineticEnergy() / self.N / self.kT)
        eP = self.state.getPotentialEnergy() / self.N / self.kT
        locTime = self.state.getTime()
        logging.info("after minimization eK={0}, eP={1}, time={2}".format(eK, eP, locTime))


    def doBlock(self, steps=None, increment=True,  reinitialize=True, maxIter=0, checkFunctions=[], getVelocities = False, save=True):
        """performs one block of simulations, doing steps timesteps,
        or steps_per_block if not specified.

        Parameters
        ----------

        steps : int or None
            Number of timesteps to perform.
        increment : bool, optional
            If true, will not increment self.block and self.steps counters
        """

        if self.forcesApplied == False:
            if self.verbose:
                logging.info("applying forces")
                sys.stdout.flush()
            self._applyForces()
            self.forcesApplied = True



        a = time.time()
        self.integrator.step(steps)  # integrate!

        if increment:
            self.block += 1
            self.step += steps
        
        self.state = self.context.getState(getPositions=True,getVelocities=getVelocities,
                                           getEnergy=True)
            
        b = time.time()        
        coords = self.state.getPositions(asNumpy=True)
        newcoords = coords / nm
        self.time = self.state.getTime() / ps

        # calculate energies in KT/particle
        eK = (self.state.getKineticEnergy() / self.N / self.kT)
        eP = self.state.getPotentialEnergy() / self.N / self.kT
        curtime = self.state.getTime() / units.picosecond

        msg = "block %4s " % int(self.block)
        msg += "pos[1]=[%.1lf %.1lf %.1lf] " % tuple(newcoords[0])


        checkFail = False
        for checkFunction in checkFunctions:
            if not checkFunction(newcoords):
                checkFail = True

        if np.isnan(newcoords).any():
            raise integrationFailError("Coordinates are NANs")
        if (eK > self.eKcritical):
            raise eKExceedsError("Ek exceeds {0}".format(self.eKcritical))
        if  (np.isnan(eK)) or (np.isnan(eP)):
            raise integrationFailError("Energy is NAN)")
        if checkFail:
            raise integrationFailError("Custom checks failed")

        dif = np.sqrt(np.mean(np.sum((newcoords - self.getData()) ** 2, axis=1)))
        msg += "dr=%.2lf " % (dif,)
        self.data = coords
        msg += "t=%2.1lfps " % (self.state.getTime() / ps)
        msg += "kin=%.2lf pot=%.2lf " % (eK,eP)
        msg += "Rg=%.3lf " % self.RG()
        msg + "SPS=%.0lf " % (steps / (float(b - a)))

        if (self.integrator_type.lower() == 'variablelangevin'
            or self.integrator_type.lower() == 'variableverlet'):
            dt = self.integrator.getStepSize()
            msg += 'dt=%.1lffs ' % (dt / fs)
            mass = self.system.getParticleMass(1)
            dx = (units.sqrt(2.0 * eK * self.kT / mass) * dt)
            msg += 'dx=%.2lfpm ' % (dx / nm * 1000.0)
          
        logging.info(msg)

        result =  {"coordinates":newcoords, "potentialEnergy":eP, "kineticEnergy":eK, "time":curtime, "block":self.block}
        if getVelocities:
            result["velocities"] = self.state.getVelocities() / (units.nanometer / units.picosecond)
        if save:
            for reporter in self.reporters:
                reporter.report("data", result)
        return result
    

    def printStats(self):
        """Prints detailed statistics of a system.
        Will be run every 50 steps
        """
        state = self.context.getState(getPositions=True,
            getVelocities=True, getEnergy=True)

        eP = state.getPotentialEnergy()
        pos = np.array(state.getPositions() / nm)
        bonds = np.sqrt(np.sum(np.diff(pos, axis=0) ** 2, axis=1))
        sbonds = np.sort(bonds)
        vel = state.getVelocities()
        mass = self.system.getParticleMass(1)
        vkT = np.array(vel / units.sqrt(self.kT / mass), dtype=float)
        self.velocs = vkT
        EkPerParticle = 0.5 * np.sum(vkT ** 2, axis=1)

        cm = np.mean(pos, axis=0)
        centredPos = pos - cm[None, :]
        dists = np.sqrt(np.sum(centredPos ** 2, axis=1))
        per95 = np.percentile(dists, 95)
        den = (0.95 * self.N) / ((4. * np.pi * per95 ** 3) / 3)
        per5 = np.percentile(dists, 5)
        den5 = (0.05 * self.N) / ((4. * np.pi * per5 ** 3) / 3)
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        minmedmax = lambda x: (x.min(), np.median(x), x.mean(), x.max())

        
        print("\n Statistics: number of particles: %d\n" % (self.N, ))        
        print("Statistics for particle position")
        print("     mean position is: ", np.mean(
            pos, axis=0), "  Rg = ", self.RG())
        print("     median bond size is ", np.median(bonds))
        print("     three shortest/longest (<10)/ bonds are ", sbonds[:3], "  ", sbonds[sbonds < 10][-3:])
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
        print("     mean kinetic energy is: ", np.mean(
            EkPerParticle), "should be:", 1.5)
        print("     fastest particles are (in kT): ", np.sort(
            EkPerParticle)[-5:])

        print()
        print("Statistics for the system:")
        print("     Forces are: ", list(self.forceDict.keys()))
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
            data = np.transpose(data)
        if len(data[0]) != 3:
            logging.error("wrong data!")
            return
        # determining the 95 percentile distance between particles,
        if scale == "auto":
            meandist = np.percentile(np.sqrt(
                np.sum(np.diff(data, axis=0) ** 2, axis=1)), 95)
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
        colors = np.array([int((j * 450.) / (len(data))) -
            225 for j in range(len(data))])

        # creating spheres along the trajectory
        newData = np.zeros(
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

