# (c) 2013 Massachusetts Institute of Technology. All Rights Reserved
# Code written by: Maksim Imakaev (imakaev@mit.edu)

from __future__ import absolute_import, division, print_function, unicode_literals
import os,sys
import polychrom
from polychrom import (simulation, starting_conformations,
                       forces, extra_forces, forcekits)
import simtk.openmm as openmm
import os

def exampleOpenmm():
    """
    An example script which generates an extended polymer, and lets it collapse to a sphere.
    Please follow comments along the text for explanations.
    """

    # ----------- Initializing general simulation parameters---------

    # Initialization for simulations with constant environment, using default integrator (langevin)
    # Fine-tune timestep and thermostat parameters so that your simulation does not blow up,
    # But is going as fast as possible. You might need to increase timestep, but don't let
    # your kinetic energy be above 1.6 in the "steady" regime

    # a = Simulation(timestep=80, thermostat=0.005)
    # a.setup(platform="cuda", verbose=True)  # Switch to platform="OpenCL" if you don't have cuda

    # Alternative initialization for dynamic simulations with strong forces,
    #  which would automatically adjusts timestep
    # This is relevant, for example, for simulations of polymer collapse
    # If simulation blows up, decrease errorTol by a factor of two and try again
    sim = simulation.Simulation(
            platform="CPU", 
            integrator="variableLangevin", 
            error_tol=0.002, 
            GPU = "0", 
            collision_rate=0.02, 
            N = 10000)  # timestep not necessary for variableLangevin

    # sim.saveFolder("trajectory")  # folder where to save trajectory


    # ------- Creation of the initial conformation-----------

    # polymer = polymerutils.load("globule")
    # loads compact polymer conformation of the length 6000

    # polymer = polymerutils.grow_rw(8000, 50, method="standard")
    # grows a compact polymer ring of a length 8000 in a 50x50x50 box

    # polymer = polymerutils.create_spiral(r1=4, r2=10, N=8000)
    # Creates a compact polymer arranged in a cylinder of radius 10, 8000 monomers long

    polymer = starting_conformations.grow_cubic(10000, 100)
    # Creates an extended "random walk" conformation of length 8000

    sim.setData(polymer, center=True)  # loads a polymer, puts a center of mass at zero

    # -----------Adding forces ---------------
    sim.addForce(
        forces.sphericalConfinement(sim, density=0.85, k=1))
    # Specifying density is more intuitive than radius
    # k is the slope of confinement potential, measured in kT/mon
    # set k=5 for harsh confinement
    # and k = 0.2 or less for collapse simulation

    # forces.polynomialRepulsiveForce(sim, trunc=10)

    sim.addForce(
        forcekits.polymerChains(
            sim,
            chains=[(0, None, False)],

                # By default the library assumes you have one polymer chain
                # If you want to make it a ring, or more than one chain, use self.setChains
                # self.setChains([(0,50,1),(50,None,0)]) will set a 50-monomer ring and a chain from monomer 50 to the end

            bondForceFunc=forces.harmonicBonds,
            bondForceKwargs={
                'bondLength':1.0,
                'bondWiggleDistance':0.05, # Bond distance will fluctuate +- 0.05 on average
             },

            angleForceFunc=forces.angleForce,
            angleForceKwargs={
                'k':0.05
                # K is more or less arbitrary, k=4 corresponds to presistence length of 4,
                # k=1.5 is recommended to make polymer realistically flexible; k=8 is very stiff
            },

            nonbondedForceFunc=forces.polynomialRepulsiveForce,
            nonbondedForceKwargs={
                'trunc':3.0, # this will let chains cross sometimes
                #'trunc':10.0, # this will resolve chain crossings and will not let chain cross anymore
            },

            exceptBonds=True,
        )
    )


    # If your simulation does not start, consider using energy minimization below

    # sim.localEnergyMinimization()
    # A very efficient algorithm to reach local energy minimum
    # Use it to minimize energy if you're doing diffusion simulations
    # If you're simulating dynamics of collapse or expansion, please do not use it
    
    # An algorithm to start a simulation
    # Which works only with langevin integrator (but will not throw an error otherwise)
    # Decreases a timestep, and then increases it slowly back to normal

    # -----------Running a simulation ---------

    
    
    #sim.save()  # save original conformationz
    
    for _ in range(10):  # Do 10 blocks
        sim.doBlock(2000)  # Of 2000 timesteps each
        #sim.save()  # and save data every block
    sim.printStats()  # In the end, print statistics
    #sim.show()  # and show the polymer if you want to see it.


exampleOpenmm()
exit()
