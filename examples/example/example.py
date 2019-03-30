# (c) 2013 Massachusetts Institute of Technology. All Rights Reserved
# Code written by: Maksim Imakaev (imakaev@mit.edu)

from __future__ import absolute_import, division, print_function, unicode_literals
import os,sys
sys.path.insert(0, os.path.abspath(".."))
from openmmlib2.openmmlib import Simulation
from openmmlib2 import polymerutils
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
    a = Simulation(thermostat=0.02)  # timestep not necessary for variableLangevin

    a.setup(platform="CPU", integrator="variableLangevin", errorTol=0.01,  GPU = "0")

    a.saveFolder("trajectory")  # folder where to save trajectory


    # ------- Creation of the initial conformation-----------

    # polymer = polymerutils.load("globule")
    # loads compact polymer conformation of the length 6000

    # polymer = polymerutils.grow_rw(8000, 50, method="standard")
    # grows a compact polymer ring of a length 8000 in a 50x50x50 box

    # polymer = polymerutils.create_spiral(r1=4, r2=10, N=8000)
    # Creates a compact polymer arranged in a cylinder of radius 10, 8000 monomers long

    polymer = polymerutils.create_random_walk(1, 1000)
    # Creates an extended "random walk" conformation of length 8000

    a.load(polymer, center=True)  # loads a polymer, puts a center of mass at zero

    # -----------Initialize conformation of the chains--------
    # By default the library assumes you have one polymer chain
    # If you want to make it a ring, or more than one chain, use self.setChains
    # self.setChains([(0,50,1),(50,None,0)]) will set a 50-monomer ring and a chain from monomer 50 to the end


    # -----------Adding forces ---------------
    a.addSphericalConfinement(density=0.85, k=1)
    # Specifying density is more intuitive than radius
    # k is the slope of confinement potential, measured in kT/mon
    # set k=5 for harsh confinement
    # and k = 0.2 or less for collapse simulation

    a.addHarmonicPolymerBonds(wiggleDist=0.05)
    # Bond distance will fluctuate +- 0.05 on average

    a.addPolynomialRepulsiveForce(trunc=10)
    # this will resolve chain crossings and will not let chain cross anymore

    # a.addGrosbergRepulsiveForce(trunc=5)
    # this will let chains cross sometimes

    a.addStiffness(k=4)
    # K is more or less arbitrary, k=4 corresponds to presistence length of 4,
    # k=1.5 is recommended to make polymer realistically flexible; k=8 is very stiff

    # If your simulation does not start, consider using energy minimization below

    # a.localEnergyMinimization()
    # A very efficient algorithm to reach local energy minimum
    # Use it to minimize energy if you're doing diffusion simulations
    # If you're simulating dynamics of collapse or expansion, please do not use it
    
    # An algorithm to start a simulation
    # Which works only with langevin integrator (but will not throw an error otherwise)
    # Decreases a timestep, and then increases it slowly back to normal

    # -----------Running a simulation ---------

    a.save()  # save original conformationz
    for _ in range(10):  # Do 10 blocks
        a.doBlock(2000)  # Of 2000 timesteps each
        a.save()  # and save data every block
    a.printStats()  # In the end, print statistics
    #a.show()  # and show the polymer if you want to see it.


exampleOpenmm()
exit()
