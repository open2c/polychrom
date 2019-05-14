# (c) 2013 Massachusetts Institute of Technology. All Rights Reserved
# Code written by: Maksim Imakaev (imakaev@mit.edu)
import matplotlib.pyplot as plt 
import numpy as np 
from polychrom import simulation, forces, forcekits, starting_conformation
import polymerutils
import os

def exampleOpenmm():
    """
    An example script which generates an extended polymer, and lets it collapse to a sphere.
    Please follow comments along the text for explanations.
    """

    sim = simulation.Simulation(
        platform="cuda",  # Switch to platform="OpenCL" if you don't have cuda
        
        # 
        # Fine-tune timestep and thermostat parameters so that your simulation does not blow up,
        # But is going as fast as possible. You might need to increase timestep, but don't let
        # your kinetic energy be above 1.6 in the "steady" regime
        # integrator="langevin", 
        # timestep=80, 
        # thermostat=0.005
        
        # Alternative initialization for dynamic simulations with strong forces,
        #  which would automatically adjusts timestep
        # This is relevant, for example, for simulations of polymer collapse
        # If simulation blows up, decrease errorTol by a factor of two and try again
        # a variableLangevin thermostat automatically calculates the timestep
        # for a given error level 
        integrator="variableLangevin", 
        errorTol=0.06, 
        thermostat=0.02,
        
        verbose=True)

    sim.saveFolder("trajectory")  # folder where to save trajectory


    # ------- Creation of the initial conformation-----------

    # polymer = polymerutils.load("globule")
    # loads compact polymer conformation of the length 6000

    # polymer = starting_conformations.grow_cubic(8000, 50, method="standard")
    # grows a compact polymer ring of a length 8000 in a 50x50x50 box

    # polymer = starting_conformations.create_spiral(r1=4, r2=10, N=8000)
    # Creates a compact polymer arranged in a cylinder of radius 10, 8000 monomers long

    polymer = starting_conformations.create_random_walk(1, 8000)
    # Creates an extended "random walk" conformation of length 8000

    sim.setData(polymer, center=True)  # loads a polymer, puts a center of mass at zero



    # -----------Adding forces ---------------
#    sim.sphericalConfinement(density=0.85, k=1)
    # Specifying density is more intuitive than radius
    # k is the slope of confinement potential, measured in kT/mon
    # set k=5 for harsh confinement
    # and k = 0.2 or less for collapse simulation

    forcekits.polymerChains(
        sim,
        # By default the library assumes you have one polymer chain
        # If you want to make it a ring, or more than one chain, provide a chains parameter
        # chains = [(0, 50, True), (50, None, False)], # set a 50-monomer ring and a chain from monomer 50 to the end
        
        #bondForceFunc=forces.FENEBonds, # uncomment to bind particles in the chains with 
                                         # a constant force (i.e. use a linear potential instead of harmonic)
        bondForceKwags={'wiggleDist':0.05}, # Bond distance will fluctuate +- 0.05 on average
        
        #angleForceFunc=None, # uncomment to disable stiffness 
        angleForceKwargs={'k':4},
        # K is more or less arbitrary, k=4 corresponds to presistence length of 4,
        # k=1.5 is recommended to make polymer realistically flexible; k=8 is very stiff
        
        #nonbondedForceFunc=None, # uncomment to disable particle repulsion
        nonbondedForceKwargs={'trunc':1.5},
        )


    # If your simulation does not start, consider using energy minimization below

    # sim.localEnergyMinimization()
    # A very efficient algorithm to reach local energy minimum
    # Use it to minimize energy if you're doing diffusion simulations
    # If you're simulating dynamics of collapse or expansion, please do not use it

    #sim.energyMinimization(stepsPerIteration=10)  # increase to 100 for larger or more complex systems
    # An algorithm to start a simulation
    # Which works only with langevin integrator (but will not throw an error otherwise)
    # Decreases a timestep, and then increases it slowly back to normal

    # -----------Running a simulation ---------

    sim.save()  # save original conformation
    allDists = []
    for _ in xrange(10):  # Do 10 blocks
        sim.doBlock(2000)  # Of 2000 timesteps each
        data = sim.getData()
        dists = np.sqrt(np.sum(np.diff(data, axis=0)**2, axis=1))
        allDists.append(dists)
     
    allDists = np.concatenate(allDists)
    plt.hist(allDists,100)
    plt.title("deviation should be around 0.05")
    
    plt.show()
        


exampleOpenmm()
exit()
