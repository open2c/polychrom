# (c) 2013 Massachusetts Institute of Technology. All Rights Reserved
# Code written by: Maksim Imakaev (imakaev@mit.edu)

from __future__ import absolute_import, division, print_function, unicode_literals
import os,sys
import polychrom
from polychrom import (simulation, starting_conformations,
                       forces, forcekits)
import simtk.openmm as openmm
import os
from polychrom.hdf5_format import HDF5Reporter


def exampleOpenmm():
    """
    An example script which generates an extended polymer, and lets it collapse to a sphere.
    Please follow comments along the text for explanations.
    """
    
    # creating a reporter - see examples/storage_formats/hdf5_reporter.ipynb for explanations/examples
    
    reporter = HDF5Reporter(folder="trajectory", max_data_length=5, overwrite=True)
        
    #Simulation object has many parameters that should be described in polychrom/simulation.py file 
    sim = simulation.Simulation(
            platform="CPU",   # <--------- change this to CUDA for simulations on a GPU 
            integrator="variableLangevin", 
            error_tol=0.002, 
            GPU = "0", 
            collision_rate=0.1, 
            N = 10000,
            save_decimals=None,
            reporters=[reporter]) 
    
    # Creates a compact conformation on a cubic lattice, length=10,000; grown in a 100x100x100 box     
    polymer = starting_conformations.grow_cubic(10000, 100)

    # Now we load the data into the simulation object 
    sim.set_data(polymer, center=True)  # loads a polymer, puts a center of mass at zero


    
    
    ### -----------Adding forces and forcekits  ---------
    
    # Many forces are independent from each other, and can be just added to the system
    
    # However, in some cases some forces go together as a group and have shared properties 
    # For example, both polymer chain and polymer stiffness demand a set of chains
    # And exceptions to the nonbonded force should be added for all polymer bonds 
    # Forcekits can explicitly take care of such dependencies 
    # While still providing flexibility of choosing which forces to use 
    
    
    # This is an example of a standalone force that implements spherical confinement 
    # This force does not depend on any other forces and is just added alone 
    sim.add_force(
        forces.spherical_confinement(sim, density=0.85, k=1))
    # Specifying density is more intuitive than radius
    # k is the slope of confinement potential, measured in kT/mon
    # set k=5 for harsh confinement
    # and k = 0.2 or less for collapse simulation


    # This is an example of a forcekit that intorudces dependencies between polymer chain related forces 
    # This forcekit takes chains as a second argument; in the same format as openmmlib 
    # It then takes a function to initialize polymer bond force, and a dictionary of parameters 
    # Then the same for angleForce and for the nonbondedForce 
    
    # Note that if you defined your own force of a certain type (e.g. your own polymer bond force)
    # Then you can simply sideload it using lambda functions ad in example below  
    #
    # myforce = openmm.customExternalForce(.......)
    # forcekits.polymerChains(...,
    # bondForceFunc = lambda x:myforce,
    # bondForceKwargs = {}
    # ... ) 
    
    sim.add_force(
        forcekits.polymer_chains(
            sim,
            chains=[(0, None, False)],
                # By default the library assumes you have one polymer chain
                # If you want to make it a ring, or more than one chain, use self.setChains
                # self.setChains([(0,50,True),(50,None,False)]) will set a 50-monomer ring and a chain from monomer 50 to the end

            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                'bondLength':1.0,
                'bondWiggleDistance':0.05, # Bond distance will fluctuate +- 0.05 on average
             },

            angle_force_func=forces.angle_force,
            angle_force_kwargs={
                'k':1.5, 
                # K is more or less arbitrary, k=4 corresponds to presistence length of 4,
                # k=1.5 is recommended to make polymer realistically flexible; k=8 is very stiff
            },

            nonbonded_force_func=forces.polynomial_repulsive,
            nonbonded_force_kwargs={
                'trunc':3.0, # this will let chains cross sometimes
                #'trunc':10.0, # this will resolve chain crossings and will not let chain cross anymore
            },

            except_bonds=True,
        )
    )


    # -----------Running a simulation ---------

    
    
    #sim.save()  # save original conformationz
    
    for _ in range(10):  # Do 10 blocks
        sim.do_block(100)  # Of 2000 timesteps each
        #sim.save()  # and save data every block
    sim.print_stats()  # In the end, print statistics
    #sim.show()  # and show the polymer if you want to see it.

    reporter.dump_data()


exampleOpenmm()
exit()
