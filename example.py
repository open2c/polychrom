"""
This is a sample simulation that does not represent any particular biological system. It is just a showcase 
of how create a Simulation object, add forces, and initialize the reporter. 

In this simulation, a simple polymer chain of 10,000 monomers is 
"""
import time
import numpy as np
import os, sys
import polychrom
from polychrom import simulation, starting_conformations, forces, forcekits
import openmm
from polychrom.hdf5_format import HDF5Reporter

N=100
density = 0.224
r = (3 * N / (4 * 3.141592 * density)) ** (1/3)
print(f"Radius of confinement: {r}")

reporter = HDF5Reporter(folder="simulations/self-avoidance", max_data_length=100, overwrite=True)
sim = simulation.Simulation(
    platform="CUDA", 
    integrator="brownian",
    timestep=40,
    #error_tol=0.003,
    GPU="1",
    collision_rate=2.0,
    N=N,
    save_decimals=2,
    PBCbox=False,
    reporters=[reporter],
)

polymer = starting_conformations.grow_cubic(N, 5)

sim.set_data(polymer, center=True)  # loads a polymer, puts a center of mass at zero
sim.set_velocities(v=np.zeros((N,3)))
sim.add_force(forces.spherical_confinement(sim, density=0.224, k=5.0))

sim.add_force(
    forcekits.polymer_chains(
        sim,
        chains=[(0, None, False)],
        # By default the library assumes you have one polymer chain
        # If you want to make it a ring, or more than one chain, use self.setChains
        # self.setChains([(0,50,True),(50,None,False)]) will set a 50-monomer ring and a chain from monomer 50 to the end
        bond_force_func=forces.harmonic_bonds,
        bond_force_kwargs={
            "bondLength": 1.0,
            "bondWiggleDistance": 0.1,  # Bond distance will fluctuate +- 0.05 on average
        },
        angle_force_func=None,
        angle_force_kwargs={},
        nonbonded_force_func=forces.polynomial_repulsive,
        nonbonded_force_kwargs={
            "trunc": 3.0,  # this will let chains cross sometimes
            #'trunc':10.0, # this will resolve chain crossings and will not let chain cross anymore
        },
        except_bonds=True,
    )
)

tic = time.perf_counter()
for _ in range(1000):  # Do 10 blocks
    sim.do_block(100)  # Of 100 timesteps each. Data is saved automatically. 
toc = time.perf_counter()
print(f'Ran simulation in {(toc - tic):0.4f}s')
sim.print_stats()  # In the end, print very simple statistics

reporter.dump_data()  # always need to run in the end to dump the block cache to the disk
