"""
This is a sample simulation that does not represent any particular biological system. It is just a showcase
of how create a Simulation object, add forces, and initialize the reporter.

In this simulation, a simple polymer chain of 10,000 monomers is simulated.
"""


import os
import sys

import openmm

import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from polychrom.hdf5_format import HDF5Reporter

N = 10000

reporter = HDF5Reporter(folder="trajectory", max_data_length=5, overwrite=True)
sim = simulation.Simulation(
    platform="CUDA",
    integrator="variableLangevin",
    error_tol=0.003,
    GPU="1",
    collision_rate=0.03,
    N=N,
    save_decimals=2,
    PBCbox=False,
    reporters=[reporter],
)

polymer = starting_conformations.grow_cubic(10000, 100)

sim.set_data(polymer, center=True)  # loads a polymer, puts a center of mass at zero

sim.add_force(forces.spherical_confinement(sim, density=0.85, k=1))

sim.add_force(
    forcekits.polymer_chains(
        sim,
        chains=[(0, None, False)],
        # By default the library assumes you have one polymer chain
        # If you want to make it a ring, or more than one chain, use self.setChains
        # self.setChains([(0,50,True),(50,None,False)]) will set a 50-monomer ring and a chain from 50 to the end
        bond_force_func=forces.harmonic_bonds,
        bond_force_kwargs={
            "bondLength": 1.0,
            "bondWiggleDistance": 0.05,  # Bond distance will fluctuate +- 0.05 on average
        },
        angle_force_func=forces.angle_force,
        angle_force_kwargs={
            "k": 1.5,
            # K is more or less arbitrary, k=4 corresponds to presistence length of 4,
            # k=1.5 is recommended to make polymer realistically flexible; k=8 is very stiff
        },
        nonbonded_force_func=forces.polynomial_repulsive,
        nonbonded_force_kwargs={
            "trunc": 3.0,  # this will let chains cross sometimes
            # 'trunc':10.0, # this will resolve chain crossings and will not let chain cross anymore
        },
        except_bonds=True,
    )
)


for _ in range(10):  # Do 10 blocks
    sim.do_block(100)  # Of 100 timesteps each. Data is saved automatically.
sim.print_stats()  # In the end, print very simple statistics

reporter.dump_data()  # always need to run in the end to dump the block cache to the disk
