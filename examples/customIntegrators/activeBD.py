"""
Polymer simulations with custom integrators
-------------------------------------------

This is a sample python script to run a polychrom simulation with the custom integrators in 
polychrom.contrib.integrators. The first example involves defining a diffusion coefficient
for each monomer and integrating the overdamped Langevin equation with the `ActiveBrownianIntegrator'.
TODO: test example

The second example illustrates how to drive the polymer with correlated active forces. To define
these correlated excitations, we assign each monomer an identity ("charge"): +, 0, or -. Monomers
of the same charge are correlated, whereas monomers of opposing charges are anti-correlated. Neutral
monomers do not experience correlated active forces.

"""
import time
import numpy as np
import os, sys
import polychrom
from polychrom import simulation, starting_conformations, forces, forcekits
from polychrom.contrib.integrators import ActiveBrownianIntegrator
import openmm
from polychrom.hdf5_format import HDF5Reporter
from simtk import unit
from pathlib import Path

N = 1000 #1000 monomers
ids = np.ones(N) #aray of 1s and 0s assigning type A and type B comonomers
#as an example, let first N/2 monomers be of type A, and second N/2 monomers be of type B.
#1 is A, 0 is B
ids[int(N/2):] = 0

def run_monomer_diffusion(gpuid, N, ids, activity_ratio, timestep=170, ntimesteps=200000, blocksize=100):
    """ Run a single simulation on GPU i of a hetero-polymer with A monomers and B monomers. A monomers
    have a larger diffusion coefficient than B monomers, with an activity ratio of D_A / D_B."""
    if len(ids) != N:
        raise ValueError("The array of monomer identities must have length equal to the total number of monomers.")
    D = np.ones((N, 3)) #Dx, Dy, Dz --> we assume the diffusion coefficient in each spatial dimension is the same
    #by default set the average diffusion coefficient to be 1 kT/friction
    #let D_A = 1 + Ddiff and D_B = 1 - Ddiff such that D_A / D_B is the given activity_ratio
    Ddiff = (activity_ratio - 1) / (activity_ratio + 1)
    D[ids==0, :] = 1.0 - Ddiff
    D[ids==1, :] = 1.0 + Ddiff
    #monomer density in confinement in units of monomers/volume
    density = 0.224
    r = (3 * N / (4 * 3.141592 * density)) ** (1/3)
    print(f"Radius of confinement: {r}")
    timestep = timestep 
    #the monomer diffusion coefficient should be in units of kT / friction, where friction = mass*collision_rate
    collision_rate = 2.0
    mass = 100 * unit.amu
    friction = collision_rate * (1.0 / unit.picosecond) * mass
    temperature = 300
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kT = kB * temperature * unit.kelvin
    particleD = unit.Quantity(D, kT/friction)
    integrator = ActiveBrownianIntegrator(timestep, collision_rate, particleD)
    gpuid = f"{gpuid}"
    reporter = HDF5Reporter(folder="active_inactive", max_data_length=100, overwrite=True)
    sim = simulation.Simulation(
        platform="CUDA", 
        integrator=integrator,
        timestep=timestep,
        temperature=temperature,
        GPU=gpuid,
        collision_rate=collision_rate,
        N=N,
        save_decimals=2,
        PBCbox=False,
        reporters=[reporter],
    )

    polymer = starting_conformations.grow_cubic(N, int(np.ceil(r)))
    sim.set_data(polymer, center=True)  # loads a polymer, puts a center of mass at zero
    sim.set_velocities(v=np.zeros((N,3))) #initializes velocities of all monomers to zero (no inertia)
    sim.add_force(forces.spherical_confinement(sim, density=density, k=5.0))
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
    for _ in range(ntimesteps):  # Do 10 blocks
        sim.do_block(blocksize)  # Of 100 timesteps each. Data is saved automatically. 
    toc = time.perf_counter()
    print(f'Ran simulation in {(toc - tic):0.4f}s')
    sim.print_stats()  # In the end, print very simple statistics
    reporter.dump_data()  # always need to run in the end to dump the block cache to the disk


if __name__ == '__main__':
    gpuid = int(sys.argv[1])
    activity_ratio = int(sys.argv[1])
    run_monomer_diffusion(gpuid, N, ids, activity_ratio)