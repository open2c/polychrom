import os, sys
import polychrom
from polychrom import simulation, starting_conformations, forces, forcekits
import simtk.openmm as openmm
import os
from polychrom.hdf5_format import HDF5Reporter

N=10000
ATTR = 0.3

reporter = HDF5Reporter(folder="trajectory", max_data_length=5, overwrite=True)
sim = simulation.Simulation(
    platform="CUDA",  # <--------- change this to CUDA for simulations on a GPU
    integrator="variableLangevin",
    error_tol=0.005,
    GPU="0",
    collision_rate=0.03,
    N=N,
    save_decimals=None,
    PBCbox=[(10*N)**(1/3)]*3,
    reporters=[reporter],
)

polymer = starting_conformations.grow_cubic(10000, 100)

sim.set_data(polymer, center=True)  # loads a polymer, puts a center of mass at zero

#sim.add_force(forces.spherical_confinement(sim, density=0.85, k=1))

sim.add_force(
    forcekits.polymer_chains(
        sim,
        chains=[(0, None, False)],
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
        nonbonded_force_func=forces.addSelectiveSSWForce,
        nonbonded_force_kwargs={
         "stickyParticlesIdxs":list(np.nonzero(comp)[0])
        "repulsionEnergy":5.0,
        "repulsionRadius":1.0,
        "attractionEnergy":0.1,
        "attractionRadius":1.6,
         "selectiveRepulsionEnergy":5,
          "selectiveAttractionEnergy":ATTR,  
        },
        except_bonds=True,
    )
)
 a.addSelectiveSSWForce(=list(np.nonzero(comp)[0]), extraHardParticlesIdxs=[], repulsionEnergy=5, repulsionRadius=1, attractionEnergy=0.1, attractionRadius=1.5, selectiveRepulsionEnergy=5, selectiveAttractionEnergy=attr)

# -----------Running a simulation ---------


# sim.save()  # save original conformationz

for _ in range(10):  # Do 10 blocks
    sim.do_block(100)  # Of 2000 timesteps each
    # sim.save()  # and save data every block
sim.print_stats()  # In the end, print statistics
# sim.show()  # and show the polymer if you want to see it.

reporter.dump_data()
