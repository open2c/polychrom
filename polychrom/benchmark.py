"""
A simple benchmark for polychrom. 

This benchmark is a simple simulation of a polymer chain in a cubic box with PBC,
at density of 0.25. 
It defaults to a chain with N=100,000 monomers, but it can be changed by passing N as a first argument.

The simulation is run for 10 blocks of 5,000 steps each.

The benchmark reports initialization time in ms, as well as the average steps per second (SPS) for the simulation.

The benchmark can be run with the following command:
python -m polychrom.benchmark [N] [GPU]
"""


from polychrom.simulation import Simulation
from polychrom.forces import harmonic_bonds, polynomial_repulsive, angle_force
from polychrom.forcekits import polymer_chains
from polychrom.starting_conformations import grow_cubic
import sys 
import datetime as dt



def run_behcnmark(*args):
    # Get the number of monomers from the command line argument
    # measure time 
    N = int(args[1]) if len(args) > 1 else 100_000
    GPU = args[2] if len(args) > 2 else "0"
    box_size = N ** (1 / 3) / 0.25
 
    polymer = grow_cubic(N, boxSize=int(box_size)-2)
 
    time_start = dt.datetime.now()

 
    # Create a simulation object
    sim = Simulation(
        collision_rate = 0.02,
        N = N,
        error_tol=0.003,
        PBCbox = (box_size, box_size, box_size),
                     GPU = GPU,
                     )
    
    sim.set_data(polymer, center=True)  # loads a polymer, puts a center of mass at zero


    sim.add_force(
        polymer_chains(
            sim,
            chains=[(0, None, False)],
            bond_force_func=harmonic_bonds,
            bond_force_kwargs={
                "bondLength": 1.0,
                "bondWiggleDistance": 0.05,  # Typical parameters used in a sim
            },
            angle_force_func=angle_force,
            angle_force_kwargs={
                "k": 1.5,  # gentle angle force
            },
            nonbonded_force_func=polynomial_repulsive,
            nonbonded_force_kwargs={
                "trunc": 5.0,  # on the border of chain crossing - good for benchmark
            },
            except_bonds=True,
        )
    )
    sim.do_block(1)
    

    # init time 
    init_time = dt.datetime.now() - time_start

    time_start = dt.datetime.now()

    for _ in range(10):  # Do 10 blocks
        sim.do_block(5000)  # Of 100 timesteps each. Data is saved automatically.

    sim_time = dt.datetime.now() - time_start
    sps = (10 * 5000) / sim_time.total_seconds()

    print(f"Initialization time: {init_time.total_seconds() * 1000:.2f} ms")
    print(f"Average steps per second: {sps:.2f} SPS")


if __name__ == "__main__":
    run_behcnmark(*sys.argv)