import matplotlib.pyplot as plt
import numpy as np

from polychrom import forcekits, simulation, starting_conformations

sim = simulation.Simulation(
    platform="cuda",  # Switch to platform="OpenCL" if you don't have cuda
    integrator="variableLangevin",
    error_tol=0.003,
    collision_rate=0.04,
    N=8000,
    verbose=True,
)


polymer = starting_conformations.create_random_walk(1, 8000)

sim.set_data(polymer, center=True)  # loads a polymer, puts a center of mass at zero

# -----------Adding forces ---------------
forcekit = forcekits.polymer_chains(
    sim,
    bond_force_kwargs={"bondWiggleDistance": 0.05},
    angle_force_kwargs={"k": 4},
    nonbonded_force_kwargs={"trunc": 1.5},
)
sim.add_force(forcekit)
sim.local_energy_minimization()
allDists = []
for _ in range(10):  # Do 10 blocks
    sim.do_block(2000)  # Of 2000 timesteps each
    data = sim.get_data()
    dists = np.sqrt(np.sum(np.diff(data, axis=0) ** 2, axis=1))
    allDists.append(dists)

allDists = np.concatenate(allDists)
plt.hist(allDists, 100)
plt.title("deviation should be around 0.05")

plt.show()
