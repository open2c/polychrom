import polychrom
from polychrom import simulation, starting_conformations, forces, forcekits
import numpy as np 


def test_harmonic_bond_force():
    N1 = 20 
    N = N1 * 2 

    for dist in [1,1.5]:
        for offset in [0, 0.05, 0.1, -0.05, -0.1, 2, -1]: 
            sim = simulation.Simulation(
                platform="CPU", 
                integrator="variableLangevin",
                error_tol=0.003,
                GPU="0",
                collision_rate=0.03,
                save_decimals=2,
                PBCbox=False,
                reporters=[],
                N = N
            )


            particles = [[0] * N1, [0] * N1, np.linspace(0,1000,N1)]
            particles = np.array(particles).T
            particles2 = particles.copy()
            particles2[:,0] += (dist + offset)

            bonds = [(i,i+N1) for i in range(N1)]

            data = np.concatenate([particles, particles2])

            sim.set_data(data, center=False)  # loads a polymer, puts a center of mass at zero
            sim.add_force(forces.harmonic_bonds(sim,bonds,bondLength=dist  ))
            res = sim.do_block(0)
            pot = res["potentialEnergy"] * 2   # only one bond per pair 
            assert np.allclose(pot, (offset / 0.05)**2, rtol=1e-3, atol=1e-3)


def run():
    test_harmonic_bond_force()

if __name__ == "__main__":
    run()