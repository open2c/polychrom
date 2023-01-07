r"""
Polymer simulations with CorrelatedNoiseIntegrator
--------------------------------------------------

This is a sample python script to run a polychrom simulation with the `CorrelatedNoiseIntegrator' custom integrator
in polychrom.contrib.integrators. This integrator is used to simulate a polymer where the Brownian forces acting on
distinct monomers could be correlated in direction. In addition, as in `ActiveBrownianIntegrator`, each monomer can
have a different diffusion coefficient :math:`D_i = k_B T_i / \xi`.

To define the correlations, we define a set of k attributes that specify the monomer's identity, such as charge,
methylation status, etc. For each attribute, all monomers are assigned type 1, type -1, or type 0. The integrator
defines a procedure by which type 1 monomers are correlated with one another with correlation coefficient
:math:`\rho`, type -1 monomers are correlated with one another with coefficient :math:`\rho`, but type 1 and type -1
monomers are anticorrelated with coefficient :math:`-\rho`. Type 0 monomers do not experience correlated fluctuations.

Here, we consider a simple example with just 1 feature (say, "charge"), and set all monomer diffusion coefficients to
be the same. Thus, all same-charge monomers will be correlated with correlation coefficient 0.5 and opposing charge
monomers are anticorrelated with coefficient -0.5. To visualize the Pearson correlation matrix of the N-dimensional
Gaussian noise that drives the polymer, use the `compute_Pearson_correlation_matrix()` function.

Run this script using
>>> python corr_noise.py [gpuid]

"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import openmm
from simtk import unit

import polychrom
from polychrom import forcekits, forces, simulation, starting_conformations
from polychrom.contrib.integrators import CorrelatedNoiseIntegrator
from polychrom.hdf5_format import HDF5Reporter

N = 100  # 100 monomers

# rhos is a (k, N) matrix where each row contains +rho, -rho, or 0 for the kth feature
rhos = 0.5 * np.ones((1, N))
rhos[0, 0:20] = -0.5
rhos[0, 20:40] = 0.0
rhos[0, 60:80] = 0.0


def compute_Pearson_correlation_matrix(rhomat):
    r"""Return the N x N Pearson correlation matrix that results from correlating same type monomers.

    Parameters
    ----------
    rhomat : array-like[float] (1, N)
        row vector of monomer types with entries +rho, -rho, or 0.

    Returns
    -------
    C : array-like[float] (N, N)
        Pearson correlation matrix (1s on diagonal, positive definite)

    Notes
    -----
    The overdamped Langevin equation for the ith monomer driven by active, correlated noise is

    .. math::
        \frac{dx(i, t)}{dt} =  \frac{1}{\xi}\vec{f}_{d} + \vec{\eta}_i(t)

    where :math:`\vec{f}_d` represents all deterministic forces and :math:`\vec{\eta}_i(t)` is a mean-zero
    Gaussian random velocity field with `:math:`\langle \eta_{ik} \eta_{jl} \rangle =
    2\sqrt{D_i}\sqrt{D_j}C_{ij}\delta_{kl}`. :math:`k, l` index the spatial components of the noise
    vector, and :math:`i, j` index the monomers."""

    N = rhomat.shape[1]
    rho = rhomat[rhomat > 0][0]
    idmat = rhomat / rho
    corr = np.outer(idmat, idmat)
    corr *= rho
    corr[np.diag_indices(N)] = 1.0
    return corr


def run_correlated_diffusion(gpuid, N, rhos, timestep=170, nblocks=10, blocksize=100):
    """Run a single simulation on a GPU of a hetero-polymer with monomers of type +, -, or 0.
    Same type monomers are positively correlated while opposite type monomers are anticorrelated.

    Parameters
    ----------
    gpuid : int
        which GPU to use. If on Mirny Lab machine, should be 0, 1, 2, or 3.
    N : int
        number of monomers in chain
    rhos : array-like[float] (k, N)
        each row vector of monomer types has entries +rho, -rho, or 0.
    timestep : int
        timestep to feed the Brownian integrator (in femtoseconds)
    nblocks : int
        number of blocks to run the simulation for. For a chain of 100 monomers, need ~10000 blocks of
        100 timesteps to equilibrate.
    blocksize : int
        number of time steps in a block

    """
    if rhos.shape[1] != N:
        raise ValueError("The array of monomer identities must have length equal to the total number of monomers.")
    # monomer density in confinement in units of monomers/volume
    density = 0.224
    r = (3 * N / (4 * 3.141592 * density)) ** (1 / 3)
    print(f"Radius of confinement: {r}")
    D = np.ones((N, 3))  # Diffusion coefficients of N monomers in x,y,z spatial dimensions
    timestep = timestep
    # the monomer diffusion coefficient should be in units of kT / friction, where friction = mass*collision_rate
    collision_rate = 2.0
    mass = 100 * unit.amu
    friction = collision_rate * (1.0 / unit.picosecond) * mass
    temperature = 300
    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kT = kB * temperature * unit.kelvin
    particleD = unit.Quantity(D, kT / friction)
    integrator = CorrelatedNoiseIntegrator(timestep, collision_rate, particleD, rhos)
    gpuid = f"{gpuid}"
    reporter = HDF5Reporter(folder="correlations", max_data_length=100, overwrite=True)
    sim = simulation.Simulation(
        platform="CUDA",
        # for custom integrators, feed a tuple with the integrator class reference and a string specifying type,
        # e.g. "brownian", "variableLangevin", "variableVerlet", or simply "UserDefined" if none of the above.
        integrator=(integrator, "brownian"),
        timestep=timestep,
        temperature=temperature,
        GPU=gpuid,
        collision_rate=collision_rate,
        N=N,
        save_decimals=2,
        PBCbox=False,
        reporters=[reporter],
    )

    polymer = starting_conformations.grow_cubic(N, 5)
    sim.set_data(polymer, center=True)  # loads a polymer, puts a center of mass at zero
    sim.set_velocities(v=np.zeros((N, 3)))  # set initial velocities to 0 (no inertia)
    sim.add_force(forces.spherical_confinement(sim, density=density, k=5.0))
    sim.add_force(
        forcekits.polymer_chains(
            sim,
            chains=[(0, None, False)],
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                "bondLength": 1.0,
                "bondWiggleDistance": 0.3,  # Bond distance will fluctuate +- 0.05 on average
            },
            angle_force_func=None,
            angle_force_kwargs={},
            nonbonded_force_func=forces.polynomial_repulsive,
            nonbonded_force_kwargs={
                "trunc": 3.0,  # this will let chains cross sometimes
                # 'trunc':10.0, # this will resolve chain crossings and will not let chain cross anymore
            },
            except_bonds=True,
        )
    )
    tic = time.perf_counter()
    for _ in range(nblocks):  # Do nblocks blocks
        sim.do_block(blocksize)  # Of 100 timesteps each. Data is saved automatically.
    toc = time.perf_counter()
    print(f"Ran simulation in {(toc - tic):0.4f}s")
    sim.print_stats()  # In the end, print very simple statistics
    reporter.dump_data()  # always need to run in the end to dump the block cache to the disk


if __name__ == "__main__":
    gpuid = int(sys.argv[1])
    run_correlated_diffusion(gpuid, N, rhos)
