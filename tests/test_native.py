"""Tests of the native CPU platform against OpenMM's Reference platform."""

import numpy as np
import pytest

import polychrom.starting_conformations
from polychrom import forcekits, forces
from polychrom.simulation import Simulation

native = pytest.importorskip("polychrom.native")


def _make(platform, N, data, build, PBCbox=False, integrator="langevin", collision_rate=0.1):
    sim = Simulation(
        platform=platform,
        integrator=integrator,
        timestep=70,
        error_tol=0.01,
        collision_rate=collision_rate,
        N=N,
        reporters=[],
        PBCbox=PBCbox,
        verbose=False,
    )
    sim.set_data(data, random_offset=0)
    build(sim)
    sim._apply_forces()
    return sim


def _forces_energy(sim):
    st = sim.context.getState(getForces=True, getEnergy=True)
    return np.asarray(st.getForces(asNumpy=True)._value), st.getPotentialEnergy()._value


def _compare(build, N=300, PBCbox=False, seed=0, rtol=2e-6, data=None):
    rng = np.random.default_rng(seed)
    if data is None:
        data = polychrom.starting_conformations.grow_cubic(N, 12) + rng.normal(0, 0.15, (N, 3))
    ref = _make("reference", N, data, build, PBCbox)
    nat = _make("native", N, data, build, PBCbox)
    f_ref, e_ref = _forces_energy(ref)
    f_nat, e_nat = _forces_energy(nat)
    scale = np.abs(f_ref).max()
    np.testing.assert_allclose(f_nat, f_ref, atol=rtol * scale, rtol=0)
    assert e_nat == pytest.approx(e_ref, rel=rtol, abs=1e-8)
    return ref, nat


def _chain(**kw):
    def build(sim):
        sim.add_force(forcekits.polymer_chains(sim, **kw))

    return build


def test_polymer_chain_open():
    _compare(
        _chain(
            bond_force_kwargs={"bondLength": 1.0, "bondWiggleDistance": 0.1},
            angle_force_kwargs={"k": 1.5},
            nonbonded_force_kwargs={"trunc": 1.5, "radiusMult": 1.05},
        )
    )


def test_polymer_chain_pbc():
    _compare(
        _chain(
            bond_force_kwargs={"bondLength": 1.0, "bondWiggleDistance": 0.1},
            angle_force_kwargs={"k": 1.5},
            nonbonded_force_kwargs={"trunc": 3.0},
        ),
        PBCbox=[9.0, 9.0, 9.0],
    )


def test_smooth_square_well_rings_and_angle_offset():
    def build(sim):
        sim.add_force(
            forcekits.polymer_chains(
                sim,
                chains=[(0, 150, True), (150, 300, False)],
                angle_force_kwargs={"k": 2.0, "theta_0": 2.5},  # not flat at pi: theta-table path
                nonbonded_force_func=forces.smooth_square_well,
                nonbonded_force_kwargs={"attractionEnergy": 0.3, "attractionRadius": 1.7},
            )
        )

    _compare(build)


def test_heteropolymer_types_confinement_tether():
    N = 300
    types = np.arange(N) % 3

    def build(sim):
        sim.add_force(
            forcekits.polymer_chains(
                sim,
                nonbonded_force_func=forces.heteropolymer_SSW,
                nonbonded_force_kwargs=dict(
                    interactionMatrix=np.array([[0.1, 0.4, 0.0], [0.4, 0.2, 0.3], [0.0, 0.3, 0.5]]),
                    monomerTypes=types,
                    extraHardParticlesIdxs=[5, 17],
                ),
            )
        )
        sim.add_force(forces.spherical_confinement(sim, r=5.5, k=3))
        sim.add_force(forces.tether_particles(sim, [0, 10, -1], k=[1, 2, 3]))

    _compare(build, N=N)


def test_constant_force_bonds():
    def build(sim):
        sim.add_force(
            forcekits.polymer_chains(
                sim,
                bond_force_func=forces.constant_force_bonds,
                bond_force_kwargs={"bondWiggleDistance": 0.1},
            )
        )

    _compare(build, rtol=1e-5)


def test_grosberg_rings():
    N = 200
    data = polychrom.starting_conformations.grow_cubic(N, 12)  # lattice: FENE bonds within range

    def build(sim):
        sim.add_force(forcekits.grosberg_polymer_chains(sim, chains=[(0, None, True)]))

    with pytest.warns(UserWarning, match="extrapolates linearly"):
        _compare(build, N=N, data=data, rtol=1e-5)


def test_update_bonds_and_parameters():
    """Loop-extrusion style bond updates and global parameter changes."""
    N = 300
    lef = {}

    def build(sim):
        sim.add_force(forcekits.polymer_chains(sim))
        f = forces.harmonic_bonds(sim, [(10, 50), (100, 180)], bondWiggleDistance=0.2, name="lef", override_checks=True)
        lef[id(sim)] = f
        sim.add_force(f)

    ref, nat = _compare(build, N=N)
    for sim in (ref, nat):
        f = lef[id(sim)]
        f.setBondParameters(0, 10, 50, 0.5, 0.0)  # like bondUpdater: same particles, new parameters
        f.setBondParameters(1, 100, 180, 0.5, 500.0)
        f.updateParametersInContext(sim.context)
        sim.context.setParameter("polynomial_repulsive_REPe", 4.0)
    f_ref, e_ref = _forces_energy(ref)
    f_nat, e_nat = _forces_energy(nat)
    np.testing.assert_allclose(f_nat, f_ref, atol=2e-6 * np.abs(f_ref).max(), rtol=0)
    assert e_nat == pytest.approx(e_ref, rel=2e-6)


def test_unsupported_force_raises():
    def build(sim):
        sim.add_force(forcekits.polymer_chains(sim))
        sim.add_force(forces.pull_force(sim, [0], [[1.0, 0, 0]]))

    with pytest.raises(NotImplementedError, match="Pull"):
        _make("native", 50, polychrom.starting_conformations.grow_cubic(50, 8), build)


@pytest.mark.parametrize("integrator", ["langevin", "variableLangevin", "langevinMiddle", "brownian"])
def test_integrators_run_and_thermalise(integrator):
    N = 400
    data = polychrom.starting_conformations.grow_cubic(N, 14)
    # parameters of the loop extrusion example; Brownian dynamics needs strong friction for a 70 fs step
    build = _chain(
        bond_force_kwargs={"bondLength": 1.0, "bondWiggleDistance": 0.1},
        angle_force_kwargs={"k": 1.5},
        nonbonded_force_kwargs={"trunc": 1.5, "radiusMult": 1.05},
    )
    sim = _make("native", N, data, build, integrator=integrator, collision_rate=30 if integrator == "brownian" else 0.1)
    sim.integrator.setRandomNumberSeed(7)
    for _ in range(4):
        res = sim.do_block(200, save=False)
    assert np.isfinite(res["pos"]).all()
    if integrator != "brownian":
        assert 1.0 < res["kineticEnergy"] < 2.0  # 1.5 kT per particle


def test_trajectory_independent_of_threads():
    N = 500
    data = polychrom.starting_conformations.grow_cubic(N, 14)
    out = []
    for threads in (1, 3):
        sim = _make("native", N, data, _chain(), PBCbox=[20.0, 20.0, 20.0])
        sim.context.engine.set_threads(threads)
        sim.integrator.setRandomNumberSeed(11)
        sim.context.setVelocitiesToTemperature(300, 5)
        sim.integrator.step(300)
        out.append(sim.context.getState(getPositions=True).getPositions(asNumpy=True)._value)
    np.testing.assert_array_equal(out[0], out[1])
