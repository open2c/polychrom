#!/usr/bin/env python3
"""Builds topology_preserving_rings.ipynb (run once; the .ipynb is committed)."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

cells.append(md(r"""# Topology-preserving ring melt simulations

This notebook shows how to run simulations in which chains **cannot pass through
each other**, so the global topology (knots and links) is exactly conserved —
and, just as importantly, how to *verify* that it actually is.

The force set is the Kremer–Grest / Halverson–Grosberg model (the `grosberg_*`
forces in polychrom): WCA repulsion + FENE bonds + mild bending, the standard
model for ring melts and topological polymer physics since
[Halverson et al., JCP 134, 204904 (2011)](https://arxiv.org/abs/1104.5653):

* **WCA (purely repulsive LJ), ε = 1 kT** — `grosberg_repulsive_force(trunc=None)`.
  Chains cannot cross: the barrier is ~70 kT.
* **FENE bonds, k = 30 kT/σ², r₀ = 1.5 σ** — `grosberg_polymer_bonds`. Bonds
  cannot stretch past 1.5 σ, so a chain cannot slip between two bonded beads.
* **Bending, k = 1.5 kT** — `grosberg_angle`. This mild stiffness *reduces* the
  entanglement length to Nₑ ≈ 28 (vs ≈ 70 for flexible chains), so topological
  effects appear with fewer monomers per chain.
* **Monomer density 0.85 σ⁻³** in periodic boundary conditions — the canonical
  melt density for this model.

All three forces are packaged in the forcekit
`polychrom.forcekits.grosberg_polymer_chains`, which also takes care of a
critical detail: bonded neighbors must **not** be excluded from the nonbonded
force (`except_bonds=False`), because the FENE potential alone has its minimum
at r = 0 — the ~0.97 σ bond length comes from the balance with WCA."""))

cells.append(md(r"""## Units and validated numerical parameters

polychrom uses OpenMM units (nm, ps, amu, kT at 300 K). With σ = 1 nm,
m = 100 amu, ε = kT: **τ_LJ = σ√(m/ε) = 6.33 ps**, so LJ timesteps map to:

| dt in τ_LJ | dt in fs | status (measured on RTX 4090, 25.6k beads) |
|---|---|---|
| 0.010 | 63 | canonical (Halverson/Virnau); rock solid |
| 0.013 | 82 | **recommended**: validated topologically exact over 30 000 τ |
| 0.015 | 95 | metastable — survives ~10³ τ, explodes on long runs. Avoid. |
| 0.020 | 127 | explodes (the WCA pair force is the limit, not FENE) |

Other validated facts worth knowing:

* Use a **fixed-timestep** Langevin integrator (`"langevinMiddle"`).
  Variable-timestep integrators (`"variableLangevin"`, the polychrom default
  for soft chromatin force fields) **explode with this stiff force set at any
  error tolerance** — the error controller reacts only after a step, and one
  step past the FENE divergence is unrecoverable.
* `collision_rate = 0.079 /ps` (γ = 0.5/τ, the literature value). Lowering to
  γ ≈ 0.05–0.15/τ speeds up diffusive sampling ~20% (motion becomes ballistic
  between collisions), but at dt = 0.013–0.015 τ weak damping destabilizes the
  run; γ = 0.5/τ is the safe default.
* Warm up a few hundred τ at a small timestep (~30–40 fs) after energy
  minimization before switching to the production timestep."""))

cells.append(code(r"""import time

import numpy as np

import polychrom
from polychrom import forcekits, simulation, starting_conformations
from polychrom.polymer_analyses import alexander_invariants, getLinkingNumber

TAU_FS = 6332.0  # tau_LJ in fs for polychrom default units (sigma=1nm, m=100amu, T=300K)

# --- system geometry: M rings x N monomers at density 0.85 -------------------
RING_N = 180           # monomers per ring (~6.4 entanglement lengths)
CELLS = 3              # rings are grown in a CELLS^3 grid of cubic cells
M_RINGS = CELLS**3     # 27 rings, 4860 monomers total
DENSITY = 0.85

N_TOTAL = M_RINGS * RING_N
L = (N_TOTAL / DENSITY) ** (1 / 3)  # periodic box side, sigma
print(f"{M_RINGS} rings x {RING_N} = {N_TOTAL} monomers, PBC box L = {L:.2f} sigma")"""))

cells.append(md(r"""## Unentangled initial conformation

Each ring is grown as a closed lattice loop with `grow_cubic` (unknotted by
construction) inside its own cubic cell; the cells tile the box, so different
rings are unlinked by construction too. The lattice (bond length 1) is then
squeezed affinely by ~3% to reach density 0.85 — conveniently landing on the
KG equilibrium bond length ≈ 0.97 σ."""))

cells.append(code(r"""def build_unentangled_rings(n_side, ring_n, density, seed=0):
    rng = np.random.default_rng(seed)
    cell = int((ring_n / 0.8) ** (1 / 3))  # lattice cell filled to ~80%
    while ring_n > 0.87 * cell**3:
        cell += 1
    coords = []
    for i in range(n_side):
        for j in range(n_side):
            for k in range(n_side):
                ring = np.array(starting_conformations.grow_cubic(ring_n, cell), float)
                ring += rng.uniform(-0.05, 0.05, ring.shape)  # break lattice degeneracy
                ring += np.array([i, j, k]) * cell
                coords.append(ring)
    coords = np.concatenate(coords)
    L0 = n_side * cell
    L = (len(coords) / density) ** (1 / 3)
    return coords * (L / L0), L


coords, L = build_unentangled_rings(CELLS, RING_N, DENSITY)
rings0 = coords.reshape(M_RINGS, RING_N, 3)
print(f"box {L:.2f} sigma, bond length after squeeze: "
      f"{np.linalg.norm(rings0[0][1] - rings0[0][0]):.3f} sigma")"""))

cells.append(md(r"""### Verify the starting topology

`alexander_invariants(ring)` returns |Δ(−1)| and the odd part of |Δ(−2)| of the
Alexander polynomial, computed exactly (integer arithmetic, after
topology-preserving simplification by the compiled `simplifyPolymer`).
**(1, 1) = unknot**; a trefoil gives (3, 7), figure-eight (5, 11), 5₁ (5, 31).

`getLinkingNumber(ring_a, ring_b)` is the exact Gauss linking number
(a Hopf link gives ±1; 0 = unlinked)."""))

cells.append(code(r"""rng = np.random.default_rng(0)

t0 = time.time()
knot_invariants = [alexander_invariants(rings0[r], rng=rng) for r in range(M_RINGS)]
n_knotted = sum(inv != (1, 1) for inv in knot_invariants)

links0 = sum(
    getLinkingNumber(rings0[i], rings0[j]) != 0
    for i in range(M_RINGS) for j in range(i + 1, M_RINGS)
)
print(f"initial state: {n_knotted} knotted rings, {links0} linked pairs "
      f"(checked in {time.time()-t0:.1f} s) — must both be 0")"""))

cells.append(md(r"""## Setting up the simulation

Everything force-related is one forcekit call. Note the integrator choice and
timestep per the table above; `chains` lists every ring as `(start, end, True)`."""))

cells.append(code(r"""def make_sim(coords, L, timestep_fs, crossable=False, seed=42):
    sim = simulation.Simulation(
        platform="CUDA",            # or "CPU" — same physics, much slower
        integrator="langevinMiddle",  # FIXED timestep; never variableLangevin here
        timestep=timestep_fs,
        collision_rate=0.079,       # 1/ps == 0.5 per tau_LJ
        N=len(coords),
        PBCbox=(L, L, L),
        reporters=[],               # add an HDF5Reporter for production runs
        verbose=False,
    )
    sim.set_data(coords, center=False)
    chains = [(r * RING_N, (r + 1) * RING_N, True) for r in range(M_RINGS)]
    if not crossable:
        sim.add_force(forcekits.grosberg_polymer_chains(sim, chains=chains))
    else:
        # crossable control: soft (truncated) repulsion NEEDS harmonic bonds —
        # FENE alone has its minimum at r=0 and collapses without full WCA
        from polychrom import forces
        sim.add_force(forcekits.polymer_chains(
            sim, chains=chains,
            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={"bondLength": 1.0, "bondWiggleDistance": 0.2},
            angle_force_func=forces.grosberg_angle,
            angle_force_kwargs={"k": 1.5},
            nonbonded_force_func=forces.grosberg_repulsive_force,
            nonbonded_force_kwargs={"trunc": 1.5},
            except_bonds=False,
        ))
    return sim


sim = make_sim(coords, L, timestep_fs=40)  # start at a gentle warmup timestep
sim.local_energy_minimization(tolerance=0.3, maxIterations=300)"""))

cells.append(code(r"""# warmup at dt = 40 fs (~0.006 tau), then switch to production dt = 82 fs (0.013 tau)
import openmm
import simtk.unit as unit

sim.do_block(20000)  # ~130 tau of warmup

sim.integrator.setStepSize(82 * unit.femtosecond)
t0 = time.time()
n_blocks, steps_per_block = 10, 5000
for _ in range(n_blocks):
    sim.do_block(steps_per_block)  # 50 000 steps = ~650 tau of production
rate = n_blocks * steps_per_block / (time.time() - t0)
print(f"\nproduction rate: {rate:,.0f} steps/s "
      f"({rate * 82e-6 * 1e3 / 6.332:,.1f} tau per wall-second)")"""))

cells.append(md(r"""## Verify topology after the run

Knots first: every ring must still be an unknot. Then links: since the box is
periodic, a ring can link with a *periodic image* of another ring, so we check
image shifts too (for speed, only for pairs whose bounding spheres overlap)."""))

cells.append(code(r"""pos = sim.get_data()
rings = pos.reshape(M_RINGS, RING_N, 3)

# --- knots -------------------------------------------------------------------
invariants = [alexander_invariants(rings[r], rng=rng) for r in range(M_RINGS)]
knotted = [(r, inv) for r, inv in enumerate(invariants) if inv != (1, 1)]
print(f"knotted rings after run: {len(knotted)}   {knotted if knotted else ''}")

# --- links (periodic-image aware) -------------------------------------------
com = rings.mean(axis=1)
rad = np.sqrt(((rings - com[:, None]) ** 2).sum(axis=2).max(axis=1))
shifts = L * np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)])

linked, checked = [], 0
for i in range(M_RINGS):
    for j in range(i, M_RINGS):
        for s in shifts:
            if i == j and not s.any():
                continue
            if np.linalg.norm(com[j] + s - com[i]) > rad[i] + rad[j] + 1.0:
                continue
            checked += 1
            lk = getLinkingNumber(rings[i], rings[j] + s)
            if lk != 0:
                linked.append((i, j, int(lk)))
print(f"linked pairs after run: {len(linked)} (of {checked} image-aware pair checks)")
assert not knotted and not linked, "topology violation detected!"
print("\ntopology exactly preserved ✓")"""))

cells.append(md(r"""## Positive control: what a *crossable* system looks like

Truncating the repulsion (`trunc`, in kT) caps the overlap energy — this is how
soft chromatin force fields allow strand passing (topo-II-style). At melt
density even `trunc=3` leaks heavily. Two things to note:

* soft repulsion requires **harmonic** bonds — FENE + truncated repulsion
  collapses (FENE alone has its minimum at r = 0) and the run NaNs, which is
  why `grosberg_polymer_chains` deliberately has no `trunc` parameter;
* this cell doubles as a check that the detector *detects*: the same analysis
  on a crossable system must light up."""))

cells.append(code(r"""sim_soft = make_sim(coords, L, timestep_fs=63, crossable=True)
sim_soft.local_energy_minimization(tolerance=0.3, maxIterations=300)
sim_soft.do_block(30000)  # ~300 tau — plenty for crossings at trunc=1.5

rings_soft = sim_soft.get_data().reshape(M_RINGS, RING_N, 3)
inv_soft = [alexander_invariants(rings_soft[r], rng=rng) for r in range(M_RINGS)]
n_knot_soft = sum(inv != (1, 1) for inv in inv_soft)
n_link_soft = sum(
    getLinkingNumber(rings_soft[i], rings_soft[j]) != 0
    for i in range(M_RINGS) for j in range(i + 1, M_RINGS)
)
print(f"trunc=1.5 control after ~300 tau: {n_knot_soft} knotted rings, "
      f"{n_link_soft} linked pairs (nonzero = detector works, crossable model crosses)")"""))

cells.append(md(r"""## Summary / checklist for production runs

1. **Forces**: `forcekits.grosberg_polymer_chains(sim, chains=..., trunc=None)`.
   Never combine the grosberg forces with `except_bonds=True`.
2. **Integrator**: `"langevinMiddle"`, `timestep=63–82 fs`, `collision_rate=0.079 /ps`.
   Never `variableLangevin` with this force set.
3. **Start**: unknotted/unlinked by construction (`grow_cubic` per cell) →
   minimize → warm up at ~40 fs → production timestep.
4. **Verify, don't assume**: run `alexander_invariants` on every ring and
   image-aware `getLinkingNumber` on neighboring pairs every ~10⁶ steps
   (the checks run on CPU and can overlap with the GPU simulation).
   Expected values: (1,1) and 0. The positive control above shows what
   violations look like.
5. Throughput reference (RTX 4090): ~3×10⁸ particle-steps/s at 25k particles,
   ~7×10⁸ at 200k — i.e. this notebook's little system is far below the GPU's
   saturation point; production systems can be much larger at similar wall time
   per τ."""))

nb.cells = cells
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "topology_preserving_rings.ipynb")
print("written")
