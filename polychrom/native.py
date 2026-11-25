"""
Native CPU backend: ``Simulation(platform="native")``
=====================================================

A specialised multi-threaded CPU engine (``polychrom/_native/engine.hpp``) that
runs polychrom simulations without an OpenMM context.  It is much faster than
OpenMM's CPU platform for polymer models, because it avoids per-step overheads
that are irrelevant for coarse-grained polymers (see the engine header for the
techniques used).

The backend is a drop-in replacement: forces are still created by the functions
in :mod:`polychrom.forces` as OpenMM force objects, and ``Simulation.context``
and ``Simulation.integrator`` provide the subset of the OpenMM ``Context`` and
``Integrator`` API that polychrom code uses, including
``force.updateParametersInContext(sim.context)`` for loop extrusion.

How forces are translated
-------------------------

* ``HarmonicBondForce`` is evaluated natively.
* ``CustomBondForce``, ``CustomAngleForce`` and ``CustomNonbondedForce`` are
  tabulated: a two- or three-particle copy of the force (same expression,
  global parameters, tabulated functions, switching function) is evaluated with
  OpenMM's Reference platform on a fine grid, once per distinct set of
  per-bond/angle/particle parameters, and interpolated with cubic Hermite
  splines that use the exact energies and derivatives.  Any energy expression
  therefore works, as long as it depends only on the distance (angle) and the
  number of distinct parameter sets is moderate.
* ``CustomExternalForce`` is supported for
  :func:`polychrom.forces.spherical_confinement` and
  :func:`polychrom.forces.tether_particles`.

Anything else raises :class:`NotImplementedError` when the context is created.

Integrators: ``langevin``, ``langevinMiddle``, ``variableLangevin`` and
``brownian`` follow the algorithms of the corresponding OpenMM Reference
implementations.  Random numbers differ from OpenMM's, so trajectories agree
statistically, not bit by bit.

Threads: ``POLYCHROM_NATIVE_THREADS``, else ``OMP_NUM_THREADS``, else
``OPENMM_CPU_THREADS``, else all cores.
"""

import copy
import os
import re
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import openmm
import simtk.unit as u

from polychrom import _native_engine

nm = u.nanometer
ps = u.picosecond
kJ_mol = u.kilojoule_per_mole

# Tables (energy and derivative per knot) start with this many knots and are
# refined by doubling until the cubic Hermite interpolant of the derivative is
# within TABLE_TOL (relative to its maximum) of OpenMM at the interval midpoints.
PAIR_KNOTS = (513, 8193)
ANGLE_KNOTS = (1025, 8193)
BOND_KNOTS = (2049, 16385)
TABLE_TOL = 1e-6
TABLE_WARN = 1e-4  # warn if a table cannot reach this accuracy
TABLE_E_CAP = 250.0  # kJ/mol (~100 kT) above the minimum: steeper regions are never sampled and not checked
MAX_TABLES = 64  # per force; more distinct parameter sets are rejected


def _val(x, unit=None):
    if u.is_quantity(x):
        return x.value_in_unit(unit) if unit is not None else x._value
    return x


def _array(x, unit):
    """Quantity / list of Vec3 / array -> float64 numpy array in `unit`."""
    if u.is_quantity(x):
        x = x.value_in_unit(unit)
    return np.asarray([list(v) for v in x] if isinstance(x, (list, tuple)) else x, dtype=np.float64)


def _thread_count():
    for var in ("POLYCHROM_NATIVE_THREADS", "OMP_NUM_THREADS", "OPENMM_CPU_THREADS"):
        v = os.environ.get(var)
        if v:
            return int(v)
    return 0  # OpenMP default


class NativePlatform:
    """Stand-in for an OpenMM Platform object."""

    def getName(self):
        return "native"

    def getSpeed(self):
        return 10.0


# ---------------------------------------------------------------------------
# Integrator
# ---------------------------------------------------------------------------


class NativeIntegrator:
    """Stand-in for an OpenMM integrator; steps the native engine of its context."""

    KINDS = ("langevin", "langevinmiddle", "variablelangevin", "brownian")

    def __init__(self, kind, temperature, collision_rate, timestep=None, error_tol=None):
        kind = kind.lower()
        if kind not in self.KINDS:
            raise NotImplementedError(
                f"integrator {kind!r} is not supported by the native platform; use one of {self.KINDS}"
            )
        self.kind = kind
        self._temperature = float(_val(temperature, u.kelvin))
        self._friction = float(_val(collision_rate, 1 / ps))
        self._dt = float(_val(timestep, ps)) if timestep is not None else 0.0
        self._error_tol = float(error_tol) if error_tol is not None else 0.0
        self._seed = 0
        self._max_dt = 0.0
        self._engine = None
        if kind == "variablelangevin" and not self._error_tol > 0:
            raise ValueError("variableLangevin needs error_tol > 0")
        if kind != "variablelangevin" and not self._dt > 0:
            raise ValueError(f"{kind} needs a positive timestep")

    def _bind(self, engine):
        self._engine = engine
        engine.set_integrator(self.kind, self._temperature, self._friction, self._dt, self._error_tol)
        seed = self._seed if self._seed != 0 else int.from_bytes(os.urandom(8), "little") or 1
        engine.set_seed(seed)
        if self._max_dt > 0:
            engine.set_max_step_size(self._max_dt)

    def _require_engine(self):
        if self._engine is None:
            raise RuntimeError("integrator is not bound to a context yet")
        return self._engine

    def step(self, steps):
        self._require_engine().step(int(steps))

    def getStepSize(self):
        dt = self._engine.get_step_size() if self._engine is not None else self._dt
        return dt * ps

    def setStepSize(self, dt):
        self._dt = float(_val(dt, ps))
        if self._engine is not None:
            self._engine.set_step_size(self._dt)

    def getTemperature(self):
        return self._temperature * u.kelvin

    def setTemperature(self, t):
        self._temperature = float(_val(t, u.kelvin))
        if self._engine is not None:
            self._engine.set_temperature(self._temperature)

    def getFriction(self):
        return self._friction / ps

    def setFriction(self, g):
        self._friction = float(_val(g, 1 / ps))
        if self._engine is not None:
            self._engine.set_friction(self._friction)

    def getErrorTolerance(self):
        return self._error_tol

    def setErrorTolerance(self, tol):
        self._error_tol = float(tol)
        if self._engine is not None:
            self._engine.set_error_tolerance(self._error_tol)

    def setMaximumStepSize(self, dt):
        self._max_dt = float(_val(dt, ps))
        if self._engine is not None:
            self._engine.set_max_step_size(self._max_dt)

    def getRandomNumberSeed(self):
        return self._seed

    def setRandomNumberSeed(self, seed):
        self._seed = int(seed)
        if self._engine is not None and self._seed != 0:
            self._engine.set_seed(self._seed)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class NativeState:
    """Stand-in for an OpenMM State."""

    def __init__(self, time, step, positions=None, velocities=None, forces=None, ekin=None, epot=None, box=None):
        self._time, self._step = time, step
        self._pos, self._vel, self._forces = positions, velocities, forces
        self._ekin, self._epot, self._box = ekin, epot, box

    @staticmethod
    def _need(x, what):
        if x is None:
            raise RuntimeError(f"{what} were not requested in getState()")
        return x

    def getTime(self):
        return self._time * ps

    def getStepCount(self):
        return self._step

    def getPositions(self, asNumpy=False):
        x = self._need(self._pos, "positions")
        return x * nm if asNumpy else [openmm.Vec3(*v) for v in x] * nm

    def getVelocities(self, asNumpy=False):
        x = self._need(self._vel, "velocities")
        return x * (nm / ps) if asNumpy else [openmm.Vec3(*v) for v in x] * (nm / ps)

    def getForces(self, asNumpy=False):
        x = self._need(self._forces, "forces")
        return x * (kJ_mol / nm) if asNumpy else [openmm.Vec3(*v) for v in x] * (kJ_mol / nm)

    def getKineticEnergy(self):
        return self._need(self._ekin, "energies") * kJ_mol

    def getPotentialEnergy(self):
        return self._need(self._epot, "energies") * kJ_mol

    def getPeriodicBoxVectors(self, asNumpy=False):
        b = self._box
        vecs = [openmm.Vec3(b[0], 0, 0), openmm.Vec3(0, b[1], 0), openmm.Vec3(0, 0, b[2])]
        return (np.diag(b) * nm) if asNumpy else vecs * nm


# ---------------------------------------------------------------------------
# Tabulation of custom forces
# ---------------------------------------------------------------------------


class _Batch:
    """Evaluate a small configuration (pair, bond or angle) of a force for many
    values of its coordinate at once: SLOTS well-separated copies live in one
    Reference-platform context, so one getState() call yields SLOTS exact
    derivatives.  Energies of a single configuration come from filling all slots
    with it.
    """

    SLOTS = 1024

    def __init__(self, root, kind, params, spacing):
        self.kind = kind
        self.spacing = spacing
        M = self.SLOTS
        # Copy everything but the (possibly huge) per-particle/bond/angle lists.
        big, tag = {
            "pair": (("Particles", "Exclusions", "InteractionGroups"), "Particle"),
            "bond": (("Bonds",), "Bond"),
            "angle": (("Angles",), "Angle"),
        }[kind]
        small = ET.Element(root.tag, dict(root.attrib))
        for child in root:
            if child.tag in big:
                ET.SubElement(small, child.tag)
            else:
                small.append(copy.deepcopy(child))
        root = small
        if kind == "pair":
            root.set("method", "1")  # CutoffNonPeriodic
            root.set("useLongRangeCorrection", "0")
            container, per = root.find("Particles"), 2
        else:
            root.set("usesPeriodic", "0")
            container, per = root.find("Bonds" if kind == "bond" else "Angles"), 2 if kind == "bond" else 3
        proto = tag
        for k in range(M):
            if kind == "pair":
                for prm in params:
                    e = ET.SubElement(container, proto)
                    for q, v in enumerate(prm):
                        e.set(f"param{q + 1}", repr(v))
            else:
                e = ET.SubElement(container, proto)
                for q in range(per):
                    e.set(f"p{q + 1}", str(per * k + q))
                for q, v in enumerate(params):
                    e.set(f"param{q + 1}", repr(v))
        self.per = per
        system = openmm.System()
        for _ in range(per * M):
            system.addParticle(1.0)
        system.addForce(openmm.XmlSerializer.deserialize(ET.tostring(root, encoding="unicode")))
        self.ctx = openmm.Context(
            system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference")
        )
        self.offsets = np.zeros((M, 3))
        self.offsets[:, 1] = np.arange(M) * spacing

    def _positions(self, xs):
        M = self.SLOTS
        pos = np.zeros((M, self.per, 3))
        pos += self.offsets[:, None, :]
        if self.kind == "angle":  # i on the x axis, j at the origin, k on the unit circle
            pos[:, 0, 0] += 1.0
            pos[:, 2, 0] += np.cos(xs)
            pos[:, 2, 1] += np.sin(xs)
        else:  # second particle on the x axis
            pos[:, 1, 0] += xs
        return pos.reshape(-1, 3)

    def derivative(self, xs):
        """dE/dx at each x (x = distance, or angle in radians)."""
        xs = np.asarray(xs, dtype=float)
        out = np.empty(len(xs))
        M = self.SLOTS
        for start in range(0, len(xs), M):
            chunk = xs[start : start + M]
            full = np.full(M, chunk[0])
            full[: len(chunk)] = chunk
            self.ctx.setPositions(self._positions(full) * nm)
            F = self.ctx.getState(getForces=True).getForces(asNumpy=True).value_in_unit(kJ_mol / nm)
            F = F.reshape(M, self.per, 3)
            if self.kind == "angle":  # dE/dtheta = -f_k . e_theta
                D = -(F[:, 2, 0] * -np.sin(full) + F[:, 2, 1] * np.cos(full))
            else:  # dE/dr = -f_2 . e_x
                D = -F[:, 1, 0]
            out[start : start + len(chunk)] = D[: len(chunk)]
        return out

    def energy(self, x):
        self.ctx.setPositions(self._positions(np.full(self.SLOTS, float(x))) * nm)
        return self.ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kJ_mol) / self.SLOTS


def _hermite_derivative(x, E, D, xq):
    """Derivative of the cubic Hermite interpolant (uniform knots x, values E, slopes D) at xq."""
    h = x[1] - x[0]
    k = np.clip(((xq - x[0]) / h).astype(int), 0, len(x) - 2)
    t = (xq - x[k]) / h
    e0, e1, m0, m1 = E[k], E[k + 1], D[k] * h, D[k + 1] * h
    c1, c2, c3 = m0, -3 * e0 - 2 * m0 + 3 * e1 - m1, 2 * e0 + m0 - 2 * e1 + m1
    return (c1 + t * (2 * c2 + t * 3 * c3)) / h


def _integrate(x, D, Dm, energy_at, anchor, what):
    """E on knots x by Simpson integration of D (knots) and Dm (midpoints).

    Integrates over the run of finite intervals that contains the knot nearest to
    `anchor`, where E is evaluated directly.  Returns (E, lo, hi): E is NaN
    outside the finite window [lo, hi] of knot indices.
    """
    n = len(x)
    h = x[1] - x[0]
    ok_knot = np.isfinite(D)
    ok_int = ok_knot[1:] & ok_knot[:-1] & np.isfinite(Dm)
    ia = int(np.clip(round((anchor - x[0]) / h), 0, n - 1))
    if not ok_knot[ia]:
        finite = np.flatnonzero(ok_knot)
        if len(finite) == 0:
            raise ValueError(f"{what}: the force is not finite anywhere on the table range")
        ia = finite[np.argmin(np.abs(finite - ia))]
    lo = ia
    while lo > 0 and ok_int[lo - 1]:
        lo -= 1
    hi = ia
    while hi < n - 1 and ok_int[hi]:
        hi += 1
    E = np.full(n, np.nan)
    with np.errstate(all="ignore"):
        integ = h / 6 * (D[:-1] + 4 * Dm + D[1:])  # E[k+1] - E[k]
    E[ia] = energy_at(x[ia])
    if not np.isfinite(E[ia]):
        raise ValueError(f"{what}: energy is not finite at x = {x[ia]}")
    E[ia + 1 : hi + 1] = E[ia] + np.cumsum(integ[ia:hi])
    E[lo:ia] = E[ia] - np.cumsum(integ[lo:ia][::-1])[::-1]
    return E, lo, hi


def _quarter_error(deriv, x, E, D, lo, hi):
    """Relative error of the interpolated derivative at the quarter points of
    the finite intervals whose energy is within TABLE_E_CAP of the minimum;
    returns (max error, interval index of the max, xq, Dq)."""
    h = x[1] - x[0]
    n = len(x)
    xq = np.concatenate([x[:-1] + 0.25 * h, x[:-1] + 0.75 * h])
    with np.errstate(all="ignore"):
        Dq = deriv(xq)
    sel = np.concatenate([np.arange(n - 1)] * 2)
    with np.errstate(invalid="ignore"):
        low = np.minimum(E[:-1], E[1:]) <= np.nanmin(E) + TABLE_E_CAP  # per interval
    good = (sel >= lo) & (sel < hi) & np.isfinite(Dq) & low[sel]
    if not good.any():
        return 0.0, 0, xq, Dq
    err = np.abs(_hermite_derivative(x, E, D, xq[good]) - Dq[good]) / (np.abs(Dq[good]) + 1.0)
    k = int(np.argmax(err))
    return float(err[k]), int(sel[good][k]), xq, Dq


def _locate_kink(deriv, a, b):
    """Position of a jump in dD/dx (a kink of the force) inside [a, b], by zooming."""
    for _ in range(4):
        xs = np.linspace(a, b, 65)
        with np.errstate(all="ignore"):
            d2 = np.abs(np.diff(deriv(xs), 2))
        k = int(np.nanargmax(d2)) + 1  # second difference is centred on xs[k]
        a, b = xs[max(k - 1, 0)], xs[min(k + 1, 64)]
    return 0.5 * (a + b)


def _table(deriv, energy_at, x0, x1, knots, anchor, what, extend=True):
    """Tabulate E and dE/dx on uniform knots in [x0, x1'] (x1' >= x1).

    dE/dx comes from deriv(xs) (exact, batched).  E is obtained by integrating
    dE/dx with Simpson's rule from the knot nearest to `anchor`, where E is
    evaluated directly.  The resolution is doubled (from knots[0] up to knots[1]
    knots) until the derivative of the cubic Hermite interpolant agrees with
    deriv() at the quarter points of every interval to
    TABLE_TOL * (|dE/dx| + 1 kJ/mol/nm).  If that fails because the force has a
    kink (e.g. where smooth_square_well switches from repulsion to attraction),
    the knot spacing is adjusted so that a knot falls exactly on the kink; the
    table may then extend slightly beyond x1 (only if `extend`).  Where dE/dx is not finite (e.g.
    beyond the range of FENE bonds), the table is extrapolated linearly from the
    finite range, with a warning.
    """
    n, nmax = knots
    x = np.linspace(x0, x1, n)
    with np.errstate(all="ignore"):
        D = deriv(x)
        Dm = deriv(0.5 * (x[1:] + x[:-1]))
    best = None
    prev_err = None
    aligned = not extend
    while True:
        E, lo, hi = _integrate(x, D, Dm, energy_at, anchor, what)
        err, kbad, xq, Dq = _quarter_error(deriv, x, E, D, lo, hi)
        if best is None or err < best[-1]:
            best = (x, D, E, lo, hi, err)
        if err <= TABLE_TOL or 2 * len(x) - 1 > nmax:
            break
        # cubic Hermite gains ~8x per doubling where the force is smooth
        slow = prev_err is not None and err > 0.3 * prev_err
        if slow and not aligned and lo < kbad < hi - 1:
            # A kink strictly inside the finite range (not a singularity at its
            # edge): put a knot on it and (as closely as possible) on x1, which is
            # a kink too for pair potentials that vanish at the cutoff.  Later
            # doublings keep both on knots.
            aligned = True
            h = x[1] - x[0]
            xk = _locate_kink(deriv, x[kbad], x[kbad + 1])
            if xk - x0 > 4 * h:
                m0 = int(round((xk - x0) / h))
                ratio = (x1 - x0) / (xk - x0)
                ms = np.arange(max(1, m0 // 2), 2 * m0 + 1)
                frac = np.abs(ms * ratio - np.round(ms * ratio))
                good = ms[frac <= 1e-4]
                m = int(good[0]) if len(good) else int(ms[np.argmin(frac)])
                h2 = (xk - x0) / m
                n2 = int(np.ceil((x1 - x0) / h2 - 1e-6)) + 1
                if n2 <= nmax:
                    x = x0 + h2 * np.arange(n2)
                    with np.errstate(all="ignore"):
                        D = deriv(x)
                        Dm = deriv(0.5 * (x[1:] + x[:-1]))
                    prev_err = None
                    continue
        if slow and err > 0.9 * prev_err:
            # No improvement: the error is noise in OpenMM's own derivative (e.g.
            # of the angle near theta = pi).  Keep the best table.
            break
        prev_err = err
        # refine: midpoints become knots, quarter points become midpoints
        n = len(x)
        xn = np.empty(2 * n - 1)
        Dn = np.empty(2 * n - 1)
        xn[0::2], xn[1::2] = x, 0.5 * (x[1:] + x[:-1])
        Dn[0::2], Dn[1::2] = D, Dm
        order = np.argsort(xq)
        x, D, Dm = xn, Dn, Dq[order]
    x, D, E, lo, hi, err = best
    if err > TABLE_WARN:
        warnings.warn(f"{what}: tabulated force has a relative error of up to {err:.1e}")
    if lo > 0 or hi < len(x) - 1:
        D = D.copy()
        E[:lo] = E[lo] + D[lo] * (x[:lo] - x[lo])
        D[:lo] = D[lo]
        E[hi + 1 :] = E[hi] + D[hi] * (x[hi + 1 :] - x[hi])
        D[hi + 1 :] = D[hi]
        warnings.warn(
            f"{what}: the force is not finite for x < {x[lo]:.4g} or x > {x[hi]:.4g}; "
            "the native platform extrapolates linearly there"
        )
    return x, E, D


def _force_xml(force):
    return ET.fromstring(openmm.XmlSerializer.serialize(force))


def _param_tuple(elem, n):
    return tuple(float(elem.get(f"param{k + 1}")) for k in range(n))


def _set_global_defaults(root, values):
    gp = root.find("GlobalParameters")
    if gp is not None:
        for p in gp:
            if p.get("name") in values:
                p.set("default", repr(float(values[p.get("name")])))


def _pair_table(root, params_a, params_b, cutoff, what):
    """E(r), dE/dr on [0, cutoff] of a CustomNonbondedForce between two particles."""
    b = _Batch(root, "pair", (params_a, params_b), spacing=3 * cutoff)
    lo, hi = 1e-6 * cutoff, cutoff * (1 - 1e-12)  # no direction at r = 0; stay inside the cutoff

    def deriv(r):
        return np.where(r < cutoff, b.derivative(np.clip(r, lo, hi)), 0.0)

    def energy(r):
        return b.energy(min(max(r, lo), hi)) if r < cutoff else 0.0

    return _table(deriv, energy, 0.0, cutoff, PAIR_KNOTS, cutoff, what)


ANGLE_COS_SPLIT = 0.25  # rad: cos(theta) tables stop this far from a non-flat end


def _angle_table(root, params, what):
    """Tables of a CustomAngleForce for one parameter set.

    Returns (E_theta, D_theta, cos_table) where the first two tabulate E and
    dE/dtheta on [0, pi], and cos_table is None or (c_lo, c_hi, E, dE/dcos).
    The engine uses the cos table (no atan2 per angle and step) for
    cos(theta) in [c_lo, c_hi] and the theta table elsewhere.  dE/dcos =
    -dE/dtheta / sin(theta) is finite at theta = pi (or 0) only if the potential
    is flat there; otherwise the cos table stops ANGLE_COS_SPLIT short of it.
    """
    b = _Batch(root, "angle", params, spacing=10.0)
    eps = 1e-5  # OpenMM's angle derivative is inaccurate within ~1e-6 rad of 0 and pi

    def deriv_theta(th):
        return b.derivative(np.clip(th, eps, np.pi - eps))

    _, Et, Dt = _table(deriv_theta, b.energy, 0.0, np.pi, ANGLE_KNOTS, 0.5 * np.pi, what, extend=False)
    if not (np.all(np.isfinite(Et)) and np.all(np.isfinite(Dt))):
        raise ValueError(f"{what}: energy is not finite for all angles")
    d_end = deriv_theta(np.array([1e-4, np.pi - 1e-4]))
    flat = np.abs(d_end) <= 1e-3 * max(np.abs(Dt).max(), 1e-300)  # at theta = 0, pi
    c_hi = 1.0 if flat[0] else np.cos(ANGLE_COS_SPLIT)
    c_lo = -1.0 if flat[1] else -np.cos(ANGLE_COS_SPLIT)

    def deriv_cos_inner(c):  # dE/dcos = -dE/dtheta / sin(theta), for |c| < 1
        td = np.arccos(c)
        return b.derivative(td) / -np.sin(td)

    def deriv_cos(c):
        """At c = +-1 the limit is taken by quadratic extrapolation from c -+ 1e-3 k
        (k = 1, 2, 3), which ignores sub-1e-3 features such as the one caused by
        the truncated pi in forces.grosberg_angle."""
        c = np.asarray(c, dtype=float)
        out = np.empty_like(c)
        inner = np.abs(c) < 1 - 1e-12
        if inner.any():
            out[inner] = deriv_cos_inner(c[inner])
        for end in (-1.0, 1.0):
            at_end = ~inner & (np.sign(c) == end)
            if at_end.any():
                d = deriv_cos_inner(end - end * 1e-3 * np.arange(1, 4))
                out[at_end] = 3 * d[0] - 3 * d[1] + d[2]
        return out

    def energy_cos(c):
        return b.energy(np.arccos(np.clip(c, -1, 1)))

    _, Ec, Dc = _table(deriv_cos, energy_cos, c_lo, c_hi, ANGLE_KNOTS, 0.5 * (c_lo + c_hi), what, extend=False)
    return Et, Dt, (c_lo, c_hi, Ec, Dc)


def _bond_table(root, params, rmax, what):
    b = _Batch(root, "bond", params, spacing=2 * rmax + 10)

    def deriv(r):
        return b.derivative(np.maximum(r, 1e-7))

    # anchor: the force-free point (usually the rest length) if inside the range
    r = np.linspace(0, rmax, 257)
    with np.errstate(all="ignore"):
        d = np.abs(deriv(r))
    anchor = r[np.nanargmin(np.where(np.isfinite(d), d, np.nan))] if np.isfinite(d).any() else 0.5 * rmax
    return _table(deriv, lambda x: b.energy(max(x, 1e-7)), 0.0, rmax, BOND_KNOTS, anchor, what)


# ---------------------------------------------------------------------------
# External forces recognised by expression
# ---------------------------------------------------------------------------

_SPHERICAL = (
    "step(invert_sign*(r-aa)) * kb * (sqrt((r-aa)*(r-aa) + t*t) - t); "
    "r = sqrt((x-x0)^2 + (y-y0)^2 + (z-z0)^2 + tt^2)"
)
_TETHER = "kx * (x - x0)^2 + ky * (y - y0)^2 + kz * (z - z0)^2"
_TETHER_PBC = (
    "kx * periodicdistance(x, 0, 0, x0, 0, 0)^2 + ky * periodicdistance(0, y, 0, 0, y0, 0)^2 "
    "+ kz * periodicdistance(0, 0, z, 0, 0, z0)^2"
)


def _strip_prefix(energy, name):
    return re.sub(r"(?<!\w)" + re.escape(name) + "_", "", energy) if name else energy


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


class NativeContext:
    """Stand-in for an OpenMM Context backed by the native engine.

    Parameters
    ----------
    system : openmm.System
        Provides particles, masses and the periodic box.
    integrator : NativeIntegrator
    forces : list of OpenMM force objects
        The force objects as held by the caller (``Simulation.force_dict``),
        so that ``force.updateParametersInContext(context)`` can be redirected.
    pbc : bool
    length_scale : float
        Used to choose the range of bond tables.
    """

    def __init__(self, system, integrator, forces, pbc, length_scale=1.0):
        if not isinstance(integrator, NativeIntegrator):
            raise TypeError("the native platform needs a NativeIntegrator")
        if system.getNumForces() != len(forces):
            raise NotImplementedError("forces added directly to Simulation.system are not supported by the native platform")
        self._system = system
        self._integrator = integrator
        self._forces = list(forces)
        self._pbc = bool(pbc)
        self._length_scale = float(length_scale)
        self._N = system.getNumParticles()
        self._params = {}  # current values of global parameters, by name
        self._handlers = {}  # id(force) -> update function
        self._engine = _native_engine.NativeEngine(self._N)
        nthreads = _thread_count()
        if nthreads > 0:
            self._engine.set_threads(nthreads)
        masses = np.array([_val(system.getParticleMass(i), u.dalton) for i in range(self._N)])
        self._engine.set_masses(masses)
        if self._pbc:
            a, b, c = system.getDefaultPeriodicBoxVectors()
            self._box = np.array([_val(a, nm)[0], _val(b, nm)[1], _val(c, nm)[2]])
            self._engine.set_box(True, *self._box)
        else:
            self._box = np.zeros(3)
            self._engine.set_box(False)
        for f in self._forces:
            self._add_force(f)
        integrator._bind(self._engine)

    # -- force translation ---------------------------------------------------
    def _globals(self, f):
        names = []
        if hasattr(f, "getNumGlobalParameters"):
            for i in range(f.getNumGlobalParameters()):
                name = f.getGlobalParameterName(i)
                names.append(name)
                if name not in self._params:
                    self._params[name] = float(f.getGlobalParameterDefaultValue(i))
        return names

    def _add_force(self, f):
        name = getattr(f, "name", type(f).__name__)
        if f.getForceGroup() != 0:
            warnings.warn(f"force {name}: force groups are ignored by the native platform")
        if isinstance(f, openmm.HarmonicBondForce):
            handler = self._add_harmonic_bonds(f)
        elif isinstance(f, openmm.CustomBondForce):
            handler = self._add_custom_bonds(f, name)
        elif isinstance(f, openmm.CustomAngleForce):
            handler = self._add_custom_angles(f, name)
        elif isinstance(f, openmm.CustomNonbondedForce):
            handler = self._add_custom_nonbonded(f, name)
        elif isinstance(f, openmm.CustomExternalForce):
            handler = self._add_external(f, name)
        else:
            raise NotImplementedError(f"force {name} ({type(f).__name__}) is not supported by the native platform")
        self._handlers[id(f)] = handler
        self._redirect_update(f, handler)

    def _redirect_update(self, f, handler):
        cls_update = type(f).updateParametersInContext
        native_ctx = self

        def updateParametersInContext(context):
            if context is native_ctx:
                handler()
            else:
                cls_update(f, context)

        f.updateParametersInContext = updateParametersInContext

    def _add_harmonic_bonds(self, f):
        s = self._engine.add_bond_set(False, f.usesPeriodicBoundaryConditions())

        def update():
            root = _force_xml(f)
            bonds = root.find("Bonds")
            n = len(bonds)
            i = np.fromiter((int(b.get("p1")) for b in bonds), np.intc, n)
            j = np.fromiter((int(b.get("p2")) for b in bonds), np.intc, n)
            d = np.fromiter((float(b.get("d")) for b in bonds), np.float64, n)
            k = np.fromiter((float(b.get("k")) for b in bonds), np.float64, n)
            self._engine.set_harmonic_bonds(s, i, j, d, k)

        update()
        return update

    def _add_custom_bonds(self, f, name):
        if f.getNumBonds() == 0:
            return lambda: None
        s = self._engine.add_bond_set(True, f.usesPeriodicBoundaryConditions())
        globals_ = self._globals(f)
        npar = f.getNumPerBondParameters()
        rmax = 8.0 * self._length_scale

        def update():
            root = _force_xml(f)
            _set_global_defaults(root, {g: self._params[g] for g in globals_})
            bonds = root.find("Bonds")
            keys = [_param_tuple(b, npar) for b in bonds]
            uniq = sorted(set(keys))
            if len(uniq) > MAX_TABLES:
                raise NotImplementedError(f"force {name}: more than {MAX_TABLES} distinct bond parameter sets")
            self._engine.clear_bond_tables(s)
            for key in uniq:
                x, E, D = _bond_table(root, key, rmax, f"bond force {name}")
                self._engine.add_bond_table(s, x[0], x[-1], E, D)
            index = {key: t for t, key in enumerate(uniq)}
            i = [int(b.get("p1")) for b in bonds]
            j = [int(b.get("p2")) for b in bonds]
            self._engine.set_tabulated_bonds(s, i, j, [index[k] for k in keys])

        update()
        return update

    def _add_custom_angles(self, f, name):
        if f.getNumAngles() == 0:
            return lambda: None
        s = self._engine.add_angle_set(f.usesPeriodicBoundaryConditions())
        globals_ = self._globals(f)
        npar = f.getNumPerAngleParameters()

        def update():
            root = _force_xml(f)
            _set_global_defaults(root, {g: self._params[g] for g in globals_})
            angles = root.find("Angles")
            keys = [_param_tuple(a, npar) for a in angles]
            uniq = sorted(set(keys))
            if len(uniq) > MAX_TABLES:
                raise NotImplementedError(f"force {name}: more than {MAX_TABLES} distinct angle parameter sets")
            self._engine.clear_angle_tables(s)
            for key in uniq:
                Et, Dt, (c_lo, c_hi, Ec, Dc) = _angle_table(root, key, f"angle force {name}")
                self._engine.add_angle_table(s, Et, Dt, c_lo, c_hi, Ec, Dc)
            index = {key: t for t, key in enumerate(uniq)}
            self._engine.set_angles(
                s,
                [int(a.get("p1")) for a in angles],
                [int(a.get("p2")) for a in angles],
                [int(a.get("p3")) for a in angles],
                [index[k] for k in keys],
            )

        update()
        return update

    def _add_custom_nonbonded(self, f, name):
        if f.getNumInteractionGroups() > 0:
            raise NotImplementedError(f"force {name}: interaction groups are not supported by the native platform")
        if f.getNonbondedMethod() == f.NoCutoff:
            raise NotImplementedError(f"force {name}: the native platform needs a cutoff")
        if f.getNumComputedValues() if hasattr(f, "getNumComputedValues") else 0:
            raise NotImplementedError(f"force {name}: computed values are not supported")
        cutoff = float(_val(f.getCutoffDistance(), nm))
        if self._pbc and cutoff >= 0.5 * self._box.min():
            raise ValueError(f"force {name}: cutoff {cutoff} must be below half the periodic box")
        globals_ = self._globals(f)
        npar = f.getNumPerParticleParameters()
        state = {}

        def build(first):
            root = _force_xml(f)
            _set_global_defaults(root, {g: self._params[g] for g in globals_})
            particles = root.find("Particles")
            keys = [_param_tuple(p, npar) for p in particles]
            uniq = sorted(set(keys))
            if len(uniq) > MAX_TABLES:
                raise NotImplementedError(
                    f"force {name}: more than {MAX_TABLES} distinct per-particle parameter sets"
                )
            index = {key: t for t, key in enumerate(uniq)}
            types = np.array([index[k] for k in keys], dtype=np.intc)
            if first:
                state["s"] = self._engine.add_pair_force(cutoff, default_skin(cutoff), len(uniq), types)
            elif len(uniq) != state["ntypes"] or not np.array_equal(types, state["types"]):
                raise NotImplementedError(
                    f"force {name}: changing per-particle parameters after context creation is not supported "
                    "by the native platform; call Simulation.reinitialize()"
                )
            s = state["s"]
            state["ntypes"], state["types"] = len(uniq), types
            for a in range(len(uniq)):
                for b in range(a, len(uniq)):
                    x, E, D = _pair_table(root, uniq[a], uniq[b], cutoff, f"nonbonded force {name}")
                    self._engine.set_pair_table(s, a, b, x[0], x[-1], E, D)
                    if a != b:
                        self._engine.set_pair_table(s, b, a, x[0], x[-1], E, D)
            excl = root.find("Exclusions")
            ei = [int(e.get("p1")) for e in excl] if excl is not None else []
            ej = [int(e.get("p2")) for e in excl] if excl is not None else []
            self._engine.set_pair_exclusions(s, ei, ej)

        build(True)
        return lambda: build(False)

    def _add_external(self, f, name):
        energy = _strip_prefix(f.getEnergyFunction(), name)
        globals_ = self._globals(f)
        n = f.getNumParticles()
        particles = np.empty(n, dtype=np.intc)
        params = np.empty((n, f.getNumPerParticleParameters()))
        for k in range(n):
            p, par = f.getParticleParameters(k)
            particles[k] = p
            params[k] = par

        def g(key):
            return self._params[f"{name}_{key}" if f"{name}_{key}" in self._params else key]

        if energy == _SPHERICAL:
            s = self._engine.add_spherical_confinement(particles, 0, 0, 0, 0, 1, 0, 0, 0)

            def update():
                self._engine.set_spherical_confinement(
                    s, g("kb"), g("aa"), g("t"), g("tt"), g("invert_sign"), g("x0"), g("y0"), g("z0")
                )

        elif energy in (_TETHER, _TETHER_PBC):
            pbc = energy == _TETHER_PBC
            s = self._engine.add_tether(particles, params[:, :3], 0, 0, 0, pbc)

            def update():
                self._engine.set_tether_k(s, g("kx"), g("ky"), g("kz"))

        else:
            raise NotImplementedError(
                f"force {name}: CustomExternalForce with energy {f.getEnergyFunction()!r} is not supported "
                "by the native platform (supported: spherical_confinement, tether_particles)"
            )
        update()
        return update

    # -- OpenMM Context API ------------------------------------------------------
    def getSystem(self):
        return self._system

    def getIntegrator(self):
        return self._integrator

    def getPlatform(self):
        return NativePlatform()

    def setPositions(self, positions):
        x = _array(positions, nm)
        if x.shape != (self._N, 3):
            raise ValueError(f"expected positions of shape ({self._N}, 3), got {x.shape}")
        self._engine.set_positions(x)

    def setVelocities(self, velocities):
        v = _array(velocities, nm / ps)
        if v.shape != (self._N, 3):
            raise ValueError(f"expected velocities of shape ({self._N}, 3), got {v.shape}")
        self._engine.set_velocities(v)

    def setVelocitiesToTemperature(self, temperature, randomSeed=None):
        seed = randomSeed if randomSeed is not None else int.from_bytes(os.urandom(8), "little")
        self._engine.set_velocities_to_temperature(float(_val(temperature, u.kelvin)), seed)

    def getTime(self):
        return self._engine.get_time() * ps

    def setTime(self, t):
        self._engine.set_time(float(_val(t, ps)))

    def getStepCount(self):
        return self._engine.get_step_count()

    def getParameter(self, name):
        return self._params[name]

    def getParameters(self):
        return dict(self._params)

    def setParameter(self, name, value):
        if name not in self._params:
            raise KeyError(f"unknown global parameter {name!r}")
        self._params[name] = float(_val(value))
        for f in self._forces:
            if hasattr(f, "getNumGlobalParameters") and any(
                f.getGlobalParameterName(i) == name for i in range(f.getNumGlobalParameters())
            ):
                self._handlers[id(f)]()

    def getState(
        self,
        getPositions=False,
        getVelocities=False,
        getForces=False,
        getEnergy=False,
        getParameters=False,
        enforcePeriodicBox=False,
        groups=-1,
        **kwargs,
    ):
        e = self._engine
        pos = vel = forces = ekin = epot = None
        if getPositions:
            pos = e.get_positions()
            if enforcePeriodicBox and self._pbc:
                pos = pos - np.floor(pos / self._box) * self._box
        if getVelocities:
            vel = e.get_velocities()
        if getForces:
            forces = e.get_forces()
        if getEnergy:
            epot = e.potential_energy()
            ekin = e.kinetic_energy()
        return NativeState(e.get_time(), e.get_step_count(), pos, vel, forces, ekin, epot, self._box)

    def reinitialize(self, preserveState=False):
        st = self.getState(getPositions=True, getVelocities=True) if preserveState else None
        t = self._engine.get_time()
        self.__init__(self._system, self._integrator, self._forces, self._pbc, self._length_scale)
        if st is not None:
            self._engine.set_positions(st._pos)
            self._engine.set_velocities(st._vel)
            self._engine.set_time(t)

    # -- extras ------------------------------------------------------------------
    def minimize(self, tolerance=10.0, maxIterations=0):
        """Local energy minimisation with OpenMM's L-BFGS on its CPU platform."""
        integ = openmm.VerletIntegrator(0.001)
        ctx = openmm.Context(self._system, integ, openmm.Platform.getPlatformByName("CPU"))
        ctx.setPositions(self._engine.get_positions() * nm)
        for name, value in self._params.items():
            ctx.setParameter(name, value)
        openmm.LocalEnergyMinimizer.minimize(ctx, tolerance, maxIterations)
        self._engine.set_positions(ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nm))
        del ctx

    @property
    def engine(self):
        return self._engine


def default_skin(cutoff):
    """Verlet-list skin.  Polymer beads move ~1-2% of a bond per step, so lists
    are rebuilt every few steps; a large skin keeps rebuilds rare."""
    return 0.75 * cutoff
