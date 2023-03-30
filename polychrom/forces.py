"""

Detailed description of forces in polychrom
-------------------------------------------

This module defines forces commonly used in polychrom. Most forces are implemented using custom forces in openmm. The
force equations were generally derived such that the force and the first derivative both go to zero at the cutoff
radius.

Parametrization of bond forces
******************************

Most of the bond forces are parametrized using two parametrs: bondLength and bondWiggleDistance. The parameter
*bondLength* is length of the bond at rest, while *bondWiggleDistance* is the estension of the bond at which energy
reaches 1kT.

Note that the actual standard deviation of the bond length is bondWiggleDistance/sqrt(2) for a harmonic bond force,
and is bondWiggleDistance*sqrt(2) for constant force bonds, so if you are switching from harmonic bonds to constant
force, you may choose to decrease the wiggleDistance by a factor of 2.




Note on energy equations
************************

Energy  equations are passed as strings to one of the OpenMM customXXXForce class (e.g. customNonbondedForce). Note
two things. First, sub-equations are separated by semicolon, and are evaluated "bottom up", last equation first.
Second, equations seem much more scary than they actually are (see below).

All energy equations have to be continuous, and we strongly believe that the first derivative has to be continuous as
well. As a result, all equations were carefully crafted to be smooth functions. This makes things more complicated.
For example, a simple "k * abs(x-x0)" becomes "k * (sqrt((x-x0)^2 + a^2) - a)" where a is a small number (defined to
be 0.01 for example).

All energy equations have to be calculatable in single precision. Any rounding error will throw you off. For example,
you should never have sqrt(A - B) where A and B are expressions, and A >= B. Because by chance, due to rounding,
you may and up with A slightly less than B, and you will receive NaN, and the whole simulation will blow up.
Similarly, atan(very_large_number), while defined mathematically, could easily become NaN, because very_large_number
may be larger than the largest allowable float.

Note that basically all nonbonded forces were written before OpenMM introduced a switching function
http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomNonbondedForce.html Therefore,
we always manually sticth the value and the first derivative of the force to be 0 at the cutoff distance. For custom
user-defined forces, it may be better to use switching function instead. This does not apply to custom
external forces, there stitching is still necessary.

Force equations don't have "if" statements, but it is possible to avoid them where they would be normally used. For
example,  "if a: b= b0 + c" can be replaced with  "b = b0 + c * delta(a)". Similarly "f(r) if r < r0; 0 otherwise" is
just "f(r) * step(r0 - r)". These examples appear frequently in the forces that we have. One of the finest examples
of crafting complex forces with on-the-fly generation of force equation is in
:py:func:`polychrom.forces.heteropolymer_SSW`. One of the best examples of optimizing complex forces using
polynomials is in :py:func:`polychrom.forces.polynomial_repulsive`.

"""

import itertools
import re
import warnings
from collections.abc import Iterable
import hoomd
import numpy as np

try:
    import openmm
except Exception:
    import simtk.openmm as openmm

import simtk.unit


def _prepend_force_name_to_params(force):
    """
    This function is called by :py:mod:`polychrom.simulation.Simulation.add_force` method. It's goal is to avoid
    using the same names of global parameters defined in different forces. To this end, it modifies names of
    parameters of each force to start with the force name, which should be unique.
    """
    if not hasattr(force, "getEnergyFunction"):
        return

    energy = force.getEnergyFunction()
    if hasattr(force, "getNumGlobalParameters"):
        for i in range(force.getNumGlobalParameters()):
            old_name = force.getGlobalParameterName(i)
            new_name = force.name + "_" + old_name
            force.setGlobalParameterName(i, new_name)
            energy = re.sub(r"(?<!\w)" + f"{old_name}" + r"(?!\w)", new_name, energy)

    force.setEnergyFunction(energy)


def _check_bonds(bonds, N):
    # check for repeating bond
    if len(set(bonds)) != len(bonds):
        for bond in set(bonds):
            bonds.remove(bond)

        raise ValueError(f"Bonds {bonds} are repeated. Set override_checks=True to override this check.")

    # check that all monomers make at least one bond
    monomer_not_in_bond = ~np.zeros(N).astype(bool)
    bonds_arr = np.array(bonds)
    monomer_not_in_bond[bonds_arr.reshape(-1)] = False
    if monomer_not_in_bond.any():
        raise ValueError(
            f"Monomers {np.where(monomer_not_in_bond)[0]} are not in any bonds."
            "Set override_checks=True to override this check."
        )

    # check that no bonds of the form (i, i) exist
    if (bonds_arr[:, 0] == bonds_arr[:, 1]).any():
        index = np.where(bonds_arr[:, 0] == bonds_arr[:, 1])[0]
        raise ValueError(
            f"Bonds {bonds_arr[index].tolist()} are self-bonds. Set override_checks=True to"
            " override this check."
        )


def _check_angle_bonds(triplets):
    # check that triplets are unique
    if len(set(triplets)) != len(triplets):
        for triplet in set(triplets):
            triplets.remove(triplet)

        raise ValueError(f"Triplets {triplets} are repeated. Set override_checks=True to override this check.")

    # check that no triplet of the form (i, i, j) exists
    # check that no bonds of the form (i, i) exist
    triplet_arr = np.array(triplets)
    err_condition = (
        (triplet_arr[:, 0] == triplet_arr[:, 1])
        | (triplet_arr[:, 0] == triplet_arr[:, 2])
        | (triplet_arr[:, 1] == triplet_arr[:, 2])
    )
    if err_condition.any():
        index = np.where(err_condition)[0]
        raise ValueError(
            f"Triplets {triplet_arr[index].tolist()} contain monomers with the same index. "
            "Set override_checks=True to override this check."
        )


def _to_array_1d(scalar_or_array, arrlen, dtype=float):
    """
    A helper function for writing forces that can accept either a single parameter,
    or an array of per-particle parameters.
    If passed a scalar, it converts it to an array of the length arrlen.
    If passed an iterable, it verifies that its length equals to arrlen, and
    returns a numpy array.
    """
    if not hasattr(scalar_or_array, "__iter__"):
        outarr = np.full(arrlen, scalar_or_array, dtype)
    else:
        outarr = np.asarray(scalar_or_array, dtype=dtype)

    if len(outarr) != arrlen:
        raise ValueError("The length of the array differs from the expected one!")

    return outarr


def harmonic_bonds(
    sim_object,
    bonds,
    bondWiggleDistance=0.05,
    bondLength=1.0,
    name="harmonic_bonds",
    override_checks=False,
):
    """Adds harmonic bonds.

    Bonds are parametrized in the following way.

    * A length of a bond at rest is `bondLength`
    * Bond energy equal to 1kT at bondWiggleDistance

    Note that bondWiggleDistance is not the standard deviation of the bond extension:
    that is actually smaller by a factor of sqrt(2).


    Parameters
    ----------

    bonds : iterable of (int, int)
        Pairs of particle indices to be connected with a bond.
    bondWiggleDistance : float or iterable of float
        Distance at which bond energy equals kT.
        Can be provided per-particle.
        If 0 then set k=0.
    bondLength : float or iterable of float
        The length of the bond.
        Can be provided per-particle.
    override_checks: bool
        If True then do not check that no bonds are repeated.
        False by default.
    """

    # check for repeated bonds
    if not override_checks:
        _check_bonds(bonds, sim_object.N)
    if sim_object.backend == "openmm":
        force = openmm.HarmonicBondForce()
    else:
        force = hoomd.md.bond.Harmonic()

        try:
            ntypes = len(sim_object.system.bonds.types)
            sim_object.system.bonds.types.append(name)
            sim_object.system.bonds.N += len(bonds)
            sim_object.system.bonds.typeid += [ntypes] * len(bonds)

        except TypeError or AttributeError:
            ntypes = 0
            sim_object.system.bonds.types = [name]
            sim_object.system.bonds.N += len(bonds)
            sim_object.system.bonds.typeid = [ntypes] * len(bonds)
            sim_object.system.bonds.group = []

    force.name = name

    bondLength = _to_array_1d(bondLength, len(bonds)) * sim_object.length_scale
    bondWiggleDistance = _to_array_1d(bondWiggleDistance, len(bonds)) * sim_object.length_scale

    # using kbondScalingFactor because force accepts parameters with units
    kbond = sim_object.kbondScalingFactor / (bondWiggleDistance**2)
    kbond[bondWiggleDistance == 0] = 0

    if sim_object.backend == "hoomd":
        force.params[name] = dict(k=kbond[0], r0=bondLength[0])
    for bond_idx, (i, j) in enumerate(bonds):
        if (i >= sim_object.N) or (j >= sim_object.N):
            raise ValueError(
                "\nCannot add bond with monomers %d,%d that" "are beyound the polymer length %d" % (i, j, sim_object.N)
            )

        force.addBond(int(i), int(j), float(bondLength[bond_idx]), float(kbond[bond_idx]))

    return force


def constant_force_bonds(
    sim_object,
    bonds,
    bondWiggleDistance=0.05,
    bondLength=1.0,
    quadraticPart=0.02,
    name="abs_bonds",
    override_checks=False,
):
    """

    Constant force bond force. Energy is roughly linear with estension
    after r=quadraticPart; before it is quadratic to make sure the force
    is differentiable.

    Force is parametrized using the same approach as bond force:
    it reaches U=kT at extension = bondWiggleDistance

    Note that, just as with bondForce, mean squared extension
    is actually larger than wiggleDistance by sqrt(2) factor.

    Parameters
    ----------

    bonds : iterable of (int, int)
        Pairs of particle indices to be connected with a bond.
    bondWiggleDistance : float
        Displacement at which bond energy equals 1 kT.
        Can be provided per-particle.
    bondLength : float
        The length of the bond.
        Can be provided per-particle.
    override_checks: bool
        If True then do not check that no bonds are repeated.
        False by default.
    """

    # check for repeated bonds
    if not override_checks:
        _check_bonds(bonds, sim_object.N)

    energy = "(1. / wiggle) * univK * (sqrt((r-r0 * conlen) * (r - r0 * conlen) + a * a) - a)"
    force = openmm.CustomBondForce(energy)
    force.name = name

    force.addPerBondParameter("wiggle")
    force.addPerBondParameter("r0")
    force.addGlobalParameter("univK", sim_object.kT / sim_object.conlen)
    force.addGlobalParameter("a", quadraticPart * sim_object.conlen)
    force.addGlobalParameter("conlen", sim_object.conlen)

    bondLength = _to_array_1d(bondLength, len(bonds)) * sim_object.length_scale
    bondWiggleDistance = _to_array_1d(bondWiggleDistance, len(bonds)) * sim_object.length_scale

    for bond_idx, (i, j) in enumerate(bonds):
        if (i >= sim_object.N) or (j >= sim_object.N):
            raise ValueError(
                "\nCannot add bond with monomers %d,%d that" "are beyound the polymer length %d" % (i, j, sim_object.N)
            )

        force.addBond(
            int(i),
            int(j),
            [float(bondWiggleDistance[bond_idx]), float(bondLength[bond_idx])],
        )

    return force


def angle_force(sim_object, triplets, k=1.5, theta_0=np.pi, name="angle", override_checks=False):
    """Adds harmonic angle bonds. k specifies energy in kT at one radian
    If k is an array, it has to be of the length N.
    Xth value then specifies stiffness of the angle centered at
    monomer number X.
    Values for ends of the chain will be simply ignored.

    Parameters
    ----------

    k : float or list of length N
        Stiffness of the bond.
        If list, then determines the stiffness of the i-th triplet
        Potential is k * alpha^2 * 0.5 * kT

    theta_0 : float or list of length N
              Equilibrium angle of the bond. By default it is np.pi.

    override_checks: bool
        If True then do not check that no bonds are repeated.
        False by default.
    """

    # check for repeated triplets
    if not override_checks:
        _check_angle_bonds(triplets)

    k = _to_array_1d(k, len(triplets))
    theta_0 = _to_array_1d(theta_0, len(triplets))

    energy = "kT*angK * (theta - angT0) * (theta - angT0) * (0.5)"
    force = openmm.CustomAngleForce(energy)
    force.name = name

    force.addGlobalParameter("kT", sim_object.kT)
    force.addPerAngleParameter("angK")
    force.addPerAngleParameter("angT0")

    for triplet_idx, (p1, p2, p3) in enumerate(triplets):
        force.addAngle(
            int(p1),
            int(p2),
            int(p3),
            (float(k[triplet_idx]), float(theta_0[triplet_idx])),
        )

    return force

    # check for repeated bonds
    if not override_checks:
        _check_bonds(bonds, sim_object.N)
    if sim_object.backend == "openmm":
        force = openmm.HarmonicBondForce()
    else:
        force = hoomd.md.bond.Harmonic()

def polynomial_repulsive(sim_object, trunc=3.0, radiusMult=1.0, name="polynomial_repulsive"):
    """This is a simple polynomial repulsive potential. It has the value
    of `trunc` at zero, stays flat until 0.6-0.7 and then drops to zero
    together with its first derivative at r=1.0.

    See the gist below with an example of the potential.
    https://gist.github.com/mimakaev/0327bf6ffe7057ee0e0625092ec8e318

    Parameters
    ----------

    trunc : float
        the energy value around r=0

    """
    radius = sim_object.conlen * radiusMult
    nbCutOffDist = radius
    if sim_object.backend == "openmm":
        repul_energy = (
            "rsc12 * (rsc2 - 1.0) * REPe / emin12 + REPe;"
            "rsc12 = rsc4 * rsc4 * rsc4;"
            "rsc4 = rsc2 * rsc2;"
            "rsc2 = rsc * rsc;"
            "rsc = r / REPsigma * rmin12;"
        )

        force = openmm.CustomNonbondedForce(repul_energy)

        force.addGlobalParameter("REPe", trunc * sim_object.kT)
        force.addGlobalParameter("REPsigma", radius)
        # Coefficients for x^8*(x*x-1)
        # force.addGlobalParameter('emin12', 256.0 / 3125.0)
        # force.addGlobalParameter('rmin12', 2.0 / np.sqrt(5.0))
        # Coefficients for x^12*(x*x-1)
        force.addGlobalParameter("emin12", 46656.0 / 823543.0)
        force.addGlobalParameter("rmin12", np.sqrt(6.0 / 7.0))
        for _ in range(sim_object.N):
            force.addParticle(())
        force.setCutoffDistance(nbCutOffDist)
        force.name = name
    else:
        typecomps = itertools.combinations(sim_object.system.particles.types, 2)
        if sim_object.kwargs["integrator"].lower() == "dpd":
            for i, j in typecomps:
                sim_object.dpd.params[(i, j)] = dict(
                    A=trunc * sim_object.kT.value_in_unit(simtk.unit.kilojoule_per_mole),
                    gamma=sim_object.kwargs["mass"] * sim_object.kwargs["collision_rate"],
                )
            force = sim_object.dpd
            force.name = None
        else:
            sim_object.nl = hoomd.md.nlist.Cell(0.4)
            sim_object.dpd = hoomd.md.pair.DPDConservative(
                default_r_cut=1.0,
                nlist=sim_object.nl,
            )
            sim_object.dpd.params.default = dict(
                A=trunc * sim_object.kT.value_in_unit(simtk.unit.kilojoule_per_mole),
            )
            force = sim_object.dpd
            force.name = name

    return force


def smooth_square_well(
    sim_object,
    repulsionEnergy=3.0,
    repulsionRadius=1.0,
    attractionEnergy=0.5,
    attractionRadius=2.0,
    name="smooth_square_well",
):
    """
    This is a simple and fast polynomial force that looks like a smoothed
    version of the square-well potential. The energy equals `repulsionEnergy`
    around r=0, stays flat until 0.6-0.7, then drops to zero together
    with its first derivative at r=1.0. After that it drop down to
    `attractionEnergy` and gets back to zero at r=`attractionRadius`.

    The energy function is based on polynomials of 12th power. Both the
    function and its first derivative is continuous everywhere within its
    domain and they both get to zero at the boundary.

    Parameters
    ----------

    repulsionEnergy: float
        the heigth of the repulsive part of the potential.
        E(0) = `repulsionEnergy`
    repulsionRadius: float
        the radius of the repulsive part of the potential.
        E(`repulsionRadius`) = 0,
        E'(`repulsionRadius`) = 0
    attractionEnergy: float
        the depth of the attractive part of the potential.
        E(`repulsionRadius`/2 + `attractionRadius`/2) = `attractionEnergy`
    attractionRadius: float
        the radius of the attractive part of the potential.
        E(`attractionRadius`) = 0,
        E'(`attractionRadius`) = 0
    """
    nbCutOffDist = sim_object.conlen * attractionRadius
    energy = (
        "step(REPsigma - r) * Erep + step(r - REPsigma) * Eattr;"
        ""
        "Erep = rsc12 * (rsc2 - 1.0) * REPe / emin12 + REPe;"
        "rsc12 = rsc4 * rsc4 * rsc4;"
        "rsc4 = rsc2 * rsc2;"
        "rsc2 = rsc * rsc;"
        "rsc = r / REPsigma * rmin12;"
        ""
        "Eattr = - rshft12 * (rshft2 - 1.0) * ATTRe / emin12 - ATTRe;"
        "rshft12 = rshft4 * rshft4 * rshft4;"
        "rshft4 = rshft2 * rshft2;"
        "rshft2 = rshft * rshft;"
        "rshft = (r - REPsigma - ATTRdelta) / ATTRdelta * rmin12"
    )

    force = openmm.CustomNonbondedForce(energy)
    force.name = name

    force.addGlobalParameter("REPe", repulsionEnergy * sim_object.kT)
    force.addGlobalParameter("REPsigma", repulsionRadius * sim_object.conlen)

    force.addGlobalParameter("ATTRe", attractionEnergy * sim_object.kT)
    force.addGlobalParameter("ATTRdelta", sim_object.conlen * (attractionRadius - repulsionRadius) / 2.0)
    # Coefficients for the minimum of x^12*(x*x-1)
    force.addGlobalParameter("emin12", 46656.0 / 823543.0)
    force.addGlobalParameter("rmin12", np.sqrt(6.0 / 7.0))

    for _ in range(sim_object.N):
        force.addParticle(())

    force.setCutoffDistance(nbCutOffDist)

    return force


def selective_SSW(
    sim_object,
    stickyParticlesIdxs,
    extraHardParticlesIdxs,
    repulsionEnergy=3.0,  # base repulsion energy for **all** particles
    repulsionRadius=1.0,
    attractionEnergy=3.0,  # base attraction energy for **all** particles
    attractionRadius=1.5,
    selectiveRepulsionEnergy=20.0,  # **extra** repulsive energy for **extraHard** particles
    selectiveAttractionEnergy=1.0,  # **extra** attractive energy for **sticky** particles
    name="selective_SSW",
):
    """
    This is a simple and fast polynomial force that looks like a smoothed
    version of the square-well potential. The energy equals `repulsionEnergy`
    around r=0, stays flat until 0.6-0.7, then drops to zero together
    with its first derivative at r=1.0. After that it drop down to
    `attractionEnergy` and gets back to zero at r=`attractionRadius`.

    The energy function is based on polynomials of 12th power. Both the
    function and its first derivative is continuous everywhere within its
    domain and they both get to zero at the boundary.

    This is a tunable version of SSW:
    a) You can specify the set of "sticky" particles. The sticky particles
    are attracted only to other sticky particles.
    b) You can **smultaneously** select a subset of particles and make them "extra hard".


    This force was used two-ways. First was to make a small subset of particles very sticky.
    In that case, it is advantageous to make the sticky particles and their neighbours
    "extra hard" and thus prevent the system from collapsing.

    Another useage is to induce phase separation by making all B monomers sticky. In that case,
    extraHard particles may not be needed at all, because the system would not collapse on iteslf.

    Parameters
    ----------

    stickyParticlesIdxs: list of int
        the list of indices of the "sticky" particles. The sticky particles
        are attracted to each other with extra `selectiveAttractionEnergy`
    extraHardParticlesIdxs : list of int
        the list of indices of the "extra hard" particles. The extra hard
        particles repel all other particles with extra
        `selectiveRepulsionEnergy`
    repulsionEnergy: float
        the heigth of the repulsive part of the potential.
        E(0) = `repulsionEnergy`
    repulsionRadius: float
        the radius of the repulsive part of the potential.
        E(`repulsionRadius`) = 0,
        E'(`repulsionRadius`) = 0
    attractionEnergy: float
        the depth of the attractive part of the potential.
        E(`repulsionRadius`/2 + `attractionRadius`/2) = `attractionEnergy`
    attractionRadius: float
        the maximal range of the attractive part of the potential.
    selectiveRepulsionEnergy: float
        the **EXTRA** repulsion energy applied to the **extra hard** particles
    selectiveAttractionEnergy: float
        the **EXTRA** attraction energy applied to the **sticky** particles
    """

    energy = (  # + ESlide;"
        "step(REPsigma - r) * Erep + step(r - REPsigma) * Eattr;"
        ""
        "Erep = rsc12 * (rsc2 - 1.0) * REPeTot / emin12 + REPeTot;"
        "REPeTot = REPe + (ExtraHard1 + ExtraHard2) * REPeAdd;"
        "rsc12 = rsc4 * rsc4 * rsc4;"
        "rsc4 = rsc2 * rsc2;"
        "rsc2 = rsc * rsc;"
        "rsc = r / REPsigma * rmin12;"
        ""
        "Eattr = - rshft12 * (rshft2 - 1.0) * ATTReTot / emin12 - ATTReTot;"
        "ATTReTot = ATTRe + min(Sticky1, Sticky2) * ATTReAdd;"
        "rshft12 = rshft4 * rshft4 * rshft4;"
        "rshft4 = rshft2 * rshft2;"
        "rshft2 = rshft * rshft;"
        "rshft = (r - REPsigma - ATTRdelta) / ATTRdelta * rmin12;"
        ""
    )

    if selectiveRepulsionEnergy == float("inf"):
        energy += (
            "REPeAdd = 4 * ((REPsigma / (2.0^(1.0/6.0)) / r)^12 - (REPsigma / (2.0^(1.0/6.0)) /"
            " r)^6) + 1;"
        )

    force = openmm.CustomNonbondedForce(energy)
    force.name = name

    force.setCutoffDistance(attractionRadius * sim_object.conlen)

    force.addGlobalParameter("REPe", repulsionEnergy * sim_object.kT)
    if selectiveRepulsionEnergy != float("inf"):
        force.addGlobalParameter("REPeAdd", selectiveRepulsionEnergy * sim_object.kT)
    force.addGlobalParameter("REPsigma", repulsionRadius * sim_object.conlen)

    force.addGlobalParameter("ATTRe", attractionEnergy * sim_object.kT)
    force.addGlobalParameter("ATTReAdd", selectiveAttractionEnergy * sim_object.kT)
    force.addGlobalParameter("ATTRdelta", sim_object.conlen * (attractionRadius - repulsionRadius) / 2.0)

    # Coefficients for x^12*(x*x-1)
    force.addGlobalParameter("emin12", 46656.0 / 823543.0)
    force.addGlobalParameter("rmin12", np.sqrt(6.0 / 7.0))

    force.addPerParticleParameter("Sticky")
    force.addPerParticleParameter("ExtraHard")

    counts = np.bincount(stickyParticlesIdxs, minlength=sim_object.N)

    for i in range(sim_object.N):
        force.addParticle((float(counts[i]), float(i in extraHardParticlesIdxs)))

    return force


def heteropolymer_SSW(
    sim_object,
    interactionMatrix,
    monomerTypes,
    extraHardParticlesIdxs,
    repulsionEnergy=3.0,  # base repulsion energy for **all** particles
    repulsionRadius=1.0,
    attractionEnergy=3.0,  # base attraction energy for **all** particles
    attractionRadius=1.5,
    selectiveRepulsionEnergy=20.0,  # **extra** repulsive energy for **extraHard** particles
    selectiveAttractionEnergy=1.0,  # **extra** attraction energy that is multiplied by interactionMatrix
    keepVanishingInteractions=False,
    name="heteropolymer_SSW",
):
    """
    A version of smooth square well potential that enables the simulation of
    heteropolymers. Every monomer is assigned a number determining its type,
    then one can specify additional attraction between the types with the
    interactionMatrix. Repulsion between all monomers is the same, except for
    extraHardParticles, which, if specified, have higher repulsion energy.

    The overall potential is the same as in :py:func:`polychrom.forces.smooth_square_well`

    Treatment of extraHard particles is the same as in :py:func:`polychrom.forces.selective_SSW`

    This is an extension of SSW (smooth square well) force in which:

    a) You can give monomerTypes (e.g. 0, 1, 2 for A, B, C)
       and interaction strengths between these types. The corresponding entry in
       interactionMatrix is multiplied by selectiveAttractionEnergy to give the actual
       **additional** depth of the potential well.
    b) You can select a subset of particles and make them "extra hard". See selective_SSW force for descrition.

    Force summary
    *************

    Potential is the same as smooth square well, with the following parameters for particles i and j:

    * Attraction energy (i,j) = attractionEnergy + selectiveAttractionEnergy * interactionMatrix[i,j]

    * Repulsion Energy (i,j) = repulsionEnergy + selectiveRepulsionEnergy;  if (i) or (j) are extraHard
    * Repulsion Energy (i,j) = repulsionEnergy;  otherwise

    Parameters
    ----------

    interactionMatrix: np.array
        the **EXTRA** interaction strenghts between the different types.
        Only upper triangular values are used. See "Force summary" above
    monomerTypes: list of int or np.array
        the type of each monomer, starting at 0
    extraHardParticlesIdxs : list of int
        the list of indices of the "extra hard" particles. The extra hard
        particles repel all other particles with extra
        `selectiveRepulsionEnergy`
    repulsionEnergy: float
        the heigth of the repulsive part of the potential.
        E(0) = `repulsionEnergy`
    repulsionRadius: float
        the radius of the repulsive part of the potential.
        E(`repulsionRadius`) = 0,
        E'(`repulsionRadius`) = 0
    attractionEnergy: float
        the depth of the attractive part of the potential.
        E(`repulsionRadius`/2 + `attractionRadius`/2) = `attractionEnergy`
    attractionRadius: float
        the maximal range of the attractive part of the potential.
    selectiveRepulsionEnergy: float
        the **EXTRA** repulsion energy applied to the "extra hard" particles
    selectiveAttractionEnergy: float
        the **EXTRA** attraction energy (prefactor for the interactionMatrix interactions)
    keepVanishingInteractions : bool
        a flag that determines whether the terms that have zero interaction are
        still added to the force. This can be useful when changing the force
        dynamically (i.e. switching interactions on at some point)
    """

    # Check type info for consistency
    Ntypes = max(monomerTypes) + 1  # IDs should be zero based
    if any(np.less(interactionMatrix.shape, [Ntypes, Ntypes])):
        raise ValueError("Need interactions for {0:d} types!".format(Ntypes))
    if not np.allclose(interactionMatrix.T, interactionMatrix):
        raise ValueError("Interaction matrix should be symmetric!")

    indexpairs = []
    for i in range(0, Ntypes):
        for j in range(0, Ntypes):
            if (not interactionMatrix[i, j] == 0) or keepVanishingInteractions:
                indexpairs.append((i, j))

    energy = (  # + ESlide;"
        "step(REPsigma - r) * Erep + step(r - REPsigma) * Eattr;"
        ""
        "Erep = rsc12 * (rsc2 - 1.0) * REPeTot / emin12 + REPeTot;"
        "REPeTot = REPe + (ExtraHard1 + ExtraHard2) * REPeAdd;"
        "rsc12 = rsc4 * rsc4 * rsc4;"
        "rsc4 = rsc2 * rsc2;"
        "rsc2 = rsc * rsc;"
        "rsc = r / REPsigma * rmin12;"
        ""
        "Eattr = - rshft12 * (rshft2 - 1.0) * ATTReTot / emin12 - ATTReTot;"
        "ATTReTot = ATTRe"
    )
    if len(indexpairs) > 0:
        energy += (" + ATTReAdd*(delta(type1-{0:d})*delta(type2-{1:d})" "*INT_{0:d}_{1:d}").format(
            indexpairs[0][0], indexpairs[0][1]
        )
        for i, j in indexpairs[1:]:
            energy += "+delta(type1-{0:d})*delta(type2-{1:d})*INT_{0:d}_{1:d}".format(i, j)
        energy += ")"
    energy += (
        ";"
        "rshft12 = rshft4 * rshft4 * rshft4;"
        "rshft4 = rshft2 * rshft2;"
        "rshft2 = rshft * rshft;"
        "rshft = (r - REPsigma - ATTRdelta) / ATTRdelta * rmin12;"
        ""
    )

    if selectiveRepulsionEnergy == float("inf"):
        energy += (
            "REPeAdd = 4 * ((REPsigma / (2.0^(1.0/6.0)) / r)^12 - (REPsigma / (2.0^(1.0/6.0)) /"
            " r)^6) + 1;"
        )

    force = openmm.CustomNonbondedForce(energy)
    force.name = name

    force.setCutoffDistance(attractionRadius * sim_object.conlen)

    force.addGlobalParameter("REPe", repulsionEnergy * sim_object.kT)
    if selectiveRepulsionEnergy != float("inf"):
        force.addGlobalParameter("REPeAdd", selectiveRepulsionEnergy * sim_object.kT)
    force.addGlobalParameter("REPsigma", repulsionRadius * sim_object.conlen)

    force.addGlobalParameter("ATTRe", attractionEnergy * sim_object.kT)
    force.addGlobalParameter("ATTReAdd", selectiveAttractionEnergy * sim_object.kT)
    force.addGlobalParameter("ATTRdelta", sim_object.conlen * (attractionRadius - repulsionRadius) / 2.0)

    # Coefficients for x^12*(x*x-1)
    force.addGlobalParameter("emin12", 46656.0 / 823543.0)
    force.addGlobalParameter("rmin12", np.sqrt(6.0 / 7.0))

    for i, j in indexpairs:
        force.addGlobalParameter("INT_{0:d}_{1:d}".format(i, j), interactionMatrix[i, j])

    force.addPerParticleParameter("type")
    force.addPerParticleParameter("ExtraHard")

    for i in range(sim_object.N):
        force.addParticle((float(monomerTypes[i]), float(i in extraHardParticlesIdxs)))

    return force


def cylindrical_confinement(sim_object, r, bottom=None, k=0.1, top=9999, name="cylindrical_confinement"):
    """As it says."""

    if bottom:
        warnings.warn(DeprecationWarning("Use bottom=0 instead of bottom = True! "))
        bottom = 0

    if bottom is not None:
        force = openmm.CustomExternalForce(
            "kt * k * ("
            " step(dr) * (sqrt(dr*dr + t*t) - t)"
            " + step(-z + bottom) * (sqrt((z - bottom)^2 + t^2) - t) "
            " + step(z - top) * (sqrt((z - top)^2 + t^2) - t)"
            ") ;"
            "dr = sqrt(x^2 + y^2 + tt^2) - r + 10*t"
        )
    else:
        force = openmm.CustomExternalForce(
            "kt * k * step(dr) * (sqrt(dr*dr + t*t) - t);" "dr = sqrt(x^2 + y^2 + tt^2) - r + 10*t"
        )
    force.name = name

    for i in range(sim_object.N):
        force.addParticle(i, [])

    force.addGlobalParameter("k", k / simtk.unit.nanometer)
    force.addGlobalParameter("r", r * sim_object.conlen)
    force.addGlobalParameter("kt", sim_object.kT)
    force.addGlobalParameter("t", 0.1 / k * simtk.unit.nanometer)
    force.addGlobalParameter("tt", 0.01 * simtk.unit.nanometer)
    force.addGlobalParameter("top", top * sim_object.conlen)
    if bottom is not None:
        force.addGlobalParameter("bottom", bottom * sim_object.conlen)

    return force


def spherical_confinement(
    sim_object,
    r="density",  # radius... by default uses certain density
    k=5.0,  # How steep the walls are
    density=0.3,  # target density, measured in particles
    # per cubic nanometer (bond size is 1 nm)
    center=[0, 0, 0],
    invert=False,
    particles=None,
    name="spherical_confinement",
):
    """Constrain particles to be within a sphere.
    With no parameters creates sphere with density .3

    Parameters
    ----------
    r : float or "density", optional
        Radius of confining sphere. If "density" requires density,
        or assumes density = .3
    k : float, optional
        Steepness of the confining potential, in kT/nm
    density : float, optional, <1
        Density for autodetection of confining radius.
        Density is calculated in particles per nm^3,
        i.e. at density 1 each sphere has a 1x1x1 cube.
    center : [float, float, float]
        The coordinates of the center of the sphere.
    invert : bool
        If True, particles are not confinded, but *excluded* from the sphere.
    particles : list of int
        The list of particles affected by the force.
        If None, apply the force to all particles.
    """

    force = openmm.CustomExternalForce(
        "step(invert_sign*(r-aa)) * kb * (sqrt((r-aa)*(r-aa) + t*t) - t); "
        "r = sqrt((x-x0)^2 + (y-y0)^2 + (z-z0)^2 + tt^2)"
    )
    force.name = name

    particles = range(sim_object.N) if particles is None else particles
    for i in particles:
        force.addParticle(int(i), [])

    if r == "density":
        r = (3 * sim_object.N / (4 * 3.141592 * density)) ** (1 / 3.0)

    if sim_object.verbose:
        print("Spherical confinement with radius = %lf" % r)
    # assigning parameters of the force
    force.addGlobalParameter("kb", k * sim_object.kT / simtk.unit.nanometer)
    force.addGlobalParameter("aa", (r - 1.0 / k) * simtk.unit.nanometer)
    force.addGlobalParameter("t", (1.0 / k) * simtk.unit.nanometer / 10.0)
    force.addGlobalParameter("tt", 0.01 * simtk.unit.nanometer)
    force.addGlobalParameter("invert_sign", (-1) if invert else 1)

    force.addGlobalParameter("x0", center[0] * simtk.unit.nanometer)
    force.addGlobalParameter("y0", center[1] * simtk.unit.nanometer)
    force.addGlobalParameter("z0", center[2] * simtk.unit.nanometer)

    # TODO: move 'r' elsewhere?..
    sim_object.sphericalConfinementRadius = r

    return force


def spherical_well(sim_object, particles, r, center=[0, 0, 0], width=1, depth=1, name="spherical_well"):
    """
    A spherical potential well, suited for example to simulate attraction to a lamina.

    Parameters
    ----------

    particles : list of int or np.array
        indices of particles that are attracted
    r : float
        Radius of the nucleus
    center : vector, optional
        center position of the sphere. This parameter is useful when confining
        chromosomes to their territory.
    width : float, optional
        Width of attractive well, nm.
    depth : float, optional
        Depth of attractive potential in kT
        NOTE: switched sign from openmm-polymer, because it was confusing. Now
        this parameter is really the depth of the well, i.e. positive =
        attractive, negative = repulsive
    """

    force = openmm.CustomExternalForce(
        "step(1+d) * step(1-d) * SPHWELLdepth * (1 + cos(3.1415926536*d)) / 2;"
        "d = (sqrt((x-SPHWELLx)^2 + (y-SPHWELLy)^2 + (z-SPHWELLz)^2) - SPHWELLradius) / SPHWELLwidth"
    )

    force.name = name

    force.addGlobalParameter("SPHWELLradius", r * sim_object.conlen)
    force.addGlobalParameter("SPHWELLwidth", width * sim_object.conlen)
    force.addGlobalParameter("SPHWELLdepth", depth * sim_object.kT)
    force.addGlobalParameter("SPHWELLx", center[0] * sim_object.conlen)
    force.addGlobalParameter("SPHWELLy", center[1] * sim_object.conlen)
    force.addGlobalParameter("SPHWELLz", center[2] * sim_object.conlen)

    # adding all the particles on which force acts
    for i in particles:
        # NOTE: the explicit type cast seems to be necessary if we have an np.array...
        force.addParticle(int(i), [])

    return force


def tether_particles(
    sim_object, particles, *, pbc=False, k=30, positions="current", name="Tethers"
):
    """tethers particles in the 'particles' array.
    Increase k to tether them stronger, but watch the system!

    Parameters
    ----------

    particles : list of ints
        List of particles to be tethered (fixed in space).
        Negative values are allowed.

    pbc : Bool, optional
        If True, periodicdistance function is applied
    k : int, optional
        The steepness of the tethering potential.
        Values >30 will require decreasing potential, but will make tethering
        rock solid.
        Can be provided as a vector [kx, ky, kz].
    """

    if pbc:
        energy = (
            "kx * periodicdistance(x, 0, 0, x0, 0, 0)^2 + ky * periodicdistance(0, y, 0, 0, y0, 0)^2 "
            "+ kz * periodicdistance(0, 0, z, 0, 0, z0)^2"
        )
    else:
        energy = "kx * (x - x0)^2 + ky * (y - y0)^2 + kz * (z - z0)^2"

    force = openmm.CustomExternalForce(energy)
    force.name = name

    # assigning parameters of the force

    if isinstance(k, Iterable):
        k = list(k)
        if len(k) != 3:
            raise ValueError("k must either be a scalar or a 3D vector!")
        kx, ky, kz = k
    else:
        kx, ky, kz = k, k, k

    nm2 = simtk.unit.nanometer * simtk.unit.nanometer
    force.addGlobalParameter("kx", kx * sim_object.kT / nm2)
    force.addGlobalParameter("ky", ky * sim_object.kT / nm2)
    force.addGlobalParameter("kz", kz * sim_object.kT / nm2)

    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    particles = [sim_object.N + i if i < 0 else i for i in particles]

    if positions == "current":
        positions = [sim_object.data[i] for i in particles]
    else:
        positions = simtk.unit.Quantity(positions, simtk.unit.nanometer)

    # adding all the particles on which force acts
    for i, pos in zip(particles, positions):
        i = int(i)
        force.addParticle(i, list(pos))
        if sim_object.verbose:
            print("particle %d tethered! " % i)

    return force


def pull_force(sim_object, particles, force_vecs, name="Pull"):
    """
    adds force pulling on each particle
    particles: list of particle indices
    force_vecs: list of forces [[f0x,f0y,f0z],[f1x,f1y,f1z], ...]
    if there are fewer forces than particles forces are padded with forces[-1]
    """
    force = openmm.CustomExternalForce("- x * fx - y * fy - z * fz")
    force.name = name

    force.addPerParticleParameter("fx")
    force.addPerParticleParameter("fy")
    force.addPerParticleParameter("fz")

    for num, force_vec in itertools.zip_longest(particles, force_vecs, fillvalue=force_vecs[-1]):
        force_vec = [float(f) * (sim_object.kT / sim_object.conlen) for f in force_vec]
        force.addParticle(int(num), force_vec)

    return force


def grosberg_polymer_bonds(sim_object, bonds, k=30, name="grosberg_polymer", override_checks=False):
    """Adds FENE bonds according to Halverson-Grosberg paper.
    (Halverson, Jonathan D., et al. "Molecular dynamics simulation study of
     nonconcatenated ring polymers in a melt. I. Statics."
     The Journal of chemical physics 134 (2011): 204904.)

    This method has a repulsive potential build-in,
    so that Grosberg bonds could be used with truncated potentials.
    Is of no use unless you really need to simulate Grosberg-type system.

    Parameters
    ----------
    k : float, optional
        Arbitrary parameter; default value as in Grosberg paper.

    override_checks: bool
        If True then do not check that no bonds are repeated.
        False by default.
    """

    # check for repeated bonds
    if not override_checks:
        _check_bonds(bonds, sim_object.N)

    equation = "- 0.5 * k * r0 * r0 * log(1-(r/r0)* (r / r0))"
    force = openmm.CustomBondForce(equation)
    force.name = name

    force.addGlobalParameter("k", k * sim_object.kT / (sim_object.conlen * sim_object.conlen))
    force.addGlobalParameter("r0", sim_object.conlen * 1.5)

    for bond_idx, (i, j) in enumerate(bonds):
        if (i >= sim_object.N) or (j >= sim_object.N):
            raise ValueError(
                "\nCannot add bond with monomers %d,%d that" "are beyound the polymer length %d" % (i, j, sim_object.N)
            )

        force.addBond(int(i), int(j))

    return force


def grosberg_angle(sim_object, triplets, k=1.5, name="grosberg_angle", override_checks=False):
    """
    Adds stiffness according to the Grosberg paper.
    (Halverson, Jonathan D., et al. "Molecular dynamics simulation study of
    nonconcatenated ring polymers in a melt. I. Statics."
    The Journal of chemical physics 134 (2011): 204904.)

    Parameters are synchronized with normal stiffness

    If k is an array, it has to be of the length N.
    Xth value then specifies stiffness of the angle centered at
    monomer number X.
    Values for ends of the chain will be simply ignored.

    Parameters
    ----------

    k : float or N-long list of floats
        Synchronized with regular stiffness.
        Default value is very flexible, as in Grosberg paper.
        Default value maximizes entanglement length.

    override_checks: bool
        If True then do not check that no bonds are repeated.
        False by default.
    """

    # check for repeated triplets
    if not override_checks:
        _check_angle_bonds(triplets)

    k = _to_array_1d(k, len(triplets))

    force = openmm.CustomAngleForce("GRk * kT * (1 - cos(theta - 3.141592))")

    force.name = name
    force.addGlobalParameter("kT", sim_object.kT)
    force.addPerAngleParameter("GRk")

    for triplet_idx, (p1, p2, p3) in enumerate(triplets):
        force.addAngle(p1, p2, p3, [k[triplet_idx]])

    return force


def grosberg_repulsive_force(
    sim_object,
    trunc=None,
    radiusMult=1.0,
    name="grosberg_repulsive",
    trunc_function="min(trunc1, trunc2)",
):
    """This is the fastest non-transparent repulsive force.
    (that preserves topology, doesn't allow chain passing)
    Done according to the paper:
    (Halverson, Jonathan D., et al. "Molecular dynamics simulation study of
     nonconcatenated ring polymers in a melt. I. Statics."
     The Journal of chemical physics 134 (2011): 204904.)
    Parameters
    ----------

    trunc : None, float or N-array of floats
        "transparency" values for each particular particle, which correspond to the truncation
        values in kT for the grosberg repulsion energy between a pair of such particles.
        Value of 1.5 yields frequent passing, 3 - average passing, 5 - rare passing.
    radiusMult : float (optional)
        Multiplier for the size of the force. To make scale the energy larger, set to be more than 1.
    trunc_function : str (optional)
        a formula to calculate the truncation between a pair of particles with transparencies trunc1 and trunc2
        Default is min(trunc1, trunc2)


    """
    radius = sim_object.conlen * radiusMult
    nbCutOffDist = radius * 2.0 ** (1.0 / 6.0)
    if trunc is None:
        repul_energy = "4 * e * ((sigma/r)^12 - (sigma/r)^6) + e"
    else:
        trunc = _to_array_1d(trunc, sim_object.N)
        repul_energy = (
            "step(cut2*trunc_pair - U) * U + step(U - cut2*trunc_pair) * cut2 * trunc_pair * (1 +"
            f" tanh(U/(cut2*trunc_pair) - 1));trunc_pair={trunc_function};U = 4 * e *"
            " ((sigma/r2)^12 - (sigma/r2)^6) + e;r2 = (r^10. + (sigma03)^10.)^0.1"
        )
    force = openmm.CustomNonbondedForce(repul_energy)
    force.name = name

    force.addGlobalParameter("e", sim_object.kT)
    force.addGlobalParameter("sigma", radius)
    force.addGlobalParameter("sigma03", 0.3 * radius)

    if trunc is not None:
        force.addGlobalParameter("cut2", 0.5 * sim_object.kT)
        force.addPerParticleParameter("trunc")

        for i in range(sim_object.N):  # adding all the particles on which force acts
            force.addParticle([float(trunc[i])])
    else:
        for i in range(sim_object.N):  # adding all the particles on which force acts
            force.addParticle(())

    force.setCutoffDistance(nbCutOffDist)

    return force
