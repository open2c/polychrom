import simtk.openmm as openmm
import simtk.unit as units

nm = units.meter * 1e-9
fs = units.second * 1e-15
ps = units.second * 1e-12

import numpy as np 

def _initHarmonicBondForce(sim_object):
    "Internal, inits harmonic forse for polymer and non-polymer bonds"
    if "HarmonicBondForce" not in list(sim_object.forceDict.keys()):
        sim_object.forceDict["HarmonicBondForce"] = openmm.HarmonicBondForce()
    sim_object.bondType = "Harmonic"


def _initAbsBondForce(sim_object):
    "inits abs(x) FENE bond force"
    if "AbsBondForce" not in list(sim_object.forceDict.keys()):
        force = "(1. / ABSwiggle) * ABSunivK * "\
        "(sqrt((r-ABSr0 * ABSconlen)* "\
        " (r - ABSr0 * ABSconlen) + ABSa * ABSa) - ABSa)"

        bondforceAbs = openmm.CustomBondForce(force)
        bondforceAbs.addPerBondParameter("ABSwiggle")
        bondforceAbs.addPerBondParameter("ABSr0")
        bondforceAbs.addGlobalParameter("ABSunivK", sim_object.kT / sim_object.conlen)
        bondforceAbs.addGlobalParameter("ABSa", 0.02 * sim_object.conlen)
        bondforceAbs.addGlobalParameter("ABSconlen", sim_object.conlen)
        sim_object.forceDict["AbsBondForce"] = bondforceAbs



def addBond(sim_object,
            i, j,  # particles connected by bond
            bondWiggleDistance=0.2,
            # Flexibility of the bond,
            # measured in distance at which energy equals kT
            distance=None,  # Equilibrium length of the bond, default = sim_object.length_scale
            bondType=None,  # Harmonic, Grosberg, ABS
            verbose=None):  # Set this to False or True to override sim_object.verbose for this function
            # and don't want to contaminate output by 10000 messages
    """Adds bond between two particles, allows to specify parameters

    Parameters
    ----------

    i,j : int
        Particle numbers

    bondWiggleDistance : float
        Average displacement from the equilibrium bond distance

    bondType : "Harmonic" or "abs"
        Type of bond. Distance and bondWiggleDistance can be
        specified for harmonic bonds only

    verbose : bool
        Set this to False if you're in verbose mode and don't want to
        print "bond added" message

    """

    if not hasattr(sim_object, "bondLengths"):
        sim_object.bondLengths = []

    if verbose is None:
        verbose = sim_object.verbose
    if (i >= sim_object.N) or (j >= sim_object.N):
        raise ValueError("\nCannot add bond with monomers %d,%d that"\
        "are beyound the polymer length %d" % (i, j, sim_object.N))

    bondSize = float(bondWiggleDistance)

    if distance is None:
        distance = sim_object.length_scale
    else:
        distance = sim_object.length_scale * distance
    distance = float(distance)

    if not hasattr(sim_object, "kbondScalingFactor"):  # caching kbondScalingFactor - performance improvement... 
        t = (2 * sim_object.kT / (sim_object.conlen) ** 2) / (units.kilojoule_per_mole / nm ** 2)
        print(t)
        sim_object.kbondScalingFactor = float((2 * sim_object.kT / (sim_object.conlen) ** 2) / (units.kilojoule_per_mole / nm ** 2))
    
    kbondScalingFactor = sim_object.kbondScalingFactor  #... not to calculate it eevry time we add bond 
    # this will be an integer, so we don't have to deal with slow simtk.units every time we add a bond 

    if bondType is None:
        bondType = sim_object.bondType

    if bondType.lower() == "harmonic":
        _initHarmonicBondForce(sim_object)
        kbond = kbondScalingFactor / (bondSize ** 2)  # using kbondScalingFactor because force accepts parameters with units
        sim_object.forceDict["HarmonicBondForce"].addBond(int(i), int(j), float(distance), float(kbond))
        sim_object.bondLengths.append([int(i), int(j), float(distance), float(bondSize)])
    elif bondType.lower() == "abs":
        _initAbsBondForce(sim_object)
        sim_object.forceDict["AbsBondForce"].addBond(int(i), int(
            j), [float(bondWiggleDistance), float(distance)])  # force is initialized to accept floats already
        sim_object.bondLengths.append([int(i), int(j), float(distance), float(bondSize)])
    else:
        sim_object._exitProgram("Bond type not known")
    if verbose == True:
        print("%s bond added between %d,%d, wiggle %lf dist %lf" % (
            bondType, i, j, float(bondWiggleDistance), float(distance)))



def harmonicPolymerBonds(sim_object,
                            wiggleDist=0.05,
                            bondLength=1.0,
                            exceptBonds=True):
    """Adds harmonic bonds connecting polymer chains

    Parameters
    ----------

    wiggleDist : float
        Average displacement from the equilibrium bond distance
    bondLength : float
        The length of the bond
    exceptBonds : bool
        If True then do not calculate non-bonded forces between the
        particles connected by a bond. True by default.
    """


    for start, end, isRing in sim_object.chains:
        for j in range(start, end - 1):
            addBond(sim_object, j, j + 1, wiggleDist,
                distance=bondLength,
                bondType="Harmonic", verbose=False)
            if exceptBonds:
                sim_object.bondsForException.append((j, j + 1))

        if isRing:
            addBond(sim_object, start, end - 1, wiggleDist,
                distance=bondLength, bondType="Harmonic")
            if exceptBonds:
                sim_object.bondsForException.append((start, end - 1))
            if sim_object.verbose == True:
                print("ring bond added", start, end - 1)


def angleForce(sim_object, k=1.5):
    """Adds harmonic angle bonds. k specifies energy in kT at one radian
    If k is an array, it has to be of the length N.
    Xth value then specifies stiffness of the angle centered at
    monomer number X.
    Values for ends of the chain will be simply ignored.

    Parameters
    ----------

    k : float or list of length N
        Stiffness of the bond.
        If list, then determines stiffness of the bond at monomer i.
        Potential is k * alpha^2 * 0.5 * kT
    """
    try:
        k[0]
    except:
        k = np.zeros(sim_object.N, float) + k
    stiffForce = openmm.CustomAngleForce(
        "kT*angK * (theta - 3.141592) * (theta - 3.141592) * (0.5)")
    sim_object.forceDict["AngleForce"] = stiffForce
    for start, end, isRing in sim_object.chains:
        for j in range(start + 1, end - 1):
            stiffForce.addAngle(j - 1, j, j + 1, [float(k[j])])
        if isRing:
            stiffForce.addAngle(int(end - 2),int( end - 1), int(start), [float(k[end - 1])  ])
            stiffForce.addAngle(int(end - 1),int( start), int(start + 1), [float(k[start])  ])

    stiffForce.addGlobalParameter("kT", sim_object.kT)
    stiffForce.addPerAngleParameter("angK")



def polynomialRepulsiveForce(sim_object, trunc=3.0, radiusMult=1.):
    """This is a simple polynomial repulsive potential. It has the value
    of `trunc` at zero, stays flat until 0.6-0.7 and then drops to zero
    together with its first derivative at r=1.0.

    Parameters
    ----------

    trunc : float
        the energy value around r=0

    """
    radius = sim_object.conlen * radiusMult
    nbCutOffDist = radius
    repul_energy = (
        "rsc12 * (rsc2 - 1.0) * REPe / REPemin + REPe;"
        "rsc12 = rsc4 * rsc4 * rsc4;"
        "rsc4 = rsc2 * rsc2;"
        "rsc2 = rsc * rsc;"
        "rsc = r / REPsigma * REPrmin;")
    sim_object.forceDict["polynomialRepulsiveForce"] = openmm.CustomNonbondedForce(repul_energy)
    repulforceGr = sim_object.forceDict["polynomialRepulsiveForce"]

    repulforceGr.addGlobalParameter('REPe', trunc * sim_object.kT)
    repulforceGr.addGlobalParameter('REPsigma', radius)
    # Coefficients for x^8*(x*x-1)
    # repulforceGr.addGlobalParameter('REPemin', 256.0 / 3125.0)
    # repulforceGr.addGlobalParameter('REPrmin', 2.0 / np.sqrt(5.0))
    # Coefficients for x^12*(x*x-1)
    repulforceGr.addGlobalParameter('REPemin', 46656.0 / 823543.0)
    repulforceGr.addGlobalParameter('REPrmin', np.sqrt(6.0 / 7.0))
    for _ in range(sim_object.N):
        repulforceGr.addParticle(())

    repulforceGr.setCutoffDistance(nbCutOffDist)

def smoothSquareWellForce(sim_object,
    repulsionEnergy=3.0, repulsionRadius=1.,
    attractionEnergy=0.5, attractionRadius=2.0,
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
    attractionEnergy: float
        the maximal range of the attractive part of the potential.

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
    sim_object.forceDict["Nonbonded"] = openmm.CustomNonbondedForce(
        energy)
    repulforceGr = sim_object.forceDict["Nonbonded"]

    repulforceGr.addGlobalParameter('REPe', repulsionEnergy * sim_object.kT)
    repulforceGr.addGlobalParameter('REPsigma', repulsionRadius * sim_object.conlen)

    repulforceGr.addGlobalParameter('ATTRe', attractionEnergy * sim_object.kT)
    repulforceGr.addGlobalParameter('ATTRdelta',
        sim_object.conlen * (attractionRadius - repulsionRadius) / 2.0)
    # Coefficients for the minimum of x^12*(x*x-1)
    repulforceGr.addGlobalParameter('emin12', 46656.0 / 823543.0)
    repulforceGr.addGlobalParameter('rmin12', np.sqrt(6.0 / 7.0))

    for _ in range(sim_object.N):
        repulforceGr.addParticle(())

    repulforceGr.setCutoffDistance(nbCutOffDist)

def selectiveSSWForce(sim_object,
    stickyParticlesIdxs,
    extraHardParticlesIdxs,
    repulsionEnergy=3.0,
    repulsionRadius=1.,
    attractionEnergy=3.0,
    attractionRadius=1.5,
    selectiveRepulsionEnergy=20.0,
    selectiveAttractionEnergy=1.0):
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
    b) You can select a subset of particles and make them "extra hard".

    This force was used two-ways. First was to make a small subset of particles very sticky. 
    In that case, it is advantageous to make the sticky particles and their neighbours
    "extra hard" and thus prevent the system from collapsing.

    Another useage is to induce phase separation by making all B monomers sticky. In that case, 
    extraHard particles may not be needed at all, because the system would not collapse on itsim_object. 


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
        the EXTRA repulsion energy applied to the "extra hard" particles
    selectiveAttractionEnergy: float
        the EXTRA attraction energy applied to the "sticky" particles
    """

    energy = (
        "step(REPsigma - r) * Erep + step(r - REPsigma) * Eattr;"
        ""
        "Erep = rsc12 * (rsc2 - 1.0) * REPeTot / emin12 + REPeTot;"  # + ESlide;"
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

    if selectiveRepulsionEnergy == float('inf'):
        energy += (
        "REPeAdd = 4 * ((REPsigma / (2.0^(1.0/6.0)) / r)^12 - (REPsigma / (2.0^(1.0/6.0)) / r)^6) + 1;"
        )

    repulforceGr = openmm.CustomNonbondedForce(energy)

    repulforceGr.setCutoffDistance(attractionRadius * sim_object.conlen)

    repulforceGr.addGlobalParameter('REPe', repulsionEnergy * sim_object.kT)
    if selectiveRepulsionEnergy != float('inf'):
        repulforceGr.addGlobalParameter('REPeAdd', selectiveRepulsionEnergy * sim_object.kT)
    repulforceGr.addGlobalParameter('REPsigma', repulsionRadius * sim_object.conlen)

    repulforceGr.addGlobalParameter('ATTRe', attractionEnergy * sim_object.kT)
    repulforceGr.addGlobalParameter('ATTReAdd', selectiveAttractionEnergy * sim_object.kT)
    repulforceGr.addGlobalParameter('ATTRdelta',
        sim_object.conlen * (attractionRadius - repulsionRadius) / 2.0)

    # Coefficients for x^12*(x*x-1)
    repulforceGr.addGlobalParameter('emin12', 46656.0 / 823543.0)
    repulforceGr.addGlobalParameter('rmin12', np.sqrt(6.0 / 7.0))

    repulforceGr.addPerParticleParameter("Sticky")
    repulforceGr.addPerParticleParameter("ExtraHard")
    counts = np.bincount(stickyParticlesIdxs, minlength=sim_object.N)

    for i in range(sim_object.N):
        repulforceGr.addParticle(
            (float(counts[i]),
             float(i in extraHardParticlesIdxs)))

    sim_object.forceDict["Nonbonded"] = repulforceGr



def cylindricalConfinement(sim_object, r, bottom=None, k=0.1, top=9999):
    "As it says."

    if bottom == True:
        warnings.warn(DeprecationWarning(
            "Use bottom=0 instead of bottom = True! "))
        bottom = 0

    if bottom is not None:
        extforce2 = openmm.CustomExternalForce(
            "step(r-CYLaa) * CYLkb * (sqrt((r-CYLaa)*(r-CYLaa) + CYLt*CYLt) - CYLt)"
            "+ step(-z + CYLbot) * CYLkb * (sqrt((z - CYLbot)^2 + CYLt^2) - CYLt) "
            "+ step(z - CYLtop) * CYLkb * (sqrt((z - CYLtop)^2 + CYLt^2) - CYLt);"
            "r = sqrt(x^2 + y^2 + CYLtt^2)")
    else:
        extforce2 = openmm.CustomExternalForce(
            "step(r-CYLaa) * CYLkb * (sqrt((r-CYLaa)*(r-CYLaa) + CYLt*CYLt) - CYLt);"
            "r = sqrt(x^2 + y^2 + CYLtt^2)")

    sim_object.forceDict["CylindricalConfinement"] = extforce2
    for i in range(sim_object.N):
        extforce2.addParticle(i, [])
    extforce2.addGlobalParameter("CYLkb", k * sim_object.kT / nm)
    extforce2.addGlobalParameter("CYLtop", top * sim_object.conlen)
    if bottom is not None:
        extforce2.addGlobalParameter("CYLbot", bottom * sim_object.conlen)
    extforce2.addGlobalParameter("CYLkt", sim_object.kT)
    extforce2.addGlobalParameter("CYLweired", nm)
    extforce2.addGlobalParameter("CYLaa", (r - 1. / k) * nm)
    extforce2.addGlobalParameter("CYLt", (1. / (10 * k)) * nm)
    extforce2.addGlobalParameter("CYLtt", 0.01 * nm)

def sphericalConfinement(sim_object,
            r="density",  # radius... by default uses certain density
            k=5.,  # How steep the walls are
            density=.3):  # target density, measured in particles
                            # per cubic nanometer (bond size is 1 nm)
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
    """

    spherForce = openmm.CustomExternalForce(
        "step(r-SPHaa) * SPHkb * (sqrt((r-SPHaa)*(r-SPHaa) + SPHt*SPHt) - SPHt) "
        ";r = sqrt(x^2 + y^2 + z^2 + SPHtt^2)")
    sim_object.forceDict["SphericalConfinement"] = spherForce

    for i in range(sim_object.N):
        spherForce.addParticle(i, [])
    if r == "density":
        r = (3 * sim_object.N / (4 * 3.141592 * density)) ** (1 / 3.)

    sim_object.sphericalConfinementRadius = r
    if sim_object.verbose == True:
        print("Spherical confinement with radius = %lf" % r)
    # assigning parameters of the force
    spherForce.addGlobalParameter("SPHkb", k * sim_object.kT / nm)
    spherForce.addGlobalParameter("SPHaa", (r - 1. / k) * nm)
    spherForce.addGlobalParameter("SPHt", (1. / k) * nm / 10.)
    spherForce.addGlobalParameter("SPHtt", 0.01 * nm)
    return r




def tetherParticles(sim_object, particles, k=30, positions="current"):
    """tethers particles in the 'particles' array.
    Increase k to tether them stronger, but watch the system!

    Parameters
    ----------

    particles : list of ints
        List of particles to be tethered (fixed in space)
    k : int, optional
        Steepness of the tethering potential.
        Values >30 will require decreasing potential,
        but will make tethering rock solid.
    """
    if "Tethering Force" not in sim_object.forceDict:
        tetherForce = openmm.CustomExternalForce(
          " TETHkb * ((x - TETHx0)^2 + (y - TETHy0)^2 + (z - TETHz0)^2)")
        sim_object.forceDict["Tethering Force"] = tetherForce
    else:
        tetherForce = sim_object.forceDict["Tethering Force"]

    # assigning parameters of the force
    tetherForce.addGlobalParameter("TETHkb", k * sim_object.kT / nm)
    tetherForce.addPerParticleParameter("TETHx0")
    tetherForce.addPerParticleParameter("TETHy0")
    tetherForce.addPerParticleParameter("TETHz0")
    if positions == "current":
        positions = [sim_object.data[i] for i in particles]
    else:
        positions = sim_object.addUnits(positions)

    for i, pos in zip(particles, positions):  # adding all the particles on which force acts
        i = int(i)
        tetherForce.addParticle(i, list(pos))
        if sim_object.verbose == True:
            print("particle %d tethered! " % i)

def pullForce(sim_object, particles, forces):
    """
    adds force pulling on each particle
    particles: list of particle indices
    forces: list of forces [[f0x,f0y,f0z],[f1x,f1y,f1z], ...]
    if there are fewer forces than particles forces are padded with forces[-1]
    """
    import itertools
    pullForce = openmm.CustomExternalForce(
        "PULLx * x + PULLy * y + PULLz * z")
    pullForce.addPerParticleParameter("PULLx")
    pullForce.addPerParticleParameter("PULLy")
    pullForce.addPerParticleParameter("PULLz")
    for num, force in itertools.zip_longest(particles, forces, fillvalue=forces[-1]):
        force = [float(i) * (sim_object.kT / sim_object.conlen) for i in force]
        pullForce.addParticle(num, force)
    sim_object.forceDict["PullForce"] = pullForce
