import simtk.openmm as openmm
import numpy as np 


"""
This is a collection of old forces that are likely no longer used 
These were a part of openmmlib before April 2019, but were removed during spring cleaning. 

They should be importable and may or may not just work. It should not be difficult to make them compatible 
with the new library. 

"""




def minimizing_repulsive_Force(sim_object):
    """
    Adds a special force which could be use for very efficient resolution of crossings
    Use this force to perform (local) energy minimization if your monomers are all "on top of each other"
    E.g. if you start your simulations with fractional brownyan motion with h < 0.4
    Then switch to a normal force, and re-do energy minimization. 
    """
    radius = sim_object.conlen * 1.3

    nbCutOffDist = radius * 1.
    repul_energy = "1000* REPe * (1-r/REPr)^2 "

    sim_object.force_dict["Nonbonded_minimizing_Force"] = openmm.CustomNonbondedForce(
        repul_energy)
    repulforceGr = sim_object.force_dict["Nonbonded_minimizing_Force"]
    repulforceGr.addGlobalParameter('REPe', sim_object.kT)
    repulforceGr.addGlobalParameter('REPr', sim_object.kT)
    for _ in range(sim_object.N):
        repulforceGr.addParticle(())
    repulforceGr.setCutoffDistance(nbCutOffDist)

    
            

def fix_particles_Z_coordinate(sim_object, particles, zCoordinates, k=0.3,
                            useOtherAxis="z", mode="abs", gap=None):
    """Limits position of a set of particles in z coordinate

    Parameters
    ----------
    particles : list
        List of particles to be fixed.
    zCoordinates : list, or tuple of length 2
        If has length of particles, then should contain all Z coordinates
        If has length 2, then contains z coordinates of first and
        Nth particles, and the rest is approximated linearly.
    k : float, optional
        Strength of attraction, measured in kT/(bondlength)
    useOtherAxis : "x","y" or "z", optional
        Apply the same in the other dimension
    gap: float or None
        if gap is not None, then the force creates a gap of the width "gap" 
        (+- 0.5 * gap) during which the force is not acting. The force starts acting 
        after the particle moved 0.5 * gap. 
    """

    if not len(particles) == len(zCoordinates):
        assert len(zCoordinates) == 2
        start, stop = tuple(zCoordinates)
        zCoordinates = []
        for par in particles:
            zCoordinates.append(start + float(
                stop - start) * (par / float(sim_object.N)))

    if (mode == "abs") and (gap is None):
        zFixForce = openmm.CustomExternalForce(
        "ZFIXk * (sqrt((%s - ZFIXr0)^2 + ZFIXa^2) - ZFIXa)" % (
                                                       useOtherAxis,))
        zFixForce.addGlobalParameter("ZFIXk", k * sim_object.kT / (sim_object.conlen))
    elif (mode == "abs") and (gap is not None):
        zFixForce = openmm.CustomExternalForce(
        "ZFIXk * step(%s - ZFIXr0 - ZFIXgap * 0.5) *"\
        " (sqrt((%s - ZFIXr0 - ZFIXgap * 0.5)^2 + ZFIXa^2) - ZFIXa) + "\
        "ZFIXk * step(-%s + ZFIXr0 - ZFIXgap * 0.5) * "\
        "(sqrt((-%s + ZFIXr0 - ZFIXgap * 0.5)^2 + ZFIXa^2) - ZFIXa)"\
        % (useOtherAxis, useOtherAxis, useOtherAxis, useOtherAxis))

        zFixForce.addGlobalParameter("ZFIXk", k * sim_object.kT / (sim_object.conlen))
        zFixForce.addGlobalParameter("ZFIXgap", sim_object.conlen * gap)

    elif (mode == "quadratic") and (gap is None):
        zFixForce = openmm.CustomExternalForce(
            "ZFIXk * ((%s - ZFIXr0)^2)" % (useOtherAxis,))
        zFixForce.addGlobalParameter("ZFIXk", k * sim_object.kT /
            (sim_object.conlen * sim_object.conlen))
    elif (mode == "quadratic") and (gap is not None):
        zFixForce = openmm.CustomExternalForce(
        "ZFIXk * (step(%s - ZFIXr0 - ZFIXgap * 0.5) * "\
        "(%s - ZFIXr0 - ZFIXgap * 0.5)^2 +  "\
        "step(-%s + ZFIXr0 - ZFIXgap * 0.5) * "\
        "(-%s + ZFIXr0 - ZFIXgap * 0.5)^2)" \
        % (useOtherAxis, useOtherAxis, useOtherAxis, useOtherAxis))

        zFixForce.addGlobalParameter("ZFIXk", k * sim_object.kT /
            (sim_object.conlen * sim_object.conlen))
        zFixForce.addGlobalParameter("ZFIXgap", sim_object.conlen * gap)
    else:
        raise RuntimeError("Not implemented")

    zFixForce.addPerParticleParameter("ZFIXr0")

    zFixForce.addGlobalParameter("ZFIXa", 0.05 * sim_object.conlen)
    for par, zcoor in zip(particles, zCoordinates):
        zFixForce.addParticle(int(par), [float(zcoor)])
    sim_object.force_dict["fixZCoordinates"] = zFixForce




    
def lamina_attraction(sim_object, width=1, depth=1, r=None):
    """Attracts one domain to the lamina. Infers radius
    from spherical confinement, that has to be initialized already.

    Parameters
    ----------

    width : float, optional
        Width of attractive layer next to the lamina, nm.
    depth : float, optional
        Depth of attractive potential in kT
        note- depth < 0 for attractive!  >0 is repulsive
    r : float, optional
        Radius of an attractive cage. If not specified, inferred
        from previously defined spherical potential.
    """

    sim_object.metadata["laminaAttraction"] = repr({"width": width,
        "depth": depth, "r": r})
    laminaForce = openmm.CustomExternalForce(
        "step(LAMr-LAMaa + LAMwidth) * step(LAMaa + LAMwidth - LAMr) "
        "* LAMdepth * (LAMr-LAMaa + LAMwidth) * (LAMaa + LAMwidth - LAMr) "
        "/ (LAMwidth * LAMwidth);"
        "LAMr = sqrt(x^2 + y^2 + z^2 + LAMtt^2)")
    sim_object.force_dict["Lamina attraction"] = laminaForce

    # adding all the particles on which force acts
    for i in range(sim_object.N):
        if sim_object.domains[i] > 0.5:
            laminaForce.addParticle(i, [])
    if r is None:
        try:
            r = sim_object.sphericalConfinementRadius
        except:
            raise ValueError("No spherical confinement radius defined"\
                             " yet. Apply spherical confinement first!")
    if sim_object.verbose == True:
        print("Lamina attraction added with r = %d" % r)

    laminaForce.addGlobalParameter("LAMaa", r * nm)
    laminaForce.addGlobalParameter("LAMwidth", width * nm)
    laminaForce.addGlobalParameter("LAMdepth", depth * sim_object.kT)
    laminaForce.addGlobalParameter("LAMtt", 0.01 * nm)


def useDomains(sim_object, domains=None, filename=None):
    """
    Sets up domains for the simulation.
    Also, pickles domain vector to "domains.dat".

    Parameters
    ----------

    domains : boolean array or None
        N-long array with domain vector
    filename : str or None
        Filename with pickled domain vector

    """

    if domains is not None:
        sim_object.domains = domains

    elif filename is not None:
        sim_object.domains = pickle.load(open(domains))
    else:
        sim_object.exit("You have to specify domain vector or filename!")

    if len(sim_object.domains) != sim_object.N:
        sim_object._exitProgram("Wrong domain lengths")

    pickle.dump(sim_object.domains, open(os.path.join(sim_object.folder,
        "domains.dat"), 'wb'))

def lennard_jones_force(
    sim_object, cutoff=2.5, domains=False, epsilonRep=0.24, epsilonAttr=0.27,
    blindFraction=(-1), sigmaRep=None, sigmaAttr=None):

    """
    Adds a lennard-jones force, that allows for mutual attraction.
    This is the slowest force out of all repulsive.

    .. note ::
        This is the only force that allows for so-called "exceptions'.
        Exceptions allow you to change parameters of the force
        for a specific pair of particles.
        This can be used to create short-range attraction between
        pairs of particles.
        See manual for Openmm.NonbondedForce.addException.

    Parameters
    ----------

    cutoff : float, optional
        Radius cutoff value. Default is good.
    domains : bool, optional
        Use domains, defined by
        :py:func:'setDomains <Simulation.setDomains>'
    epsilonRep : float, optional
        Epsilon (attraction strength) for LJ-force for all particles
        (except for domain) in kT
    epsilonAttr : float, optional
        Epsilon for attractive domain (if domains are used) in kT
    blindFraction : float, 0<x<1
        Fraction of particles that are "transparent" -
        used here instead of truncation
    sigmaRep, sigmaAttr: float, optional
        Radius of particles in the LJ force. For advanced fine-tuning.

     """
    sim_object.metadata["LennardJonesForce"] = repr({"cutoff": cutoff,
              "domains": domains, "epsilonRep": epsilonRep,
              "epsilonAttr": epsilonAttr, "blindFraction": blindFraction})

    if blindFraction > 0.99:
        sim_object._exitProgram("why do you need this force without particles???"\
                         " set blindFraction between 0 and 1")
    if (sigmaRep is None) and (sigmaAttr is None):
        sigmaAttr = sigmaRep = sim_object.conlen
    else:
        sigmaAttr = sigmaAttr * sim_object.conlen
        sigmaRep = sigmaRep * sim_object.conlen

    epsilonRep = epsilonRep * sim_object.kT
    epsilonAttr = epsilonAttr * sim_object.kT

    nbCutOffDist = sim_object.conlen * cutoff
    sim_object.epsilonRep = epsilonRep
    repulforce = openmm.NonbondedForce()

    sim_object.force_dict["Nonbonded"] = repulforce
    for i in range(sim_object.N):
        particleParameters = [0., 0., 0.]

        if np.random.random() > blindFraction:
            particleParameters[1] = (sigmaRep)
            particleParameters[2] = (epsilonRep)

            if domains == True:
                if sim_object.domains[i] != 0:
                    particleParameters[1] = (sigmaAttr)
                    particleParameters[2] = (epsilonAttr)

        repulforce.addParticle(*particleParameters)

    repulforce.setCutoffDistance(nbCutOffDist)




def soft_lennard_jones_force(sim_object, epsilon=0.42, trunc=2, cutoff=2.5):
    """A softened version of lennard-Jones force.
    Now we're moving to polynomial forces, so go there instead.
    """

    nbCutOffDist = sim_object.conlen * cutoff

    repul_energy = (
        'step(REPcut2 - REPU) * REPU +'
        ' step(REPU - REPcut2) * REPcut2 * (1 + tanh(REPU/REPcut2 - 1));'
        'REPU = 4 * REPe * ((REPsigma/r2)^12 - (REPsigma/r2)^6);'
        'r2 = (r^10. + (REPsigma03)^10.)^0.1')
    sim_object.force_dict["Nonbonded"] = openmm.CustomNonbondedForce(
        repul_energy)
    repulforceGr = sim_object.force_dict["Nonbonded"]
    repulforceGr.addGlobalParameter('REPe', sim_object.kT * epsilon)

    repulforceGr.addGlobalParameter('REPsigma', sim_object.conlen)
    repulforceGr.addGlobalParameter('REPsigma03', 0.3 * sim_object.conlen)
    repulforceGr.addGlobalParameter('REPcut', sim_object.kT * trunc)
    repulforceGr.addGlobalParameter('REPcut2', 0.5 * trunc * sim_object.kT)

    for _ in range(sim_object.N):
        repulforceGr.addParticle(())

    repulforceGr.setCutoffDistance(nbCutOffDist)

def attractive_interaction(sim_object, i, j, epsilon, sigma=None, length=3):
    """Adds attractive short-range interaction of strength epsilon
    between particles i,j and a few neighboring particles
    requires :py:func:'LennardJones Force<Simulation.addLennardJonesForce>'

    Parameters
    ----------
    i,j : int
        Interacting particles
    epsilon : float
        LJ strength
    sigma : float, optional
        LJ length. If you increase it past 1.5, note the cutoff!
    length : int, optional, default = 3
        Number of particles around i,j that also attract each other

    """

    if type(sim_object.force_dict["Nonbonded"]) != openmm.NonbondedForce:
        sim_object.exit("Cannot add interactions"\
                  " without Lennard-Jones nonbonded force")

    if sigma is None:
        sigma = 1.1 * sim_object.conlen
    epsilon = epsilon * units.kilocalorie_per_mole
    if (min(i, j) < length) or (max(i, j) > sim_object.N - length):
        print("!!!!!!!!!bond with %d and %d is out of range!!!!!" % (i, j))
        return
    repulforce = sim_object.force_dict["Nonbonded"]
    for t1 in range(int(np.ceil(i - length / 2)),int( np.ceil( i + (length - length / 2)))):
        for t2 in range(int(np.ceil(j - length / 2)), int(np.ceil( j + (length - length / 2))  )):
            repulforce.addException(t1, t2, 0, sigma, epsilon, True)
            if sim_object.verbose == True:
                print("Exception added between"\
                " particles %d and %d" % (t1, t2))

    for tt in range(i - length, i + length):
        repulforce.setParticleParameters(
            tt, 0, sim_object.conlen, sim_object.epsilonRep)
    for tt in range(j - length, j + length):
        repulforce.setParticleParameters(
            tt, 0, sim_object.conlen, sim_object.epsilonRep)



def gravity(sim_object, k=0.1, cutoff=None):
    """adds force pulling downwards in z direction
    When using cutoff, acts only when z>cutoff"""
    sim_object.metadata["gravity"] = repr({"k": k, "cutoff": cutoff})
    if cutoff is None:
        gravity = openmm.CustomExternalForce("kG * z")
    else:
        gravity = openmm.CustomExternalForce(
            "kG * (z - cutoffG) * step(z - cutoffG)")
        gravity.addGlobalParameter("cutoffG", cutoff * nm)
    gravity.addGlobalParameter("kG", k * sim_object.kT / (nm))

    for i in range(sim_object.N):
        gravity.addParticle(i, [])
    sim_object.force_dict["Gravity"] = gravity


def exclude_sphere(sim_object, r=5, position=(0, 0, 0)):
    """Excludes particles from a sphere of radius r at certain position.
    """

    spherForce = openmm.CustomExternalForce(
        "step(EXaa-r) * EXkb * (sqrt((r-EXaa)*(r-EXaa) + EXt*EXt) - EXt) ;"
        "r = sqrt((x-EXx)^2 + (y-EXy)^2 + (z-EXz)^2 + EXtt^2)")
    sim_object.force_dict["ExcludeSphere"] = spherForce

    for i in range(sim_object.N):
        spherForce.addParticle(i, [])

    sim_object.sphericalConfinementRadius = r
    if sim_object.verbose == True:
        print("Spherical confinement with radius = %lf" % r)
    # assigning parameters of the force
    spherForce.addGlobalParameter("EXkb", 2 * sim_object.kT / nm)
    spherForce.addGlobalParameter("EXaa", (r - 1. / 3) * nm)
    spherForce.addGlobalParameter("EXt", (1. / 3) * nm / 10.)
    spherForce.addGlobalParameter("EXtt", 0.01 * nm)
    spherForce.addGlobalParameter("EXx", position[0] * sim_object.conlen)
    spherForce.addGlobalParameter("EXy", position[1] * sim_object.conlen)
    spherForce.addGlobalParameter("EXz", position[2] * sim_object.conlen)
    
def attraction_to_the_core(sim_object, k, r0, coreParticles=[]):

    """Attracts a subset of particles to the core,
     repells the rest from the core"""

    attractForce = openmm.CustomExternalForce(
        " COREk * ((COREr - CORErn) ^ 2)  ; "\
        "COREr = sqrt(x^2 + y^2 + COREtt^2)")
    attractForce.addGlobalParameter(
        "COREk", k * sim_object.kT / (sim_object.conlen * sim_object.conlen))
    attractForce.addGlobalParameter("CORErn", r0 * sim_object.conlen)
    attractForce.addGlobalParameter("COREtt", 0.001 * sim_object.conlen)
    sim_object.force_dict["CoreAttraction"] = attractForce
    for i in coreParticles:
        attractForce.addParticle(int(i), [])

    if r0 > 0.1:

        excludeForce = openmm.CustomExternalForce(
            " CORE2k * ((CORE2r - CORE2rn) ^ 2) * step(CORE2rn - CORE2r) ;"
            "CORE2r = sqrt(x^2 + y^2 + CORE2tt^2)")
        excludeForce.addGlobalParameter("CORE2k", k *
            sim_object.kT / (sim_object.conlen * sim_object.conlen))
        excludeForce.addGlobalParameter("CORE2rn", r0 * sim_object.conlen)
        excludeForce.addGlobalParameter("CORE2tt", 0.001 * sim_object.conlen)
        sim_object.force_dict["CoreExclusion"] = excludeForce
        for i in range(sim_object.N):
            excludeForce.addParticle(i, [])



def create_walls(sim_object, left=None, right=None, k=0.5):
    "creates walls at x = left, x = right, x direction only"
    if left is None:
        left = sim_object.data[0][0] + 1. * nm
    else:
        left = 1. * nm * left
    if right is None:
        right = sim_object.data[-1][0] - 1. * nm
    else:
        right = 1. * nm * right

    if sim_object.verbose == True:
        print("left wall created at ", left / (1. * nm))
        print("right wall created at ", right / (1. * nm))

    extforce2 = openmm.CustomExternalForce(
        " WALLk * (sqrt((x - WALLright) * (x-WALLright) + WALLa * WALLa ) - WALLa) * step(x-WALLright) "
        "+ WALLk * (sqrt((x - WALLleft) * (x-WALLleft) + WALLa * WALLa ) - WALLa) * step(WALLleft - x) ")
    extforce2.addGlobalParameter("WALLk", k * sim_object.kT / nm)
    extforce2.addGlobalParameter("WALLleft", left)
    extforce2.addGlobalParameter("WALLright", right)
    extforce2.addGlobalParameter("WALLa", 1 * nm)
    for i in range(sim_object.N):
        extforce2.addParticle(i, [])
    sim_object.force_dict["WALL Force"] = extforce2



def spherical_well(sim_object, r=10, depth=1):
    """pushes particles towards a boundary
    of a cylindrical well to create uniform well coverage"""

    extforce4 = openmm.CustomExternalForce(
        "WELLdepth * (((sin((WELLr * 3.141592 * 0.5) / WELLwidth)) ^ 10)  -1) * step(-WELLr + WELLwidth);"
        "WELLr = sqrt(x^2 + y^2 + z^2 + WELLtt^2)")
    sim_object.force_dict["Well attraction"] = extforce4

    # adding all the particles on which force acts
    for i in range(sim_object.N):
        if sim_object.domains[i] > 0.5:
            extforce4.addParticle(i, [])
    if r is None:
        try:
            r = sim_object.sphericalConfinementRadius * 0.5
        except:
            exit("No spherical confinement radius defined yet."\
                 " Apply spherical confinement first!")
    if sim_object.verbose == True:
        print("Well attraction added with r = %d" % r)

    # assigning parameters of the force
    extforce4.addGlobalParameter("WELLwidth", r * nm)
    extforce4.addGlobalParameter("WELLdepth", depth * sim_object.kT)
    extforce4.addGlobalParameter("WELLtt", 0.01 * nm)


## from class "yeast simulation" 


def add_nucleolus(sim_object, k=1, r=None):
    "method"
    if r is None:
        r = sim_object.sphericalConfinementRadius

    extforce3 = openmm.CustomExternalForce(
        "step(r-NUCaa) * NUCkb * (sqrt((r-NUCaa)*(r-NUCaa) + NUCt*NUCt) - NUCt);"
        "r = sqrt(x^2 + y^2 + (z + NUCoffset )^2 + NUCtt^2)")

    sim_object.force_dict["NucleolusConfinement"] = extforce3
    # adding all the particles on which force acts
    if sim_object.verbose == True:
        print("NUCleolus confinement from radius = %lf" % r)
    # assigning parameters of the force
    extforce3.addGlobalParameter("NUCkb", k * sim_object.kT / nm)
    extforce3.addGlobalParameter("NUCaa", (r - 1. / k) * nm * 1.75)
    extforce3.addGlobalParameter("NUCoffset", (r - 1. / k) * nm * 1.1)
    extforce3.addGlobalParameter("NUCt", (1. / k) * nm / 10.)
    extforce3.addGlobalParameter("NUCtt", 0.01 * nm)
    for i in range(sim_object.N):
        extforce3.addParticle(i, [])

def add_lamina_attraction(sim_object, width=1, depth=1, r=None, particles=None):
    extforce3 = openmm.CustomExternalForce(
        "-1 * step(LAMr-LAMaa + LAMwidth) * step(LAMaa + LAMwidth - LAMr) * LAMdepth"
        "* abs( (LAMr-LAMaa + LAMwidth) * (LAMaa + LAMwidth - LAMr)) / (LAMwidth * LAMwidth);"
        "LAMr = sqrt(x^2 + y^2 + z^2 + LAMtt^2)")
    sim_object.force_dict["Lamina attraction"] = extforce3

    # re-defines lamina attraction based on particle index instead of domains.

    # adding all the particles on which force acts
    if particles is None:
        for i in range(sim_object.N):
            extforce3.addParticle(i, [])
            if sim_object.verbose == True:
                print("particle %d laminated! " % i)

    else:
        for i in particles:
            extforce3.addParticle(i, [])
            if sim_object.verbose == True:
                print("particle %d laminated! " % i)

    if r is None:
        try:
            r = sim_object.sphericalConfinementRadius
        except:
            exit("No spherical confinement radius defined yet."\
                 "Apply spherical confinement first!")

    if sim_object.verbose == True:
        print("Lamina attraction added with r = %d" % r)

    # assigning parameters of the force
    extforce3.addGlobalParameter("LAMaa", r * nm)
    extforce3.addGlobalParameter("LAMwidth", width * nm)
    extforce3.addGlobalParameter("LAMdepth", depth * sim_object.kT)
    extforce3.addGlobalParameter("LAMtt", 0.01 * nm)

    
    
    
# old energy minimization 

def old_energy_minimization(sim_object, stepsPerIteration=100,
                       maxIterations=1000,
                       failNotConverged=True):
    """Runs system at smaller timestep and higher collision
    rate to resolve possible conflicts.

    Now we're moving towards local energy minimization,
    this is here for backwards compatibility.
    """

    print("Performing energy minimization")
    sim_object._applyForces()
    oldName = sim_object.name
    sim_object.name = "minim"
    if (maxIterations is True) or (maxIterations is False):
        raise ValueError(
            "Please stop using the old notation and read the new energy minimization code")
    if (failNotConverged is not True) and (failNotConverged is not False):
        raise ValueError(
            "Please stop using the old notation and read the new energy minimization code")

    def_step = sim_object.integrator.getStepSize()
    def_fric = sim_object.integrator.getFriction()

    def minimizeDrop():
        drop = 10.
        for dummy in range(maxIterations):
            if drop < 1:
                drop = 1.
            if drop > 10000:
                raise RuntimeError("Timestep too low. Perhaps, "\
                                   "something is wrong!")

            sim_object.integrator.setStepSize(def_step / float(drop))
            sim_object.integrator.setFriction(def_fric * drop)
            # sim_object.reinitialize()
            numAttempts = 5
            for attempt in range(numAttempts):
                a = sim_object.doBlock(stepsPerIteration, increment=False,
                    reinitialize=False)
                # sim_object.initVelocities()
                if a == False:
                    drop *= 2
                    print("Timestep decreased {0}".format(1. / drop))
                    sim_object.initVelocities()
                    break
                if attempt == numAttempts - 1:
                    if drop == 1.:
                        return 0
                    drop /= 2
                    print("Timestep decreased by {0}".format(drop))
                    sim_object.initVelocities()
        return -1

    if failNotConverged and (minimizeDrop() == -1):
        raise RuntimeError(
            "Reached maximum number of iterations and still not converged\n"\
            "increase maxIterations or set failNotConverged=False")
    sim_object.name = oldName
    sim_object.integrator.setFriction(def_fric)
    sim_object.integrator.setStepSize(def_step)
    # sim_object.reinitialize()
    print("Finished energy minimization")

    
    
def check_connectivity(sim_object, newcoords=None, maxBondSizeMultipler=10):
    ''' checks connectivity of all harmonic (& abslim) bonds
        can be passed to doBlock as a checkFunction, in which case it will also trigger re-initialization
        to modify the maximum bond size multipler, pass this function to doBlock as, 
        e.g. doBlock( 100,checkFunctions = [lambda x:a.checkConnectivity(x,6)])
    '''

    if not hasattr(sim_object, "bondLengths"):
        raise ValueError('must use either harmonic or abs bonds to use checkConnectivty')

    if newcoords == None:
        newcoords = sim_object.get_data()
        printPositiveResult = True
    else: printPositiveResult = False

    # sim_object.bondLengths is a list of lists (see above) [..., [int(i), int(j), float(distance), float(bondSize)], ...]
    bondArray = np.array(sim_object.bondLengths)
    bondDists = np.sqrt(np.sum((newcoords[  np.array(bondArray[:, 0], dtype=int) ] - newcoords[ np.array(bondArray[:, 1], dtype=int) ]) ** 2, axis=1))
    bondDistsSorted = np.sort(bondDists)
    if (bondDists > (bondArray[:, 2] + maxBondSizeMultipler * bondArray[:, 3])).any():
        isConnected = False
        print("!! connectivity check failed !!")
        print("median bond size is ", np.median(bondDists))
        print("longest 10 bonds are", bondDistsSorted[-10:])

    else:
        isConnected = True
        if printPositiveResult:
            print("connectivity check passed.")
            print("median bond size is ", np.median(bondDists))
            print("longest 10 bonds are", bondDistsSorted[-10:])

    return isConnected

