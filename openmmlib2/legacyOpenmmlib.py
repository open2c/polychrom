
"""
This is a collection of old forces that are likely no longer used 
These were a part of openmmlib before April 2019, but were removed during spring cleaning. 



"""

def _initGrosbergBondForce(self):
    "inits Grosberg FENE bond force"
    if "GrosbergBondForce" not in list(self.forceDict.keys()):
        force = ("- 0.5 * GROSk * GROSr0 * GROSr0 * log(1-(r/GROSr0)* (r / GROSr0))"
            " + (4 * GROSe * ((GROSs/r)^12 - (GROSs/r)^6) + GROSe) * step(GROScut - r)")
        bondforceGr = self.mm.CustomBondForce(force)
        bondforceGr.addGlobalParameter("GROSk", 30 *
            self.kT / (self.conlen * self.conlen))
        bondforceGr.addGlobalParameter("GROSr0", self.conlen * 1.5)
        bondforceGr.addGlobalParameter('GROSe', self.kT)
        bondforceGr.addGlobalParameter('GROSs', self.conlen)
        bondforceGr.addGlobalParameter(
            "GROScut", self.conlen * 2. ** (1. / 6.))
        self.forceDict["GrosbergBondForce"] = bondforceGr
"""
elif bondType.lower() == "grosberg":
    self._initGrosbergBondForce()  # has no per-bond parameters - yay! 
    self.forceDict["GrosbergBondForce"].addBond(int(i), int(j), [])
"""

def addGrosbergPolymerBonds(self, k=30):
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

     """

    for start, end, isRing in self.chains:
        for j in range(start, end - 1):
            self.addBond(j, j + 1, bondType="Grosberg")
            self.bondsForException.append((j, j + 1))

        if isRing:
            self.addBond(start, end - 1, distance=1, bondType="Harmonic")
            self.bondsForException.append((start, end - 1))
            if self.verbose == True:
                print("ring bond added", start, end - 1)

    self.metadata["GorsbergPolymerForce"] = repr({"k": k})


def addGrosbergStiffness(self, k=1.5):
    """Adds stiffness according to the Grosberg paper.
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

    """
    try:
        k[0]
    except:
        k = numpy.zeros(self.N, float) + k
    stiffForce = self.mm.CustomAngleForce(
        "GRk * kT * (1 - cos(theta - 3.141592))")
    self.forceDict["AngleForce"] = stiffForce

    stiffForce.addGlobalParameter("kT", self.kT)
    stiffForce.addPerAngleParameter("GRk")
    for start, end, isRing in self.chains:
        for j in range(start + 1, end - 1):
            stiffForce.addAngle(j - 1, j, j + 1, [k[j]])
        if isRing:
            stiffForce.addAngle(end - 2, end - 1, start, [k[end - 1]])
            stiffForce.addAngle(end - 1, start, start + 1, [k[start]])

    self.metadata["GrosbergAngleForce"] = repr({"stiffness": k})

def addMinimizingRepulsiveForce(self):
    """
    Adds a special force which could be use for very efficient resolution of crossings
    Use this force to perform (local) energy minimization if your monomers are all "on top of each other"
    E.g. if you start your simulations with fractional brownyan motion with h < 0.4
    Then switch to a normal force, and re-do energy minimization. 
    """
    radius = self.conlen * 1.3

    nbCutOffDist = radius * 1.
    repul_energy = "1000* REPe * (1-r/REPr)^2 "

    self.forceDict["NonbondedMinim"] = self.mm.CustomNonbondedForce(
        repul_energy)
    repulforceGr = self.forceDict["NonbondedMinim"]
    repulforceGr.addGlobalParameter('REPe', self.kT)
    repulforceGr.addGlobalParameter('REPr', self.kT)
    for _ in range(self.N):
        repulforceGr.addParticle(())
    repulforceGr.setCutoffDistance(nbCutOffDist)

    
            

def fixParticlesZCoordinate(self, particles, zCoordinates, k=0.3,
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
                stop - start) * (par / float(self.N)))

    if (mode == "abs") and (gap is None):
        zFixForce = self.mm.CustomExternalForce(
        "ZFIXk * (sqrt((%s - ZFIXr0)^2 + ZFIXa^2) - ZFIXa)" % (
                                                       useOtherAxis,))
        zFixForce.addGlobalParameter("ZFIXk", k * self.kT / (self.conlen))
    elif (mode == "abs") and (gap is not None):
        zFixForce = self.mm.CustomExternalForce(
        "ZFIXk * step(%s - ZFIXr0 - ZFIXgap * 0.5) *"\
        " (sqrt((%s - ZFIXr0 - ZFIXgap * 0.5)^2 + ZFIXa^2) - ZFIXa) + "\
        "ZFIXk * step(-%s + ZFIXr0 - ZFIXgap * 0.5) * "\
        "(sqrt((-%s + ZFIXr0 - ZFIXgap * 0.5)^2 + ZFIXa^2) - ZFIXa)"\
        % (useOtherAxis, useOtherAxis, useOtherAxis, useOtherAxis))

        zFixForce.addGlobalParameter("ZFIXk", k * self.kT / (self.conlen))
        zFixForce.addGlobalParameter("ZFIXgap", self.conlen * gap)

    elif (mode == "quadratic") and (gap is None):
        zFixForce = self.mm.CustomExternalForce(
            "ZFIXk * ((%s - ZFIXr0)^2)" % (useOtherAxis,))
        zFixForce.addGlobalParameter("ZFIXk", k * self.kT /
            (self.conlen * self.conlen))
    elif (mode == "quadratic") and (gap is not None):
        zFixForce = self.mm.CustomExternalForce(
        "ZFIXk * (step(%s - ZFIXr0 - ZFIXgap * 0.5) * "\
        "(%s - ZFIXr0 - ZFIXgap * 0.5)^2 +  "\
        "step(-%s + ZFIXr0 - ZFIXgap * 0.5) * "\
        "(-%s + ZFIXr0 - ZFIXgap * 0.5)^2)" \
        % (useOtherAxis, useOtherAxis, useOtherAxis, useOtherAxis))

        zFixForce.addGlobalParameter("ZFIXk", k * self.kT /
            (self.conlen * self.conlen))
        zFixForce.addGlobalParameter("ZFIXgap", self.conlen * gap)
    else:
        raise RuntimeError("Not implemented")

    zFixForce.addPerParticleParameter("ZFIXr0")

    zFixForce.addGlobalParameter("ZFIXa", 0.05 * self.conlen)
    for par, zcoor in zip(particles, zCoordinates):
        zFixForce.addParticle(int(par), [float(zcoor)])
    self.forceDict["fixZCoordinates"] = zFixForce




def addGrosbergRepulsiveForce(self, trunc=None, radiusMult=1.):
    """This is the fastest non-transparent repulsive force.
    (that preserves topology, doesn't allow chain passing)
    Done according to the paper:
    (Halverson, Jonathan D., et al. "Molecular dynamics simulation study of
     nonconcatenated ring polymers in a melt. I. Statics."
     The Journal of chemical physics 134 (2011): 204904.)
    Parameters
    ----------

    trunc : None or float
         truncation energy in kT, used for chain crossing.
         Value of 1.5 yields frequent passing,
         3 - average passing, 5 - rare passing.

    """
    radius = self.conlen * radiusMult
    self.metadata["GrosbergRepulsiveForce"] = repr({"trunc": trunc})
    nbCutOffDist = radius * 2. ** (1. / 6.)
    if trunc is None:
        repul_energy = "4 * REPe * ((REPsigma/r)^12 - (REPsigma/r)^6) + REPe"
    else:
        repul_energy = (
            "step(REPcut2 - REPU) * REPU"
            " + step(REPU - REPcut2) * REPcut2 * (1 + tanh(REPU/REPcut2 - 1));"
            "REPU = 4 * REPe * ((REPsigma/r2)^12 - (REPsigma/r2)^6) + REPe;"
            "r2 = (r^10. + (REPsigma03)^10.)^0.1")
    self.forceDict["Nonbonded"] = self.mm.CustomNonbondedForce(
        repul_energy)
    repulforceGr = self.forceDict["Nonbonded"]
    repulforceGr.addGlobalParameter('REPe', self.kT)

    repulforceGr.addGlobalParameter('REPsigma', radius)
    repulforceGr.addGlobalParameter('REPsigma03', 0.3 * radius)
    if trunc is not None:
        repulforceGr.addGlobalParameter('REPcut', self.kT * trunc)
        repulforceGr.addGlobalParameter('REPcut2', 0.5 * trunc * self.kT)
    for _ in range(self.N):
        repulforceGr.addParticle(())

    repulforceGr.setCutoffDistance(nbCutOffDist)
    
    
def addLaminaAttraction(self, width=1, depth=1, r=None):
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

    self.metadata["laminaAttraction"] = repr({"width": width,
        "depth": depth, "r": r})
    laminaForce = self.mm.CustomExternalForce(
        "step(LAMr-LAMaa + LAMwidth) * step(LAMaa + LAMwidth - LAMr) "
        "* LAMdepth * (LAMr-LAMaa + LAMwidth) * (LAMaa + LAMwidth - LAMr) "
        "/ (LAMwidth * LAMwidth);"
        "LAMr = sqrt(x^2 + y^2 + z^2 + LAMtt^2)")
    self.forceDict["Lamina attraction"] = laminaForce

    # adding all the particles on which force acts
    for i in range(self.N):
        if self.domains[i] > 0.5:
            laminaForce.addParticle(i, [])
    if r is None:
        try:
            r = self.sphericalConfinementRadius
        except:
            raise ValueError("No spherical confinement radius defined"\
                             " yet. Apply spherical confinement first!")
    if self.verbose == True:
        print("Lamina attraction added with r = %d" % r)

    laminaForce.addGlobalParameter("LAMaa", r * nm)
    laminaForce.addGlobalParameter("LAMwidth", width * nm)
    laminaForce.addGlobalParameter("LAMdepth", depth * self.kT)
    laminaForce.addGlobalParameter("LAMtt", 0.01 * nm)


def useDomains(self, domains=None, filename=None):
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
        self.domains = domains

    elif filename is not None:
        self.domains = pickle.load(open(domains))
    else:
        self.exit("You have to specify domain vector or filename!")

    if len(self.domains) != self.N:
        self._exitProgram("Wrong domain lengths")

    pickle.dump(self.domains, open(os.path.join(self.folder,
        "domains.dat"), 'wb'))

def addLennardJonesForce(
    self, cutoff=2.5, domains=False, epsilonRep=0.24, epsilonAttr=0.27,
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
    self.metadata["LennardJonesForce"] = repr({"cutoff": cutoff,
              "domains": domains, "epsilonRep": epsilonRep,
              "epsilonAttr": epsilonAttr, "blindFraction": blindFraction})

    if blindFraction > 0.99:
        self._exitProgram("why do you need this force without particles???"\
                         " set blindFraction between 0 and 1")
    if (sigmaRep is None) and (sigmaAttr is None):
        sigmaAttr = sigmaRep = self.conlen
    else:
        sigmaAttr = sigmaAttr * self.conlen
        sigmaRep = sigmaRep * self.conlen

    epsilonRep = epsilonRep * self.kT
    epsilonAttr = epsilonAttr * self.kT

    nbCutOffDist = self.conlen * cutoff
    self.epsilonRep = epsilonRep
    repulforce = self.mm.NonbondedForce()

    self.forceDict["Nonbonded"] = repulforce
    for i in range(self.N):
        particleParameters = [0., 0., 0.]

        if numpy.random.random() > blindFraction:
            particleParameters[1] = (sigmaRep)
            particleParameters[2] = (epsilonRep)

            if domains == True:
                if self.domains[i] != 0:
                    particleParameters[1] = (sigmaAttr)
                    particleParameters[2] = (epsilonAttr)

        repulforce.addParticle(*particleParameters)

    repulforce.setCutoffDistance(nbCutOffDist)




def addSoftLennardJonesForce(self, epsilon=0.42, trunc=2, cutoff=2.5):
    """A softened version of lennard-Jones force.
    Now we're moving to polynomial forces, so go there instead.
    """

    nbCutOffDist = self.conlen * cutoff

    repul_energy = (
        'step(REPcut2 - REPU) * REPU +'
        ' step(REPU - REPcut2) * REPcut2 * (1 + tanh(REPU/REPcut2 - 1));'
        'REPU = 4 * REPe * ((REPsigma/r2)^12 - (REPsigma/r2)^6);'
        'r2 = (r^10. + (REPsigma03)^10.)^0.1')
    self.forceDict["Nonbonded"] = self.mm.CustomNonbondedForce(
        repul_energy)
    repulforceGr = self.forceDict["Nonbonded"]
    repulforceGr.addGlobalParameter('REPe', self.kT * epsilon)

    repulforceGr.addGlobalParameter('REPsigma', self.conlen)
    repulforceGr.addGlobalParameter('REPsigma03', 0.3 * self.conlen)
    repulforceGr.addGlobalParameter('REPcut', self.kT * trunc)
    repulforceGr.addGlobalParameter('REPcut2', 0.5 * trunc * self.kT)

    for _ in range(self.N):
        repulforceGr.addParticle(())

    repulforceGr.setCutoffDistance(nbCutOffDist)

def addInteraction(self, i, j, epsilon, sigma=None, length=3):
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

    if type(self.forceDict["Nonbonded"]) != self.mm.NonbondedForce:
        self.exit("Cannot add interactions"\
                  " without Lennard-Jones nonbonded force")

    if sigma is None:
        sigma = 1.1 * self.conlen
    epsilon = epsilon * units.kilocalorie_per_mole
    if (min(i, j) < length) or (max(i, j) > self.N - length):
        print("!!!!!!!!!bond with %d and %d is out of range!!!!!" % (i, j))
        return
    repulforce = self.forceDict["Nonbonded"]
    for t1 in range(int(np.ceil(i - length / 2)),int( np.ceil( i + (length - length / 2)))):
        for t2 in range(int(np.ceil(j - length / 2)), int(np.ceil( j + (length - length / 2))  )):
            repulforce.addException(t1, t2, 0, sigma, epsilon, True)
            if self.verbose == True:
                print("Exception added between"\
                " particles %d and %d" % (t1, t2))

    for tt in range(i - length, i + length):
        repulforce.setParticleParameters(
            tt, 0, self.conlen, self.epsilonRep)
    for tt in range(j - length, j + length):
        repulforce.setParticleParameters(
            tt, 0, self.conlen, self.epsilonRep)



def addMutualException(self, particles):
    """used to exclude a bunch of particles
    from calculation of nonbonded force

    Parameters
    ----------
    particles : list
        List of particles for whom to exclude nonbonded force.
    """
    for i in particles:  # xrange(len(particles)):
        for j in particles:  # xrange(len(particles)):
            if j > i:
                self.bondsForException.append((i, j))


def _initAbsDistanceLimitation(self):
    "inits abs(x) FENE bond force"
    if "AbsLimitation" not in list(self.forceDict.keys()):
        force = ("(1. / ABSwiggle) * ABSunivK * step(r - ABSr0 * ABSconlen) "
            "* (sqrt((r-ABSr0 * ABSconlen)"
            "*(r - ABSr0 * ABSconlen) + ABSa * ABSa) - ABSa)")
        bondforceAbsLim = self.mm.CustomBondForce(force)
        bondforceAbsLim.addPerBondParameter("ABSwiggle")
        bondforceAbsLim.addPerBondParameter("ABSr0")
        bondforceAbsLim.addGlobalParameter(
            "ABSunivK", self.kT / self.conlen)
        bondforceAbsLim.addGlobalParameter("ABSa", 0.02 * self.conlen)
        bondforceAbsLim.addGlobalParameter("ABSconlen", self.conlen)
        self.forceDict["AbsLimitation"] = bondforceAbsLim


        
"""
elif bondType.lower() == "abslim":
    self._initAbsDistanceLimitation()
    self.forceDict["AbsLimitation"].addBond(int(i), int(
        j), [float(bondWiggleDistance), float(distance)])  # same 
""" 

def addGravity(self, k=0.1, cutoff=None):
    """adds force pulling downwards in z direction
    When using cutoff, acts only when z>cutoff"""
    self.metadata["gravity"] = repr({"k": k, "cutoff": cutoff})
    if cutoff is None:
        gravity = self.mm.CustomExternalForce("kG * z")
    else:
        gravity = self.mm.CustomExternalForce(
            "kG * (z - cutoffG) * step(z - cutoffG)")
        gravity.addGlobalParameter("cutoffG", cutoff * nm)
    gravity.addGlobalParameter("kG", k * self.kT / (nm))

    for i in range(self.N):
        gravity.addParticle(i, [])
    self.forceDict["Gravity"] = gravity


def excludeSphere(self, r=5, position=(0, 0, 0)):
    """Excludes particles from a sphere of radius r at certain position.
    """

    spherForce = self.mm.CustomExternalForce(
        "step(EXaa-r) * EXkb * (sqrt((r-EXaa)*(r-EXaa) + EXt*EXt) - EXt) ;"
        "r = sqrt((x-EXx)^2 + (y-EXy)^2 + (z-EXz)^2 + EXtt^2)")
    self.forceDict["ExcludeSphere"] = spherForce

    for i in range(self.N):
        spherForce.addParticle(i, [])

    self.sphericalConfinementRadius = r
    if self.verbose == True:
        print("Spherical confinement with radius = %lf" % r)
    # assigning parameters of the force
    spherForce.addGlobalParameter("EXkb", 2 * self.kT / nm)
    spherForce.addGlobalParameter("EXaa", (r - 1. / 3) * nm)
    spherForce.addGlobalParameter("EXt", (1. / 3) * nm / 10.)
    spherForce.addGlobalParameter("EXtt", 0.01 * nm)
    spherForce.addGlobalParameter("EXx", position[0] * self.conlen)
    spherForce.addGlobalParameter("EXy", position[1] * self.conlen)
    spherForce.addGlobalParameter("EXz", position[2] * self.conlen)
def addAttractionToTheCore(self, k, r0, coreParticles=[]):

    """Attracts a subset of particles to the core,
     repells the rest from the core"""

    attractForce = self.mm.CustomExternalForce(
        " COREk * ((COREr - CORErn) ^ 2)  ; "\
        "COREr = sqrt(x^2 + y^2 + COREtt^2)")
    attractForce.addGlobalParameter(
        "COREk", k * self.kT / (self.conlen * self.conlen))
    attractForce.addGlobalParameter("CORErn", r0 * self.conlen)
    attractForce.addGlobalParameter("COREtt", 0.001 * self.conlen)
    self.forceDict["CoreAttraction"] = attractForce
    for i in coreParticles:
        attractForce.addParticle(int(i), [])

    if r0 > 0.1:

        excludeForce = self.mm.CustomExternalForce(
            " CORE2k * ((CORE2r - CORE2rn) ^ 2) * step(CORE2rn - CORE2r) ;"
            "CORE2r = sqrt(x^2 + y^2 + CORE2tt^2)")
        excludeForce.addGlobalParameter("CORE2k", k *
            self.kT / (self.conlen * self.conlen))
        excludeForce.addGlobalParameter("CORE2rn", r0 * self.conlen)
        excludeForce.addGlobalParameter("CORE2tt", 0.001 * self.conlen)
        self.forceDict["CoreExclusion"] = excludeForce
        for i in range(self.N):
            excludeForce.addParticle(i, [])



def addDoubleRandomLengthBonds(self, bondlength, bondRange, distance):
    begin = 4
    started = True
    past = 0
    while True:
        past
        b1 = begin
        b2 = begin + numpy.random.randint(
            0.5 * bondlength, 1.7 * bondlength)
        if b2 > self.N - 4:
            break
        self.addBond(b1, b2, bondRange, distance)
        if self.verbose == True:
            print("bond added between %d and %d" % (b1, b2))
        if started == False:
            self.addBond(past, b2, bondRange, distance)
            if self.verbose == True:
                print("bond added between %d and %d" % (past, b2))
            past = b1
        started = False
        begin = b2


def addConsecutiveRandomBonds(self, loopSize, bondWiggle, bondLength=0.,
                              smeerLoopSize=0.2, distanceBetweenBonds=2,
                              verbose=False):
    shift = int(loopSize * smeerLoopSize)
    if shift == 0:
        shift = 1
    begin = numpy.random.randint(distanceBetweenBonds)
    consecutiveRandomBondList = []
    while True:
        b1 = begin
        b2 = begin + loopSize + numpy.random.randint(shift)
        if b2 > self.N - 3:
            if (self.N - b1) > (5 * distanceBetweenBonds + 5):
                b2 = self.N - 1 - numpy.random.randint(distanceBetweenBonds)
            else:
                break

        self.addBond(b1, b2, bondWiggle, bondLength,
                     verbose=verbose)
        consecutiveRandomBondList.append([b1, b2])
        begin = b2 + numpy.random.randint(distanceBetweenBonds)
        if self.verbose == True:
            print("bond added between %d and %d" % (b1, b2))
    self.metadata['consecutiveRandomBondList'] = consecutiveRandomBondList
        
def quickLoad(self, data, mode="chain", Nchains=1,
              trunc=None, confinementDensity="NoConfinement"):
    """quickly loads a set of repulsive chains,
    possibly adds spherical confinement"""
    self.setup()
    self.load(data)
    self.setLayout(mode, Nchains)
    self.addHarmonicPolymerBonds()
    self.addSimpleRepulsiveForce(trunc=trunc)
    if type(confinementDensity) != str:
        self.addSphericalConfinement(density=confinementDensity)

def createWalls(self, left=None, right=None, k=0.5):
    "creates walls at x = left, x = right, x direction only"
    if left is None:
        left = self.data[0][0] + 1. * nm
    else:
        left = 1. * nm * left
    if right is None:
        right = self.data[-1][0] - 1. * nm
    else:
        right = 1. * nm * right

    if self.verbose == True:
        print("left wall created at ", left / (1. * nm))
        print("right wall created at ", right / (1. * nm))

    extforce2 = self.mm.CustomExternalForce(
        " WALLk * (sqrt((x - WALLright) * (x-WALLright) + WALLa * WALLa ) - WALLa) * step(x-WALLright) "
        "+ WALLk * (sqrt((x - WALLleft) * (x-WALLleft) + WALLa * WALLa ) - WALLa) * step(WALLleft - x) ")
    extforce2.addGlobalParameter("WALLk", k * self.kT / nm)
    extforce2.addGlobalParameter("WALLleft", left)
    extforce2.addGlobalParameter("WALLright", right)
    extforce2.addGlobalParameter("WALLa", 1 * nm)
    for i in range(self.N):
        extforce2.addParticle(i, [])
    self.forceDict["WALL Force"] = extforce2



def addSphericalWell(self, r=10, depth=1):
    """pushes particles towards a boundary
    of a cylindrical well to create uniform well coverage"""

    extforce4 = self.mm.CustomExternalForce(
        "WELLdepth * (((sin((WELLr * 3.141592 * 0.5) / WELLwidth)) ^ 10)  -1) * step(-WELLr + WELLwidth);"
        "WELLr = sqrt(x^2 + y^2 + z^2 + WELLtt^2)")
    self.forceDict["Well attraction"] = extforce4

    # adding all the particles on which force acts
    for i in range(self.N):
        if self.domains[i] > 0.5:
            extforce4.addParticle(i, [])
    if r is None:
        try:
            r = self.sphericalConfinementRadius * 0.5
        except:
            exit("No spherical confinement radius defined yet."\
                 " Apply spherical confinement first!")
    if self.verbose == True:
        print("Well attraction added with r = %d" % r)

    # assigning parameters of the force
    extforce4.addGlobalParameter("WELLwidth", r * nm)
    extforce4.addGlobalParameter("WELLdepth", depth * self.kT)
    extforce4.addGlobalParameter("WELLtt", 0.01 * nm)



class YeastSimulation(Simulation):
    """
    This class is maintained by Geoff to do simulations for the Yeast project
    """

    def addNucleolus(self, k=1, r=None):
        "method"
        if r is None:
            r = self.sphericalConfinementRadius

        extforce3 = self.mm.CustomExternalForce(
            "step(r-NUCaa) * NUCkb * (sqrt((r-NUCaa)*(r-NUCaa) + NUCt*NUCt) - NUCt);"
            "r = sqrt(x^2 + y^2 + (z + NUCoffset )^2 + NUCtt^2)")

        self.forceDict["NucleolusConfinement"] = extforce3
        # adding all the particles on which force acts
        if self.verbose == True:
            print("NUCleolus confinement from radius = %lf" % r)
        # assigning parameters of the force
        extforce3.addGlobalParameter("NUCkb", k * self.kT / nm)
        extforce3.addGlobalParameter("NUCaa", (r - 1. / k) * nm * 1.75)
        extforce3.addGlobalParameter("NUCoffset", (r - 1. / k) * nm * 1.1)
        extforce3.addGlobalParameter("NUCt", (1. / k) * nm / 10.)
        extforce3.addGlobalParameter("NUCtt", 0.01 * nm)
        for i in range(self.N):
            extforce3.addParticle(i, [])

    def addLaminaAttraction(self, width=1, depth=1, r=None, particles=None):
        extforce3 = self.mm.CustomExternalForce(
            "-1 * step(LAMr-LAMaa + LAMwidth) * step(LAMaa + LAMwidth - LAMr) * LAMdepth"
            "* abs( (LAMr-LAMaa + LAMwidth) * (LAMaa + LAMwidth - LAMr)) / (LAMwidth * LAMwidth);"
            "LAMr = sqrt(x^2 + y^2 + z^2 + LAMtt^2)")
        self.forceDict["Lamina attraction"] = extforce3

        # re-defines lamina attraction based on particle index instead of domains.

        # adding all the particles on which force acts
        if particles is None:
            for i in range(self.N):
                extforce3.addParticle(i, [])
                if self.verbose == True:
                    print("particle %d laminated! " % i)

        else:
            for i in particles:
                extforce3.addParticle(i, [])
                if self.verbose == True:
                    print("particle %d laminated! " % i)

        if r is None:
            try:
                r = self.sphericalConfinementRadius
            except:
                exit("No spherical confinement radius defined yet."\
                     "Apply spherical confinement first!")

        if self.verbose == True:
            print("Lamina attraction added with r = %d" % r)

        # assigning parameters of the force
        extforce3.addGlobalParameter("LAMaa", r * nm)
        extforce3.addGlobalParameter("LAMwidth", width * nm)
        extforce3.addGlobalParameter("LAMdepth", depth * self.kT)
        extforce3.addGlobalParameter("LAMtt", 0.01 * nm)


def energyMinimization(self, stepsPerIteration=100,
                       maxIterations=1000,
                       failNotConverged=True):
    """Runs system at smaller timestep and higher collision
    rate to resolve possible conflicts.

    Now we're moving towards local energy minimization,
    this is here for backwards compatibility.
    """

    print("Performing energy minimization")
    self._applyForces()
    oldName = self.name
    self.name = "minim"
    if (maxIterations is True) or (maxIterations is False):
        raise ValueError(
            "Please stop using the old notation and read the new energy minimization code")
    if (failNotConverged is not True) and (failNotConverged is not False):
        raise ValueError(
            "Please stop using the old notation and read the new energy minimization code")

    def_step = self.integrator.getStepSize()
    def_fric = self.integrator.getFriction()

    def minimizeDrop():
        drop = 10.
        for dummy in range(maxIterations):
            if drop < 1:
                drop = 1.
            if drop > 10000:
                raise RuntimeError("Timestep too low. Perhaps, "\
                                   "something is wrong!")

            self.integrator.setStepSize(def_step / float(drop))
            self.integrator.setFriction(def_fric * drop)
            # self.reinitialize()
            numAttempts = 5
            for attempt in range(numAttempts):
                a = self.doBlock(stepsPerIteration, increment=False,
                    reinitialize=False)
                # self.initVelocities()
                if a == False:
                    drop *= 2
                    print("Timestep decreased {0}".format(1. / drop))
                    self.initVelocities()
                    break
                if attempt == numAttempts - 1:
                    if drop == 1.:
                        return 0
                    drop /= 2
                    print("Timestep decreased by {0}".format(drop))
                    self.initVelocities()
        return -1

    if failNotConverged and (minimizeDrop() == -1):
        raise RuntimeError(
            "Reached maximum number of iterations and still not converged\n"\
            "increase maxIterations or set failNotConverged=False")
    self.name = oldName
    self.integrator.setFriction(def_fric)
    self.integrator.setStepSize(def_step)
    # self.reinitialize()
    print("Finished energy minimization")

    
    
def checkConnectivity(self, newcoords=None, maxBondSizeMultipler=10):
    ''' checks connectivity of all harmonic (& abslim) bonds
        can be passed to doBlock as a checkFunction, in which case it will also trigger re-initialization
        to modify the maximum bond size multipler, pass this function to doBlock as, 
        e.g. doBlock( 100,checkFunctions = [lambda x:a.checkConnectivity(x,6)])
    '''

    if not hasattr(self, "bondLengths"):
        raise ValueError('must use either harmonic or abs bonds to use checkConnectivty')

    if newcoords == None:
        newcoords = self.getData()
        printPositiveResult = True
    else: printPositiveResult = False

    # self.bondLengths is a list of lists (see above) [..., [int(i), int(j), float(distance), float(bondSize)], ...]
    bondArray = numpy.array(self.bondLengths)
    bondDists = numpy.sqrt(numpy.sum((newcoords[  numpy.array(bondArray[:, 0], dtype=int) ] - newcoords[ numpy.array(bondArray[:, 1], dtype=int) ]) ** 2, axis=1))
    bondDistsSorted = numpy.sort(bondDists)
    if (bondDists > (bondArray[:, 2] + maxBondSizeMultipler * bondArray[:, 3])).any():
        isConnected = False
        print("!! connectivity check failed !!")
        print("median bond size is ", numpy.median(bondDists))
        print("longest 10 bonds are", bondDistsSorted[-10:])

    else:
        isConnected = True
        if printPositiveResult:
            print("connectivity check passed.")
            print("median bond size is ", numpy.median(bondDists))
            print("longest 10 bonds are", bondDistsSorted[-10:])

    return isConnected

