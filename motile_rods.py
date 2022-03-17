import numpy
from openmmlib import Simulation, SimulationWithCrosslinks

from mirnylib.numutils import logbins
import polymerutils
import tempfile
import os
from mirnylib.plotting import mat_img
from simtk import openmm, unit
from scipy.stats.stats import spearmanr
from mirnylib.systemutils import setExceptionHook


class simulationOfMotileRods(Simulation):
    def getScaledData(self):
        data = Simulation.getScaledData(self)
        #data[:,2] += 2
        return data

    def createIntegrator(self, viscosity=1.3):
        integrator = openmm.CustomIntegrator(self.timestep)
        #assert isinstance(self.mm,openmm.CustomIntegrator)
        integrator.addUpdateContextState()
        v = unit.sqrt(self.kT * self.mass)
        f = self.kT / self.conlen
        gamma = viscosity * f / v
        integrator.addGlobalVariable("gamma", gamma)
        integrator.addPerDofVariable("s1", 0.)
        integrator.addComputePerDof("s1", "f2")

        integrator.addComputePerDof("v", "v+0.5*dt* (f0 - gamma * v)/m")
        integrator.addComputePerDof("v", "v+step(s1) * (0.5*dt*(f1 ) /m)")

        integrator.addComputePerDof("x", "x+dt*v")
        #integrator.addConstrainVelocities()

        integrator.addComputePerDof("v", "v+0.5*dt*(f0 - gamma * v)/m")
        integrator.addComputePerDof(
            "v", "v+step(s1) * 0.5*dt*(f1 - gamma * v) /m")

        #integrator.addConstrainPositions()

        self.integrator = integrator

    def addForcesOfMotileRods(self, startEndPairs, strength=1):
        fakeBond = openmm.CustomBondForce("-MOTk * r")
        fakeBond.setForceGroup(1)
        fakeBond.addGlobalParameter("MOTk", strength * self.kT / self.conlen)
        for i in startEndPairs:
            fakeBond.addBond(int(i[0]), int(i[1] - 1), [])

        #fakeBond = self.mm.CustomExternalForce("fx * x")
        #fakeBond.setForceGroup(1)
        #fakeBond.addGlobalParameter("fx",float(strength) * self.kT / self.conlen)
        #for i in self.chains:
        #    fakeBond.addParticle(i[1]-1,[])
            #fakeBond.addParticle(i[0],[])

        fakeExternal = self.mm.CustomExternalForce("fx * (x+y+z)")
        fakeExternal.setForceGroup(2)
        fakeExternal.addPerParticleParameter("fx")
        for i in startEndPairs:
            fakeExternal.addParticle(i[0], [0.01 * self.kT / self.conlen])
            fakeExternal.addParticle(i[1] - 1, [-0.01 * self.kT / self.conlen])

        self.forceDict["fakeBond1"] = fakeBond
        self.forceDict["fakeExternal2"] = fakeExternal

    def show(self, chains=None):
        """shows system in rasmol by drawing spheres
        draws 4 spheres in between any two points (5 * N spheres total)
        """
        if chains is None:
            chains = self.chains
        #if you want to change positions of the spheres along each segment, change these numbers
        #e.g. [0,.1, .2 ...  .9] will draw 10 spheres, and this will look better

        data = self.getScaledData()()
        if len(data[0]) != 3:
            data = numpy.transpose(data)
        if len(data[0]) != 3:
            print "wrong data!"
            return
        #determining the 95 percentile distance between particles,
        rascript = tempfile.NamedTemporaryFile()  # writing the rasmol script. Spacefill controls radius of the sphere.
        rascript.write("""wireframe off
        color temperature
        spacefill 100
        background white
        """)
        rascript.flush()

        #creating the array, linearly chanhing from -225 to 225, to serve as an array of colors
        colors = numpy.array([int((j * 450.) / (len(data))) -
                              225 for j in xrange(len(data))])

        #creating spheres along the trajectory
        newData = numpy.zeros((len(data), 4))
        newData[:, :3] = data
        newData[:, 3] = colors

        newData2 = []

        for chain in chains:
            beg, end = chain
            for i in xrange(beg, end - 1):
                for mult in [(0., 1.), (0.2, 0.8), (0.4, 0.6), (0.6, 0.4), (0.8, 0.2)]:
                    newData2.append(newData[i] * mult[
                        0] + newData[i + 1] * mult[1])
        newData = newData2

        towrite = tempfile.NamedTemporaryFile()
        towrite.write("%d\n\n" % (len(newData)))  # number of atoms and a blank line after is a requirement of rasmol

        for i in newData:
            towrite.write("CA\t%lf\t%lf\t%lf\t%d\n" % tuple(i))
        towrite.flush()

        if os.name == "posix":  # if linux
            os.system("rasmol -xyz %s -script %s" % (
                towrite.name, rascript.name))
        else:  # if windows
            os.system("C:/RasWin/raswin.exe -xyz %s -script %s" % (towrite.name, rascript.name))  # For windows you might need to change the place where your rasmol file is
        rascript.close()
        towrite.close()


def straightRodsAgain():

    fine = True

    a = simulationOfMotileRods(
        timestep=40, thermostat=.05, velocityReinitialize=False)

    a.setup(platform="OpenCL", verbose=True, PBC=False,
            PBCbox=[160, 160, 50])
    a.saveFolder("trajectory")  # folder where to save trajectory

    if fine == True:
        a.createIntegrator(viscosity=.5)

    g1 = polymerutils.load("disk6000")
    g1 = g1[:6000]
    a.load(g1)

    a.setLayout(mode="chain", Nchains=500)
        #default = chain

    a.addGrosbergRepulsiveForce(trunc=6, radiusMult=2.)
    a.addGrosbergStiffness(k=2500)

    startEndPairs = [(i[0], i[1] - 1) for i in a.chains]
    a.addHarmonicPolymerBonds(0.015)

    for pair in startEndPairs:
        a.addBond(pair[0], pair[1], 0.03, 0.4 + float(pair[1] -
                                                      pair[0]), "harmonic")

    rcyl = 90.
    #if fine == True: rcyl  = 99999.

    a.addCylindricalConfinement(r=rcyl, bottom=5., top=5.01, k=5.)
    #a.excludeSphere(10,(35.,0.,5.))
    #a.tetherParticles([1])

    #a.addSphericalConfinement(density = 0.005, k = 1)
    if fine == True:
        a.addForcesOfMotileRods(startEndPairs, strength=.7)
    #a.addGrosbergStiffness(k = 40)
    if fine == False:
        a.energyMinimization(steps=3000, twoStage=True)
    #a.show()
    a.doBlock(0, increment=False)

    for _ in xrange(3000):

        a.doBlock(100, num=100)

        #a.save()
    #    a.show()
        a.save(mode="vtf", filename="traj2.vtf")

    exit()


straightRodsAgain()


def exampleOpenmm(x):
    """
    You need to have a OpenMM-compatible GPU and OpenMM installed to run this script.
    Otherwise you can switch to "reference" platform
    a.setup(platform = "reference")
    But this will be extremely slow...

    Installing OpenMM may be not easy too... but you can try
    """

    fine = True

    a = simulationOfMotileRods(
        timestep=40, thermostat=.05, velocityReinitialize=False)

    a.setup(platform="OpenCL", verbose=True, PBC=True, PBCbox=[
        100, 100, 100])
    #thermostat = a.mm.AndersenThermostat(a.temperature, a.collisionRate)
    #if fine == True: a.forceDict["AndersenThermostat"] = thermostat

    a.saveFolder("trajectory")  # folder where to save trajectory

    if fine == True:
        a.createIntegrator(viscosity=.5)

    #a.integrator = a.mm.VerletIntegrator(a.timestep)
    #a.load("trajectory2/start.dat")  #filename to load

    g1 = polymerutils.load("cylindricalStart")
    g1 = g1[:6400]

    a.load(g1)

    a.setLayout(chains=[])  # default = chain
    a.addGrosbergRepulsiveForce(trunc=10)

    def addRodBonds():
        rodLength = 10
        monomersPerRod = 3 * rodLength + 2
        wiggle = .03
        startEndPairs = []

        def addLinearBonds(start, end):
            for i in xrange(start, end - 1):
                a.addBond(i, i + 1, bondWiggleDistance=wiggle,
                          distance=1., bondType="harmonic")

        def addBetweenRodBonds(start1, end1, start2, end2):
            for i in xrange(end1 - start1):
                a.addBond(start1 + i, start2 + i, bondWiggleDistance=wiggle,
                          distance=1., bondType="harmonic")

        for rodNum in xrange(200):
            start = rodNum * monomersPerRod

            def addBond(i, j):
                a.addBond(start + i, start + j, bondWiggleDistance=wiggle,
                          distance=1., bondType="harmonic")
            startEndPairs.append((start, start + monomersPerRod - 1))
            a.addBond(start, start + monomersPerRod - 1, bondWiggleDistance=wiggle, distance=10.3, bondType="harmonic")
            addBond(0, 1)
            addBond(0, 2)
            addBond(0, 3)
            addBond(1, 2)
            addBond(2, 3)
            addBond(1, 3)
            for layer in xrange(1, rodLength):
                cur1, cur2, cur3 = layer * 3 + 1, layer * 3 + 2, layer * 3 + 3
                prev1, prev2, prev3 = layer * 3 - 2, layer * 3 - 1, layer * 3
                addBond(cur1, cur2)
                addBond(cur2, cur3)
                addBond(cur3, cur1)
                addBond(cur1, prev2)
                addBond(cur1, prev3)
                addBond(cur2, prev1)
                addBond(cur2, prev3)
                addBond(cur3, prev1)
                addBond(cur3, prev2)
            end = rodLength * 3 + 1
            addBond(end, end - 1)
            addBond(end, end - 2)
            addBond(end, end - 3)

        return startEndPairs
    startEndPairs = addRodBonds()
    a.addCylindricalConfinement(r=1000, bottom=True, top=2., k=15.)

    #a.addSphericalConfinement(density = 0.005, k = 1)
    if fine == True:
        a.addForcesOfMotileRods(startEndPairs, strength=1.7)
    #a.addGrosbergStiffness(k = 40)
    if fine == False:
        a.energyMinimization(steps=1000, twoStage=True)
    #a.show()
    #a.doBlock(1)
    for _ in xrange(1400):
        a.doBlock(60, num=60)
    #    a.show()
        #a.save()
        a.save(mode="vtf", filename="traj.vtf")

    exit()

    for i in startEndPairs:
        print a.dist(*i)

    pos = a.getData()
    a.printStats()
    vel = a.velocs
    difSelect = []
    velSelect = []
    posSelect = []
    for i in startEndPairs:
        difSelect.append(pos[i[1]] - pos[i[0]])
        velSelect.append(numpy.mean(vel[i[0]:i[1] + 1], axis=0))
        posSelect.append(numpy.mean(pos[i[0]:i[1] + 1], axis=0))
    difSelect1 = numpy.array(difSelect)
    velSelect1 = numpy.array(velSelect)
    posSelect1 = numpy.array(posSelect)

    a.doBlock(500)

    pos = a.getData()
    a.printStats()
    vel = a.velocs
    difSelect = []
    velSelect = []
    posSelect = []
    for i in startEndPairs:
        difSelect.append(pos[i[1]] - pos[i[0]])
        velSelect.append(numpy.mean(vel[i[0]:i[1] + 1], axis=0))
        posSelect.append(numpy.mean(pos[i[0]:i[1] + 1], axis=0))
    difSelect2 = numpy.array(difSelect)
    velSelect2 = numpy.array(velSelect)
    posSelect2 = numpy.array(posSelect)

    setExceptionHook()
    corr = spearmanr

    def sc(a, b):
        import matplotlib.pyplot as plt
        plt.scatter(a, b)
        plt.title("spearman r = %lf " % corr(a, b)[0])
        plt.show()
    sc(difSelect1[:, 0], velSelect1[:, 0])
    0 / 0

    import matplotlib.pyplot as plt
    plt.scatter(difSelect[:, 2], velSelect[:, 2])
    print "Spearman r =", spearmanr(difSelect[:, 2], velSelect[:, 2])
    plt.show()

    a.printStats()
    #a.save(filename = "globule32")
    a.show(startEndPairs)


for x in xrange(11, 12):
    exampleOpenmm(x)
exit()


def averageContactMap():

    import contactmaps
    filenames = ["trajectory%d/block%d.dat" % (i, j) for i in xrange(
        10, 11) for j in xrange(100, 200)]
    mymap = contactmaps.averageContactMap(filenames, resolution=400, cutoff=2.1, usePureMap=False, n=4, loadFunction=polymerutils.load, exceptionsToIgnore=None)
    mat_img(numpy.log(mymap + 1))
    0 / 0


averageContactMap()
