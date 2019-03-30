import numpy
from openmmlib import Simulation, SimulationWithCrosslinks
from mirnylib.numutils import logbins
import polymerutils
from mirnylib.plotting import mat_img



def exampleOpenmm(x):
    """
    You need to have a OpenMM-compatible GPU and OpenMM installed to run this script.
    Otherwise you can switch to "reference" platform
    a.setup(platform = "reference")
    But this will be extremely slow...

    Installing OpenMM may be not easy too... but you can try
    """

    a = SimulationWithCrosslinks(timestep=80, thermostat=0.005)

    a.setup(platform="OpenCL", verbose=True)
    a.saveFolder("sizes_12_200_2500")  # folder where to save trajectory
    #a.load("trajectory2/start.dat")  #filename to load

    g1 = polymerutils.load("/net/evolution/home/magus/trajectories/globule_creation/128000_SAW/crumpled%d.dat" % (x + 1))

    #Comment out two lines below for a full-length chain
    print "using shorter chain for protocol refinement"
    g1 = g1[:40000]  # for faster developement of the protocol


    a.load(g1)

    a.setLayout(mode="chain")
    a.addHarmonicPolymerBonds(wiggleDist=0.1)
    a.addGrosbergRepulsiveForce(trunc=1.4)
    a.addGrosbergStiffness(k=5)

    pancakeWidth = 20
    pancakeSize = 20000
    pancakeNum = a.N / float(pancakeSize)
    print pancakeNum

    a.addCylindricalConfinement(
        r=34, bottom=0, top=pancakeWidth * pancakeNum, k=0.7)
    a.fixParticlesZCoordinate(particles=range(0, a.N, 10), zCoordinates=(0, pancakeWidth * pancakeNum), k=0.03, mode="quadratic", gap=pancakeWidth)

    #a.addRgLimitations(num = 13,volumeParameter = 0.05,relativeSmeer = 0.60,pancakeLength = 35000)

    #a.addConsecutiveRandomBonds(12,0.3,0.5,0.5,distanceBetweenBonds = 2)
    #a.addConsecutiveRandomBonds(200,0.3,0.5,0.5,distanceBetweenBonds = 5)
    a.addConsecutiveRandomBonds(12, 0.3, 0.5, 0.5, distanceBetweenBonds=2)
    a.addConsecutiveRandomBonds(200, 0.3, 0.5, 0.5, distanceBetweenBonds=6)
    a.addConsecutiveRandomBonds(2500, 1, 0.5, 0.5, distanceBetweenBonds=100)

    #coreParticles = numpy.nonzero(numpy.random.random(a.N) < 0.01)[0]
    #a.addAttractionToTheCore(.5,6, coreParticles)
    a.energyMinimization(stepsPerIteration=30, force=True)

    for _ in xrange(200):
        a.doBlock(10000)
        a.save()
    a.printStats()


for x in xrange(10):
    exampleOpenmm(x)
exit()

