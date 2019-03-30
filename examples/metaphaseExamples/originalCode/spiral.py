import numpy
from openmmlib import SimulationWithCrosslinks
from mirnylib.numutils import logbins
import sys
from polymerutils import create_spiral

if len(sys.argv) == 1:
    sys.argv.append("20")


def exampleOpenmm():
    """
    You need to have a OpenMM-compatible GPU and
    OpenMM installed to run this script.
    Otherwise you can switch to "reference" platform
    a.setup(platform = "reference")
    But this will be extremely slow...

    Installing OpenMM may be not easy too... but you can try
    """

    a = SimulationWithCrosslinks(timestep=150, thermostat=0.005)

    a.setup(platform="opencl", verbose=True)

    filename = "30_1000_20k_LJ_%s" % sys.argv[1]

    a.saveFolder(filename)  # folder where to save trajectory
    #a.load("trajectory2/start.dat")  #filename to load

    #g1 = polymerutils.load("/net/evolution/home/magus/trajectories/"\
    #"globule_creation/128000_SAW/crumpled%d.dat" % (1))
    #g1 = polymerutils.load("globule128")
    #g1 = g1[:100000]

    #a.load(create_spiral(3, 6.5, 120000), center=True)
    #a.load("30_only/block10.dat")
    a.load("../globules_expanded/crumpled%s.dat_expanded" % sys.argv[1])

    a.setLayout(mode="chain")  # default = chain
    #a.addGrosbergRepulsiveForce(trunc=2)
    a.addSoftLennardJonesForce(epsilon=0.42, trunc=2., cutoff=2.3)

    a.addHarmonicPolymerBonds(wiggleDist=0.1)
    a.addGrosbergStiffness(k=5)
    a.addConsecutiveRandomBonds(30, 0.3, 0.7, 0.5, distanceBetweenBonds=1)

    a.energyMinimization(stepsPerIteration=100, force=True)
    #a.addConsecutiveRandomBonds(2500, 1, 0.5, 0.5, distanceBetweenBonds=100)

    for _ in xrange(10):
        a.doBlock(400)
        a.save()
    a.addConsecutiveRandomBonds(1000, 0.5, 1, 0.5, distanceBetweenBonds=1)
    a.reinitialize()
    a.energyMinimization(stepsPerIteration=30, twoStage=True)
    for _ in xrange(10):
        for dummy in xrange(10):
            a.doBlock(500, increment=False)
        a.doBlock(500)
        a.save()
    a.addConsecutiveRandomBonds(20000, 0.5, 7, 0.5, distanceBetweenBonds=1)
    a.reinitialize()
    a.energyMinimization(stepsPerIteration=30, force=True)
    for _ in xrange(30):
        for dummy in xrange(10):
            a.doBlock(500, increment=False)
        a.doBlock(500)
        a.save()

    #a.addConsecutiveRandomBonds(5000, 1, 2.5, 0.3, distanceBetweenBonds=1,
    #                            verbose=True)
    a.show()
    a.printStats()

exampleOpenmm()
