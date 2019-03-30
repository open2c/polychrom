import sys
import os
import numpy
np = numpy
from openmmlib import Simulation, SimulationWithCrosslinks
from mirnylib.numutils import logbins
import polymerutils

os.system("renice -n %d %d" % (10, os.getpid()))
#renice your process to make everything run nicely

if len(sys.argv) < 4:
    print """Usage: python scaffoldExample.py loopLength replica index [platform] [GPU]

     loop length measured in monomers

     replica is appended to the filename

     index: 1=Consecutive; scaffold. 2=Consecutive, no scaffold
     index: 3=Non-consecutive, scaffold. 4=non-consecutive, no scaffold

     platform should be cuda or opencl dependent on your OpenMM setup

     GPU should be selected based on nvidia-smi output, especially if
     you have several GPUs

     a 128000-monomer file with initial conformation should be provided
     in a file with a filename globule128

     Format of the file should be loadable by  polymerutils.load
     (it can be created using polymerutils.save)

     Please refer to openmmlib-polymer manual found here:
     http://mimakaev.bitbucket.org/
     It has some descriptions of input/output formats,
     describes forces and output of the program

     example run would be python scaffoldExample.py 200 1 1
     """


def exampleOpenmm(x, y, z):
    """
    You need to have a OpenMM-compatible GPU and OpenMM installed to run this script.
    Otherwise you can switch to "reference" platform
    a.setup(platform = "reference")
    But this will be extremely slow...

    Installing OpenMM may require help of professional system administrator
    """

    a = SimulationWithCrosslinks(timestep=140, thermostat=0.01)

    platform = "cuda"
    if len(sys.argv) > 4:
        GPU = sys.argv[5]
        platform = sys.argv[4]
    else:
        GPU = "0"
        platform = "cuda"

    # Initializing the openmm-polymer setup
    a.setup(platform=platform, verbose=True, GPU=GPU)
    name = {1:"ConsScaff", 2:"ConsNoScaff", 3:"RandScaff", 4:"RandNoScaff"}[z]
    foldername = "{2}{0}_{1}".format(x, y, name)
    a.saveFolder(foldername)  # folder where to save trajectory

    a.load("globule128")  # filename to load
    a.setLayout(mode="chain")  # default = chain
    a.addGrosbergRepulsiveForce(trunc=1.)

    # Calculating parameters of confining potentials
    pancakeWidth = 60
    pancakeSize = 70000
    pancakeNum = a.N / float(pancakeSize)
    a.addCylindricalConfinement(
        r=28, bottom=True, top=pancakeWidth * pancakeNum, k=0.3)

    #finding positions of loop anchors
    import  cPickle
    L = int(sys.argv[1])
    coreParticles = numpy.nonzero(numpy.random.random(a.N) < (1. / L))[0]

    # Below alternative formulas for positions fo core particles with different distribution

    #coreParticles = numpy.random.randint(int(0.5 * L), int(1.5 * L), 10000)
    #coreParticles = numpy.cumsum(coreParticles)
    #coreParticles = coreParticles[coreParticles < a.N]

    #Dumping core particles to a file
    if z in [1, 2]:
        cPickle.dump(coreParticles,
                     open(os.path.join(foldername, "coreParticles"), "w"))
        print "Loop anchors dumped to coreParticles file using cPickle"
    else:
        # Processing coreParticles for random loops
        coreStarts = coreParticles[:-1]
        coreEnds = coreParticles[1:]
        lens = coreEnds - coreStarts
        coreStarts = [np.random.randint(0, a.N - i - 2) for i in lens]
        coreStarts = np.array(coreStarts)
        coreEnds = lens + coreStarts
        cores = zip(coreStarts, coreEnds)
        print cores
        cPickle.dump(cores,
                     open(os.path.join(foldername, "coreParticles"), "w"))

    #Force of the scaffold attachment
    extForce = a.mm.CustomExternalForce("cusK * sqrt(cusa^2 + 10*x^2 + 10*y^2 ) ")
    extForce.addGlobalParameter("cusK", 1 * a.kT / a.conlen)
    extForce.addGlobalParameter("cusa", a.conlen * 0.1)
    #Following most of Openmm-polymer forces, we add an extra regularizing constant

    #Now connecting core particles
    #Remove this and scaffold to make linear organization model.
    if z in [1, 2]:
        for i in xrange(len(coreParticles) - 1):
            beg, end = coreParticles[i], coreParticles[i + 1]
            a.addBond(beg, end, 1., 1., "harmonic")
    else:
        for i, j in cores:
                a.addBond(i, j, 1., 1., "harmonic")

    #Implementing scaffold attahcment if needed
    if z in [1, 3]:
        for i in coreParticles:
            extForce.addParticle(int(i), [])
        a.forceDict["scaffoldAttr"] = extForce

    #Force of linear oragnization
    a.fixParticlesZCoordinate(
        particles=range(0, a.N), zCoordinates=(0, pancakeWidth * pancakeNum),
        k=0.1, mode="quadratic", gap=pancakeWidth)

    a.addHarmonicPolymerBonds(wiggleDist=0.1)
    a.addGrosbergStiffness(k=5)
    #a.energyMinimization(stepsPerIteration=300)
    a.localEnergyMinimization()

    for _ in xrange(1000):
        a.doBlock(20000)
        a.save()
    a.printStats()


exampleOpenmm(sys.argv[1], sys.argv[2], int(sys.argv[3]))

