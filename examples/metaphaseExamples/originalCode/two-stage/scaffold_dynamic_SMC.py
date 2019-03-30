import matplotlib
import matplotlib.pyplot as plt
import os
import numpy
import shutil
from mirnylib.plotting import mat_img
np = numpy
from openmmlib import SimulationWithCrosslinks
from mirnylib.numutils import logbins, fillDiagonal
import polymerutils
from polymerutils import create_spiral, load
import cPickle
import random
import sys
from mirnylib.systemutils import setExceptionHook
setExceptionHook()


try:
    firstLoopLength = int(sys.argv[1])
except:
    firstLoopLength = 100

try:
    simNum = int(sys.argv[2])
except:
    simNum = 16

try:
    platform = sys.argv[3]
except:
    platform = "cuda"
try:
    GPU = sys.argv[4]
except:
    GPU = "1"

try:
    intLen = int(sys.argv[5])
except:
    intLen = 30

L = firstLoopLength
num = simNum


def createGaussianLoops(avLoopSize, N, minLoopSize="auto", maxLoopSize="auto"):
    if minLoopSize == "auto":
        minLoopSize = avLoopSize * 0.2
    if maxLoopSize == "auto":
        maxLoopSize = avLoopSize * 3
    numLoops = 5 * (N / avLoopSize)
    lens = np.random.randn(numLoops) * avLoopSize
    lens = lens[(lens > minLoopSize) * (lens < maxLoopSize)]
    lens = np.array(lens, dtype=int)
    lens = np.cumsum(lens)
    lens = lens[lens < N - avLoopSize / 2]
    lens = np.r_[0, lens , N - 1]
    return lens


pancakeWidth = 30
pancakeSize = 15000
foldername = "L_{0}_num_{1}_len_{2}".format(L, num, intLen)
if not os.path.exists(foldername): os.mkdir(foldername)
shutil.copy(sys.argv[0], foldername)
data = load("globule32")[:30000]
polymerutils.save(data, os.path.join(foldername, "block0.dat"))
N = len(data)
pancakeNum = N / float(pancakeSize)

coreParticles = numpy.nonzero(numpy.random.random(N) < (1. / L))[0]



smcStep = 5
stepsPerSmc = 6000
bondBlocks = 70
minBondLen = 5
maxBondLen = 20
bondNum = 3
secondSmcGap = 2
secondSmcStep = 60
secondSmcIncrementStep = 4
step = 0

cPickle.dump(coreParticles, open(os.path.join(foldername, "coreParticles"), "w"))
#step = 40
#data = polymerutils.load("L_80_num_11_final_protocol/block40.dat")
#coreParticles = cPickle.load(open(os.path.join(foldername, "coreParticles")))



regions = np.array(zip(coreParticles[:-1], coreParticles[1:]))
regionLens = regions[:, 1] - regions[:, 0]
maxLen = regionLens.max()
regionMids = np.ceil(np.mean(regions, axis=1))

smcBlocks = int(np.ceil(maxLen / (1.*smcStep)))

finalLength = (len(data) / (1. *pancakeSize)) * pancakeWidth
ratio = len(regions) / finalLength



coreParticles2 = createGaussianLoops(secondSmcStep, N)
coreParticles2 = np.sort(np.unique(list(coreParticles) + list(coreParticles2)))



regions2 = zip(coreParticles2[:-1], coreParticles2[1:])
regions2 = np.array(regions2)
regionLens2 = regions2[:, 1] - regions2[:, 0]
mask = regionLens2 > 3 * secondSmcGap
regions2 = regions2[mask]
regionLens2 = regionLens2[mask]
regionMids2 = np.ceil(np.mean(regions2, axis=1))
compBonds = []
for i in xrange(len(coreParticles)):
    for j in xrange(i + 1, min(len(coreParticles) - 1, i + intLen), 3):
        compBonds.append((i, j))



compBonds = list(set(compBonds))

def initSimulation(data, force="attr", power=1):
    a = SimulationWithCrosslinks(timestep=70, thermostat=0.001)
    a.setup(platform=platform, verbose=True, GPU=GPU)
    a.saveFolder(foldername)  # folder where to save trajectory
    a.load(data)
    a.setLayout(mode="chain")  # default = chain
    a.addHarmonicPolymerBonds(wiggleDist=0.1)
    a.addGrosbergStiffness(k=4)
    if force == "attr":
        a.addSoftLennardJonesForce(epsilon=.15 + 0.45 * power, trunc=3, cutoff=2.3)
    else:
        a.addGrosbergRepulsiveForce(trunc=1.2)
    return a

"""
for i in xrange(smcBlocks + 20):
    a = initSimulation(data, force="rep")
    assert isinstance(a, SimulationWithCrosslinks)
    bonds = []
    smcLen = i * smcStep
    for region, regionLen, regionMid in zip(regions, regionLens, regionMids):
        if smcLen > regionLen:
            bonds.append((region[1], region[0]))
        else:
            bonds.append((regionMid - smcLen / 2, regionMid + smcLen / 2))
    for bond in bonds:
        a.addBond(bond[0], bond[1],
                  bondWiggleDistance=0.08, distance=0.4,
                  bondType="abs")
    a.step = step
    a.doBlock(20, increment=False)
    a.doBlock(stepsPerSmc)
    a.save()
    step = a.step

    data = a.getData()
"""

for i in xrange(150):

    a = initSimulation(data, force="rep")
    assert isinstance(a, SimulationWithCrosslinks)


    for bond in regions:
        a.addBond(bond[0], bond[1],
                  bondWiggleDistance=0.25 * (200. / (i + 0.5)), distance=0.5,
                  bondType="harmonic")
    print "wiggle distance", 0.25 * (100. / (i + 0.5))
    a.step = step
    a.doBlock(20, increment=False)
    a.doBlock(2000)
    a.save()
    step = a.step
    data = a.getData()



points = list(numpy.linspace(10, 1, 200)) + [1] * 500

for j, radius in enumerate(points):

    a = initSimulation(data, force="attr", power=min(j / 200., 1))
    assert isinstance(a, SimulationWithCrosslinks)
    a._initAbsDistanceLimitation()

    for st, end in compBonds:
        a.addBond(coreParticles[st], coreParticles[end],
                  bondWiggleDistance=0.07,
                  distance=3 * radius,
                  bondType="abslim")

    #for st, end in zip(coreParticles[:-1], coreParticles[1:]):
    #    for j in xrange(st + 5, end - 5, 5):
    #        a.addBond(st, j, bondWiggleDistance=0.4, distance=13, bondType="abslim")



    for bond in regions:
        a.addBond(bond[0], bond[1],
                  bondWiggleDistance=0.2, distance=1,
                  bondType="abs")
    a.step = step
    a.doBlock(20, increment=False)
    a.doBlock(3000)
    a.save()
    step = a.step
    data = a.getData()

