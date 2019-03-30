import os
from openmmlib.polymerutils import grow_rw
import numpy as np
from openmmlib.openmmlib import Simulation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mirnylib.numutils import logbins, create_regions, continuousRegions
from openmmlib import polymerutils
from mirnylib.plotting import mat_img
from openmmlib.polymerutils import create_spiral, load
import pickle
from mirnylib.systemutils import setExceptionHook
from brushManaging import makeBondsForBrush
from openmmlib import pymol_show
setExceptionHook()
import sys

if len(sys.argv) < 2:
    print("Set GPU number as first command line argument")
    print("Default value of 0 used")
    sys.argv = ["dummy", "0"]




def exampleOpenmm(supercoilCompaction=4, plectonemeLength=10, plectonemeGap=0, stifness=2, ind=0, domainNum=30, domainLen=5):
    """
    A method that performs one simulation.
    """

    a = Simulation(timestep=100, thermostat=0.01)

    a.setup(platform="cuda",  GPU=sys.argv[1])

    foldername = "newSweep_lambda{0}_L{1}_gap{2}_stiff{3}_ind{4}".format(supercoilCompaction, plectonemeLength,
                                                         plectonemeGap, stifness, ind)
    a.saveFolder(foldername)  # folder where to save trajectory


    """
    Defining all parameters of the simulation
    """
    genomeLengthBp = 4000000
    lengthNm = 1500
    diamNm = 450
    supercoilBpPerNm = 4.5 * supercoilCompaction

    supercoilDiameterNm = 3.45 * np.sqrt(supercoilBpPerNm)
    supercoilBpPerBall = supercoilBpPerNm * supercoilDiameterNm * 2  # 2 strands
    print(supercoilBpPerBall)
    print(supercoilDiameterNm)

    ballNumber = 2 * int(0.5 * (genomeLengthBp / supercoilBpPerBall))
    radiusMon = 0.5 * (diamNm / float(supercoilDiameterNm))
    halfLengthMon = 0.5 * (lengthNm / float(supercoilDiameterNm))

    N = ballNumber
    print(N)
    avLength = plectonemeLength

    # Growing a circular polymer
    bacteria = grow_rw(N, int(halfLengthMon * 2), "line")
    bacteria = bacteria - np.mean(bacteria, axis=0)
    a.load(bacteria)  # filename to load
    print(bacteria[0])

    a.tetherParticles([0, N - 1])

    a.addCylindricalConfinement(r=radiusMon, bottom=-halfLengthMon, top=halfLengthMon, k=1.5)


    BD = makeBondsForBrush(chainLength=a.N)

    # Coordinates of top highly expressed genes
    geneCoord = [1162773, 3509071, 1180887, 543099, 1953250, 2522439, 3328524, 1503879, 900483, 242693, 3677144, 3931680, 3677704, 3762707, 3480870, 3829656, 1424678, 901855, 1439056, 3678537]
    particleCoord = [(i / 4042929.) * a.N for i in geneCoord]
    particleCoord = [int(i) for i in particleCoord]
    particleCoord = sorted(particleCoord)

    # Below making sure that if two PFRs overlap, we just create a PFR which is twice longer
    gapShift = 7
    gaps = []
    gapStart = particleCoord[0]
    gapEnd = particleCoord[0]
    for i in particleCoord:
        if i > gapEnd:
            gaps.append((gapStart, gapEnd))
            gapStart = i
            gapEnd = i + gapShift
        else:
            gapEnd += gapShift
    gaps.append((gapStart, gapEnd))

    # Adding PFRs at highly expressed genes
    M = domainNum
    particles = []
    for st, end in gaps:
        BD.addGap(st, end)

    # Making bonds
    BD.addBristles(3, plectonemeLength, 0, plectonemeGap)
    BD.sortSegments()
    print(BD.segments)
    BD.createBonds()
    BD.checkConnectivity()

    BD.save(os.path.join(a.folder, "chains"))
    a.setChains(chains=BD.getChains())


    a._initHarmonicBondForce()
    for i in BD.bonds:
        a.addBond(i[0], i[1], bondWiggleDistance=0.15, bondType="Harmonic")


    a.addGrosbergStiffness(k=stifness)
    a.addGrosbergRepulsiveForce(trunc=1.)
    # a.addSoftLennardJonesForce(epsilon=0.46, trunc=2.5, cutoff=2.3)
    a.save(os.path.join(a.folder, "start"))
    a.localEnergyMinimization()
    a.save()

    for step in range(500):
        a.doBlock(50000)
        a.save()
    a.printStats()


stifnesses = [6]
gaps = [2]
lengths = [ 35]
compactions = [3.5]
inds = list(range(200))




combinations = [(i, j, k, l, m) for i in compactions for j in lengths for k in gaps for l in stifnesses for m in inds]
import random
random.shuffle(combinations)
print(combinations)
for i in combinations:

    foldername = "newSweep_lambda{0}_L{1}_gap{2}_stiff{3}_ind{4}".format(*i)
    print(foldername)
    if os.path.exists(foldername):
        continue
    exampleOpenmm(*i)



