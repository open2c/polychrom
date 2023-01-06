# %load flagshipNormLifetime.py
import matplotlib

matplotlib.use("Agg")

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ctypes
from mirnylib.plotting import nicePlot
import multiprocessing as mp
import pyximport

from openmmlib import polymerutils

from openmmlib.polymerutils import scanBlocks

pyximport.install()
from mirnylib.h5dict import h5dict
from mirnylib.systemutils import fmap, setExceptionHook
from mirnylib.numutils import coarsegrain, completeIC
from mirnylib.numutils import zoomArray
from contextlib import closing
import sys
from smcTranslocator import smcTranslocatorDirectional

filename = "/net/levsha/share/nezar/ctcf_sites/GM12878.ctcf_narrowPeak.loj.encodeMotif.rad21.txt"


SEPARATION = 200
LIFETIME = int(sys.argv[2])


def indexing(smaller, larger, M):
    return larger + smaller * (M - 1) - smaller * (smaller - 1) / 2


def tonumpyarray(mp_arr):
    return np.frombuffer(mp_arr.get_obj())  # .reshape((N,N))


def init(*args):
    global sharedArrays
    sharedArrays = args


def worker(filenames):
    N = int(np.sqrt(sum(map(len, sharedArrays))))
    chunks = range()


def chunk(mylist, chunksize):
    N = len(mylist)
    chunks = list(range(0, N, chunksize)) + [N]
    return [mylist[i:j] for i, j in zip(chunks[:-1], chunks[1:])]


def averageContacts(contactFunction, inValues, N, **kwargs):

    arrayDtype = kwargs.get("arrayDtype", ctypes.c_int32)

    nproc = min(kwargs.get("nproc", 4), len(filenames))
    blockSize = max(min(kwargs.get("blockSize", 50), len(filenames) // (3 * nproc)), 1)

    finalSize = N * (N + 1) // 2
    boundaries = np.linspace(0, finalSize, bucketNum + 1)
    chunks = zip(boundaries[:-1], boundaries[1:])
    sharedArrays_ = [mp.Array(arrayDtype, j - i) for j, i in chunks]

    filenameChunks = [filenames[i::nproc] for i in range(nproc)]

    with closing(
        mp.Pool(processes=nproc, initializer=init, initargs=sharedARrays_)
    ) as p:
        p.map(worker, filenameChunks)


df = pd.read_csv(filename, sep="\t")
df = df.loc[(~pd.isnull(df["summitDist"]))]

mychr = 14
df = df.loc[df["chrom"] == "chr{0}".format(mychr)]
start = df["start"].values
end = df["end"].values
strand = df["summitDist"].values > 0
strength = df["fc"]

mid_1k = (start + end) // 1200
M = mid_1k.max() + 1


print(len(mid_1k[strand]))
print(len(mid_1k[~strand]))
forw = np.bincount(mid_1k[strand], weights=(strength[strand] / 20), minlength=M)
rev = np.bincount(mid_1k[~strand], weights=(strength[~strand] / 20), minlength=M)
print(len(forw[forw > 0]))
print(len(rev[rev > 0]))


low = 60000
high = 75000
lowMon = low * 1000 // 600
highMon = high * 1000 // 600
forw = forw[lowMon:highMon]
rev = rev[lowMon:highMon]

mu = float(sys.argv[3])


def logistic(x, mu=3):
    x[x == 0] = -99999999
    return 1 / (1 + np.exp(-(x - mu)))


forw = logistic(forw, mu)
rev = logistic(rev, mu)

# plt.plot(forw)
# plt.show()
# exit()


"""
plt.plot(forw)
plt.plot(rev)
plt.show()
exit()
"""

N = len(forw)


def initModel(i=0):
    birthArray = np.zeros(N, dtype=np.double) + 0.1

    deathArray = np.zeros(N, dtype=np.double) + 1.0 / LIFETIME

    stallArrayLeft = forw
    stallArrayRight = rev

    stallDeathArray = np.zeros(N, dtype=np.double) + 1 / LIFETIME

    pauseArray = np.zeros(N, dtype=np.double)

    smcNum = N // SEPARATION
    myDeathArray = deathArray + (1.0 / (LIFETIME * LIFETIME)) * i
    SMCTran = smcTranslocatorDirectional(
        birthArray,
        myDeathArray,
        stallArrayLeft,
        stallArrayRight,
        pauseArray,
        stallDeathArray,
        smcNum,
    )
    return SMCTran


def displayHeatmap():
    plt.figure(figsize=(5, 5))
    shared_arr = mp.Array(ctypes.c_double, N**2)
    arr = tonumpyarray(shared_arr)
    arr.shape = (N, N)

    def doSim(i):
        nparr = tonumpyarray(shared_arr)
        SMCTran = initModel(i)

        for j in range(1):
            SMC = []
            N1 = 10000
            for k in range(np.random.randint(N1 // 2, N1)):
                SMCTran.steps(150)
                SMC.append(SMCTran.getSMCs())
            SMC = np.concatenate(SMC, axis=1)
            SMC1D = SMC[0] * N + SMC[1]
            position, counts = np.unique(SMC1D, return_counts=True)

            with shared_arr.get_lock():
                nparr[position] += counts
        print("Finished!")

        return None

    setExceptionHook()

    low20 = low // 10
    high20 = high // 10
    mydict = h5dict(
        "/home/magus/HiC2011/Erez2014/hg19/GM12878_inSitu-all-combined-10k_HighRes.byChr",
        "r",
    )

    hicdata = mydict.get_dataset("13 13")[low20:high20, low20:high20]
    hicdata = completeIC(hicdata)
    curshape = hicdata.shape
    newshape = (1000 * (high - low)) // (600 * 20)
    print(hicdata.shape, newshape)
    hicdata = zoomArray(hicdata, (newshape, newshape))
    hicdata = np.clip(hicdata, 0, np.percentile(hicdata, 99.99))
    hicdata /= np.mean(np.sum(hicdata, axis=1))

    fmap(
        doSim, range(30), n=20
    )  # number of threads to use.  On a 20-core machine I use 20.

    arr = coarsegrain(arr, 20)
    arr = np.clip(arr, 0, np.percentile(arr, 99.9))
    arr /= np.mean(np.sum(arr, axis=1))

    ran = np.arange(len(arr))
    mask = ran[:, None] > ran[None, :]

    arr[mask] = hicdata[mask]

    logarr = np.log(arr + 0.0001)
    plt.imshow(
        logarr,
        vmax=np.percentile(logarr, 99.9),
        extent=[low, high, high, low],
        interpolation="none",
    )
    nicePlot()


# This does the heatmap  from positions of loop extrudors only. Comment it out to proceed to actual simulations.
# Heatmap from loop extrudors only is nice for testing, and to understand how a simulated Hi-C will look like
# Note that it is muuch more contrasty than the final Hi-C heatmap would be. Especially at contact radius 10 and more

displayHeatmap()


def calculateAverageLoop():
    SMCTran = initModel()
    SMCTran.steps(1000000)
    dists = []
    for i in range(10000):
        SMCTran.steps(1000)
        left, right = SMCTran.getSMCs()
        dist = np.mean(right - left)
        # print(dist)
        dists.append(dist)
    print("final dist", np.mean(dists))
    exit()


# calculateAverageLoop()


def doPolymerSimulation(steps, dens, stiff, folder):
    from openmmlib.openmmlib import Simulation
    from openmmlib.polymerutils import grow_rw
    import time

    SMCTran = initModel()

    box = (N / dens) ** 0.33  # density = 0.1
    if os.path.exists(os.path.join(folder, "block10.dat")):
        block = scanBlocks(folder)["keys"].max() - 1
        data = polymerutils.load(os.path.join(folder, "block{0}.dat".format(block)))
    else:
        data = grow_rw(N, int(box) - 2)
        block = 0
    assert len(data) == N
    skip = 0
    time.sleep(1)

    while True:
        SMCTran.steps(3)
        if (block % 2000 == 0) and (skip == 0):
            print("doing dummy steps")
            SMCTran.steps(500000)
            skip = 100
            print(skip, "blocks to skip")

        a = Simulation(timestep=80, thermostat=0.01)

        a.setup(
            platform="CUDA",
            PBC=True,
            PBCbox=[box, box, box],
            GPU=sys.argv[4],
            precision="mixed",
        )
        a.saveFolder(folder)

        a.load(data)

        a.addHarmonicPolymerBonds(wiggleDist=0.1)

        if stiff > 0:
            a.addGrosbergStiffness(stiff)

        a.addPolynomialRepulsiveForce(trunc=1.5, radiusMult=1.05)
        left, right = SMCTran.getSMCs()
        for l, r in zip(left, right):
            a.addBond(l, r, bondWiggleDistance=0.5, distance=0.3, bondType="harmonic")
        a.step = block

        if skip > 0:
            print("skipping block")
            a.doBlock(steps, increment=False)
            skip -= 1

        if skip == 0:
            a.doBlock(steps)
            block += 1
            a.save()
        if block == 50000:
            break

        data = a.getData()

        del a

        time.sleep(0.1)


# This actually does a polymer simulation
doPolymerSimulation(
    steps=5000,
    stiff=2,
    dens=0.2,
    folder="flagshipLifetime{1}Mu{2}_try={0}".format(
        sys.argv[1], sys.argv[2], sys.argv[3]
    ),
)
