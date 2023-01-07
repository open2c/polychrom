import matplotlib

# matplotlib.use("Agg")

from mirnylib.plotting import nicePlot
import os
import pickle
from openmmlib import contactmaps
from mirnylib.numutils import zoomArray
from openmmlib import polymerutils

import matplotlib.pyplot as plt
import numpy as np
from mirnylib.h5dict import h5dict
from mirnylib.genome import Genome
from mirnylib.numutils import completeIC, coarsegrain
from mirnylib.systemutils import setExceptionHook
from openmmlib.contactmapManager import averageContacts
import pandas as pd
from mirnylib.numutils import coarsegrain

setExceptionHook()

import mirnylib.plotting

filename = "/net/levsha/share/nezar/ctcf_sites/GM12878.ctcf_narrowPeak.loj.encodeMotif.rad21.txt"
SEPARATION = 400
LIFETIME = 200


class simulator(object):
    def __init__(self, i, forw, rev, blocks, steps):
        import pyximport

        pyximport.install()
        from smcTranslocatorDirectional import smcTranslocator
        import numpy as np

        N = len(forw)
        birthArray = np.zeros(N, dtype=np.double) + 0.1
        deathArray = np.zeros(N, dtype=np.double) + 1.0 / LIFETIME
        stallArrayLeft = forw
        stallArrayRight = rev
        stallDeathArray = np.zeros(N, dtype=np.double) + 1 / LIFETIME
        pauseArray = np.zeros(N, dtype=np.double)
        smcNum = N // SEPARATION
        myDeathArray = deathArray
        SMCTran = smcTranslocator(
            birthArray,
            myDeathArray,
            stallArrayLeft,
            stallArrayRight,
            pauseArray,
            stallDeathArray,
            smcNum,
        )
        self.SMCTran = SMCTran
        self.steps = steps
        self.blocks = blocks
        self.i = 0

    def next(self):
        self.i += 1
        if self.i == self.blocks:
            raise StopIteration
        self.SMCTran.steps(self.steps)
        conts = self.SMCTran.getSMCs()
        if self.i % 1000 == 500:
            print(self.i, conts[0][0])

        return np.array(conts) // 20


def getForwBacv(mu=3):

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

    forw = np.bincount(mid_1k[strand], weights=(strength[strand] / 20), minlength=M)
    rev = np.bincount(mid_1k[~strand], weights=(strength[~strand] / 20), minlength=M)

    low = 60000
    high = 75000
    lowMon = low * 1000 // 600
    highMon = high * 1000 // 600
    forw = forw[lowMon:highMon]
    rev = rev[lowMon:highMon]

    def logistic(x, mu=3):
        x[x == 0] = -99999999
        return 1 / (1 + np.exp(-(x - mu)))

    forw = logistic(forw, mu)
    rev = logistic(rev, mu)

    return forw, rev


# uncommend this to just display a simulated heatmap.
# hm = averageContacts(simulator, range(30), 1500, classInitArgs=[forw, rev, 5000, 150],  bucketNum = 20, nproc=30)
# exit()
# print(hm.shape)


class contactCalculator:
    def __init__(self, filenames, cutoff, coarsegrainBy, method):
        self.filenames = filenames
        self.cutoff = cutoff
        self.coarsegrain = coarsegrainBy
        self.method = method

    def next(self):
        if len(self.filenames) == 0:
            raise StopIteration

        data = polymerutils.load(self.filenames.pop())
        contacts = self.method(data, cutoff=self.cutoff) // self.coarsegrain
        return contacts


def getCmap(prefix="", radius=6):
    """
    This is a function to calculate a simulated Hi-C contact map from one or several folders with conformations, defined by "prefix".
    """
    n = 20  # number of processes to use = number of cores
    coarsegrainBy = 5  # how many monomers per pixel in a heatmap
    print(os.getcwd())

    folders = [i for i in os.listdir(".") if i.startswith(prefix)]
    foldes = [i for i in folders if os.path.exists(i)]
    print(folders)
    files = sum([polymerutils.scanBlocks(i)["files"] for i in folders], [])
    filegroups = [files[i::n] for i in range(n)]
    data = polymerutils.load(files[0])
    N = len(data)
    method = contactmaps.findMethod(data, radius)
    cmapN = int(np.ceil(N / coarsegrainBy))

    cmap = averageContacts(
        contactCalculator,
        filegroups,
        cmapN,
        classInitArgs=[radius, coarsegrainBy, method],
        nproc=n,
        bucketNum=60,
        useFmap=True,
    )
    pickle.dump(cmap, open("cmaps/cmap{0}_r={1}.pkl".format(prefix, radius), "wb"))


def getAllCmaps():
    """This functions iterates over different contact radii and over different contactmap names,
    and calculates contact maps. Right now set only for one contact map.
    """
    for radius in [8]:
        for prefix in [
            # "lessCTCF","lessLifetime","moreLifetime","moreSeparation","steps=500",
            #   "flagship_try","flagshipLessCTCF","flagshipMoreCtcf","flagship_cellType"
            # "flagshipMod_", "flagshipModLessCtcf", "flagshipModMoreCtcf"
            # "flagshipBoundaryStallLifetime100Mu3","flagshipBoundaryStallLifetime200Mu3",
            # "flagshipBoundaryStallLifetime300Mu3",
            # "flagshipLifetime100Mu3","flagshipLifetime200Mu3",
            "flagshipLifetime300Mu3"
            # ,"flagshipLifetime300Mu2","flagshipLifetime300Mu4",
        ]:
            print(prefix, radius)
            getCmap(prefix, radius)
    exit()


# getAllCmaps()


def pivotHeatmap(heatmap, diags=20):
    N = len(heatmap)
    newdata = np.zeros((diags, 2 * N), dtype=float)
    for i in range(diags):
        diag = np.diagonal(heatmap, i)
        pad = N - len(diag)
        newdiag = np.zeros(2 * len(diag), dtype=float)
        newdiag[::2] = diag
        newdiag[1::2] = diag
        if pad == 0:
            newdata[i] = newdiag
        else:
            newdata[i][pad:-pad] = newdiag
    return newdata


def showCmap():
    """Shows Hi-C data together with the simulated data. Hi-C data created by hiclib is needed for that,
    but you can replace the line mydict=h5dict()... and the following line with your own data loading code."""

    low = 60000
    high = 75000
    lowMon = low * 1000 // 600
    highMon = high * 1000 // 600

    low20 = low // 10
    high20 = high // 10
    # here Hi-C data is loaded for display purposes only..... replace it with your own code if your data is in a different format
    mydict = h5dict(
        "/home/magus/HiC2011/Erez2014/hg19/GM12878_inSitu-all-combined-10k_HighRes.byChr",
        "r",
    )
    hicdata = mydict.get_dataset("13 13")[low20:high20, low20:high20]

    hicdata = completeIC(hicdata)
    curshape = hicdata.shape
    newshape = (1000 * (high - low)) // (600 * 5)
    print(hicdata.shape, newshape)
    hicdata = zoomArray(hicdata, (newshape, newshape))
    hicdata = np.clip(hicdata, 0, np.percentile(hicdata, 99.99))
    hicdata /= np.mean(np.sum(hicdata, axis=1))

    # hicdata = hm / np.mean(np.sum(hm, axis=1))

    for fname in os.listdir("cmaps"):

        cmap = pickle.load(open(os.path.join("cmaps", fname), "rb"))
        # arr = coarsegrain(cmap, 2)
        arr = cmap
        if arr.shape[0] != hicdata.shape[0]:
            continue
        print(arr.shape)

        arr = arr / np.mean(np.sum(arr, axis=1))
        ran = np.arange(len(arr))
        mask = ran[:, None] > ran[None, :]
        arr[mask] = hicdata[mask]

        logarr = np.log(arr + 0.0001)
        # noinspection PyTypeChecker
        plt.imshow(
            logarr,
            vmax=np.percentile(logarr, 99.99),
            vmin=np.percentile(logarr, 10),
            extent=[low, high, high, low],
            interpolation="none",
        )
        plt.savefig(os.path.join("heatmaps", fname + ".png"))
        plt.savefig(os.path.join("heatmaps", fname + ".pdf"))
        plt.show()
        plt.clf()


# getCmap()
# showCmap()
# plt.show()


def showCmapNew():
    """Saves a bunch of heatmaps at high resolutions."""

    plt.figure(figsize=(8, 8))
    low = 60000
    high = 75000
    lowMon = low * 1000 // 600
    highMon = high * 1000 // 600

    low20 = low // 10
    high20 = high // 10
    mydict = h5dict(
        "/home/magus/HiC2011/Erez2014/hg19/GM12878_inSitu-all-combined-10k_HighRes.byChr",
        "r",
    )

    hicdata = mydict.get_dataset("13 13")[low20:high20, low20:high20]
    hicdata = completeIC(hicdata)
    curshape = hicdata.shape
    resolutionMon = 5
    newshape = (1000 * (high - low)) // (600 * resolutionMon)
    print(hicdata.shape, newshape)
    hicdata = zoomArray(hicdata, (newshape, newshape))
    hicdata = np.clip(hicdata, 0, np.percentile(hicdata, 99.99))
    hicdata /= np.mean(np.sum(hicdata, axis=1))

    # hicdata = hm / np.mean(np.sum(hm, axis=1))

    # for fname in os.listdir("cmaps"):
    for fname in ["cmapflagshipLifetime300Mu3_r=8.pkl"]:
        if ("r=8" not in fname) or ("Lifetime" not in fname):
            print("not going", fname)
            continue
        try:
            mu = float(fname.split("_r=")[0].split("Mu")[1])
        except:
            continue
        forw, rev = getForwBacv(mu)

        cmap = pickle.load(open(os.path.join("cmaps", fname), "rb"))
        # arr = coarsegrain(cmap, 2)
        arr = cmap

        if arr.shape[0] != hicdata.shape[0]:
            continue
        arr = arr / np.mean(np.sum(arr, axis=1))
        hicdata *= 1.5
        diags = 1000
        print(arr.shape)
        ax = plt.subplot(211)
        turned = pivotHeatmap(arr, diags)[::-1] * 3
        turned2 = pivotHeatmap(hicdata, diags)
        turned = np.concatenate([turned, turned2], axis=0)
        myextent = [
            low,
            high,
            -(high - low) * diags / len(arr),
            (high - low) * diags / len(arr),
        ]
        plt.imshow(
            np.log(turned + 0.0001),
            aspect=0.5,
            cmap="fall",
            vmax=-4,
            vmin=-8,
            extent=myextent,
            interpolation="none",
        )
        # plt.colorbar()

        # plt.ylim([-(high - low) *  diags/ len(arr) , (high - low) *  diags/ len(arr) ])
        # nicePlot(show=False)

        plt.subplot(413, sharex=ax)
        xaxis = np.arange(len(forw) // 20) * 12 + 60000
        forwcg = coarsegrain(forw, 20)
        revcg = coarsegrain(rev, 20)
        plt.vlines(xaxis[forwcg > 0], 0, forwcg[forwcg > 0], color="blue")
        plt.vlines(xaxis[revcg > 0], 0, revcg[revcg > 0], color="green")
        # plt.scatter(xaxis[forwcg>0], forwcg[forwcg>0], label = "forward CTCF")
        # plt.scatter(xaxis[revcg > 0],revcg[revcg>0], label = "reverse CTCF")
        plt.xlim([60000, 75000])
        plt.title(fname)
        plt.legend()
        plt.show()
        continue
        # nicePlot(show=False)
        # plt.subplot(414, sharex = ax)
        # plt.plot(xaxis, data)

        # plt.show()

        # arr = arr / np.mean(np.sum(arr, axis=1))
        # ran = np.arange(len(arr))
        # mask = ran[:,None] > ran[None,:]
        # arr[mask] = hicdata[mask]

        # logarr = np.log(arr + 0.0001)
        # noinspection PyTypeChecker
        # plt.imshow(logarr, vmax = np.percentile(logarr, 99.9), extent = [low, high, high, low], interpolation = "none")
        for st in range(60000, 75000, 1000):
            for size in [2000, 3000, 5000]:
                end = st + size
                if end > 75000:
                    continue
                plt.xlim([st, end])
                plt.savefig(os.path.join("heatmaps", "{0}_st={1}_end={2}_r=2.png".format(fname, st, end)))
                plt.savefig(os.path.join("heatmaps", "{0}_st={1}_end={2}_r=2.pdf".format(fname, st, end)))
        plt.clf()

    plt.show()


# getCmap()
showCmapNew()
# plt.show()
