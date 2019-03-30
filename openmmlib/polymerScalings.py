# (c) 2013 Massachusetts Institute of Technology. All Rights Reserved
# Code written by: Maksim Imakaev (imakaev@mit.edu)

from mirnylib.systemutils import fmap
from . import polymerutils
from mirnylib.numutils import logbins
from . import contactmaps
from .contactmaps import load, giveContacts
import pickle
from math import sqrt
import random
import numpy as np

from copy import copy


def giveCpScaling(data, bins0, cutoff=1.1, integrate=False,
                  ring=False, intContacts=False, verbose=False):

    """
    Returns contact probability scaling for a given polymer conformation

    Parameters
    ----------
    data : 3xN array of ints/floats
        Input polymer conformation
    bins0 : list
        Bins to calculate scaling.
        Bins should probably be log-spaced; log-spaced bins can be quickly calculated using mirnylib.numtuis.logbinsnew. 
    cutoff : float, optional
        Cutoff to calculate scaling
    integrate : bool, optional
        if True, will return cumulative probability
    ring : bool, optional
        If True, will calculate contacts for the ring
    intContacts : bool, optional
        If True, will speed up calculation of contacts for a cubit lattice case.
    verbose : bool, optional
        If True, print some information.

    Returns
    -------
    (bins, contact probabilities) where "bins" are centers of bins in the input bins0 array.

    """

    if len(data) != 3:
        data = data.T
    if len(data) != 3:
        raise ValueError("Wrong data shape")

    N = len(data[0])

    bins0 = np.array(bins0)
    bins = [(bins0[i], bins0[i + 1]) for i in range(len(bins0) - 1)]
    if intContacts == False:
        contacts = np.array(giveContacts(data, cutoff))

    else:
        contacts = contactmaps.giveIntContacts(
            data)  # integer contacts are faster

    contacts = contacts[:, 1] - contacts[:, 0]  # contact lengthes

    if ring == True:
        mask = contacts > N // 2
        contacts[mask] = N - contacts[mask]
    scontacts = np.sort(contacts)  # sorted contact lengthes
    connections = 1. * np.diff(np.searchsorted(
        scontacts, bins0, side="left"))  # binned contact lengthes

    if ring == True:
        possible = np.diff(N * bins0)
    else:
        possible = np.diff(N * bins0 + 0.5 * bins0 - 0.5 * (bins0 ** 2))

    if verbose:
        print("average contacts per monomer:", connections.sum() / N)

    if integrate == False:
        connections /= possible
    if integrate == True:
        connections = np.cumsum(connections) / connections.sum()

    a = [sqrt(i[0] * (i[1] - 1)) for i in bins]
    if verbose:
        print(list(connections))
    return (a, connections)


def giveEndToEndScaling(data, bins, ring=False):
    """
    Returns end-to-end distance scaling of a given polymer conformation.
    ..warning:: This method averages end-to-end scaling over bins to make better average

    Parameters
    ----------

    data: 3xN array
    bins: the same as in giveCpScaling

    """
    if len(data) != 3:
        data = data.T
    if len(data) != 3:
        raise ValueError("Wrong data shape")

    N = len(data[0])
    if ring == True:
        data = np.concatenate([data, data], axis=1)

    rads = [0. for i in range(len(bins))]
    for i in range(len(bins)):
        length = bins[i]
        if ring == True:
            rads[i] = np.mean(np.sqrt(np.sum((data[:, :N]
                                           - data[:, length:length + N]) ** 2, 0)))
        else:
            rads[i] = np.mean(np.sqrt(np.sum((data[:, :-
                                            length] - data[:, length:]) ** 2, 0)))
    return (bins, rads)


def give_distance(data, bins=None, ring=False):
    """
    Returns end-to-end distance scaling of a given polymer conformation.
    ..warning:: This method averages end-to-end scaling over bins to make better average

    Parameters
    ----------

    data: 3xN array

    """
    N = len(data[0])
    if ring == True:
        data = np.concatenate([data, data], axis=1)
    bins = [(bins[i], bins[i + 1]) for i in range(len(bins) - 1)]

    rads = [0. for i in range(len(bins))]
    for i in range(len(bins)):
        oneBin = bins[i]
        rad = 0.
        count = 0
        for j in range(oneBin[0], oneBin[1], (oneBin[1] - oneBin[0]) // 10 + 1):
            length = j
            if ring == True:
                rad += np.mean(np.sqrt(np.sum((data[:, :N]
                                               - data[:, length:length + N]) ** 2, 0)))
            else:
                rad += np.mean(np.sqrt(np.sum((data[:, :-
                                                    length] - data[:, length:]) ** 2, 0)))
            count += 1

        rads[i] = rad / count
    bins = [sqrt(i[0] * i[1]) for i in bins]
    return (bins, rads)



def giveRgScaling(data, bins=None, ring=False):
    "main working horse for radius of gyration"
    "uses dymanic programming algorithm"

    if len(data) != 3:
        data = data.T
    if len(data) != 3:
        raise ValueError("Wrong data shape")

    data = np.array(data, float)
    coms = np.cumsum(data, 1)  # cumulative sum of locations to calculate COM
    coms2 = np.cumsum(
        data ** 2, 1)  # cumulative sum of locations^2 to calculate RG

    def radius_gyration(len2):
        data
        if ring == True:
            comsadd = coms[:, :len2].copy()
            coms2add = coms2[:, :len2].copy()
            comsadd += coms[:, -1][:, None]
            coms2add += coms2[:, -1][:, None]
            comsw = np.concatenate([coms, comsadd], axis=1)
            coms2w = np.concatenate([coms2, coms2add],
                                    axis=1)  # for rings we extend need longer chain
        else:
            comsw = coms
            coms2w = coms2

        coms2d = (-coms2w[:, :-len2] + coms2w[:, len2:]) / len2
        comsd = ((comsw[:, :-len2] - comsw[:, len2:]) / len2) ** 2
        diffs = coms2d - comsd
        sums = np.sqrt(np.sum(diffs, 0))
        return np.mean(sums)

    rads = [0. for i in range(len(bins))]
    for i in range(len(bins)):
        rads[i] = radius_gyration(int(bins[i]))
    return (copy(bins), rads)


def give_radius_scaling(data, bins=None, ring=False):
    "main working horse for radius of gyration"
    "uses dymanic programming algorithm"
    "This function is here just for backwards compatibility, with auto-assigned bins" 
    bins = [int(sqrt(bins[i] * bins[i + 1])) for i in range(len(bins) - 1)]
    return giveRgScaling(data, bins, ring)

def give_radius_scaling_eig(data, bins=None):
    # gives volume^^0.33 as  defined through  eigenvectors
    if bins is None:
        bins = [2 * i for i in logbins(1, 0.45 * len(data[0]), 1.3, 40)]
    x, y, z = data[0], data[1], data[2]
    coords = [x, y, z]
    sums = [[i * j for j in coords] for i in coords]
    sums = np.array(sums)
    N = len(data[0])
    coms = np.cumsum(data, 1)
    sums = np.cumsum(sums, 2)

    def tensor(a, b):
        newsums = (sums[:, :, b] - sums[:, :, a]) / float(b - a)
        newcoms = (coms[:, b] - coms[:, a]) / float(b - a)
        tensor = newsums - np.outer(newcoms, newcoms)
        return np.linalg.eigvalsh(tensor)
    ret = []
    for i in bins:
        av = 0.
        for j in range(1000):
            t = np.random.randint(5, N - 5 - i)
            res = tensor(t, t + i)
            av += sqrt(3) * (res[0] * res[1] * res[2] * 1.) ** (1 / 6.)
        ret.append(av / 1000)
    retret = (copy(bins), ret)
    return  retret


def subchainDensityFunction(filenames, bins, normalize="Rg", maxLength=3, Nbins=30, coverage=1., centerAt="mid", **kwargs):
    """Calculates density function of subchains
    That is, for each bin size, it calculates an average density profile
    for subchains of the size within the bin.
    Density profile is how density of a subchain depends on a distance from the center.

    filenames : str
        Filenames to average density function
    bins : iterable
         List of positions of bins (lengthes at which to evaluate).
    normalize : str, not implemented
        How to normalize the subchain density function
    lengthmult : int, optional
        Calculate distribution to lengthmult*Rg distance only (default = 3)
    Nbins : int, optional
        Number of bins for each distribution (default = 30)
    coverage : float, optional
        Use each monomer 'coverage' times (on average) to calculate each distribution. Default = 1.


    """
    normalize = normalize.lower()
    centerAt = centerAt.lower()
    results = []
    newbins = [(i - 2, i + 2) for i in bins]
    binsForRg = sum([list(i) for i in newbins], [])
    midbins = [(i[0] + i[1]) // 2 for i in newbins]
    rgs = []
    for filename in filenames:
        rgs.append(give_radius_scaling(polymerutils.load(filename).T, binsForRg, ring=False)[1][::2])
    rgs = np.mean(rgs, axis=0)
    for filename in filenames:
        data = polymerutils.load(filename)
        N = len(data)


        curresults = []
        labels = []
        for onebin, rg in zip(newbins, rgs):
            labels.append("S = %.1lf; " % (np.mean(onebin)) + " Rg=%.2lf" % rg)
            if normalize == "rg":
                lengthbins = np.linspace(1, maxLength * rg, Nbins)
            else:
                lengthbins = np.linspace(1, maxLength, Nbins)

            lengthBinMids = (lengthbins[:-1] + lengthbins[1:]) * 0.5
            volumes = (4. / 3.) * 3.141592 * (lengthbins ** 3)
            volumes = np.diff(volumes)
            count = int(N * coverage / np.mean(onebin) + 1)
            sphereCounts = np.zeros(len(volumes), float)

            for i in range(count):
                size = np.random.randint(onebin[0], onebin[1])
                start = np.random.randint(0, N - size)
                subchain = data[start:start + size]
                if centerAt == "com":
                    com = np.mean(subchain, axis=0)
                elif centerAt == "mid":
                    com = subchain[len(subchain) // 2]
                else:
                    raise ValueError("Provide correct centerAt: com or mid")
                shifted = subchain - com[None, :]
                # print shifted
                dists = np.sqrt(np.sum(shifted ** 2, axis=1))
                sphereCounts += np.histogram(dists, lengthbins)[0]
            sphereCounts /= (volumes * count)
            # curresults.append(np.array([lengthBinMids/rg,sphereCounts]))
            if normalize == "rg":
                curresults.append(np.array([lengthBinMids / rg, sphereCounts]))
            elif normalize == "none":
                curresults.append(np.array([lengthBinMids, sphereCounts]))
            else:
                print("Normalize=", normalize, "is not implemented")
                raise NotImplementedError()
        results.append(curresults)
    results = np.mean(np.array(results, float), axis=0)
    for i, label in zip(results, labels):
        if "label" not in kwargs:
            kwargs["label"] = label
        import matplotlib.pyplot as plt
        plt.plot(i[0], i[1], **kwargs)

    return dict(list(zip(midbins, results)))


def give_slices(base, tosave, slices, sliceParams,
                multipliers, mode="chain", loadFunction=load,
                integrate=False, normalize=False, exceptionList=[],
                nproc=4, cutoff=1.7, binstep=1.15, integerSlices=True, verbose=False):
    np.seterr(invalid='raise')

    plotsBySlice = []

    for cur_slice in slices:

        files = []

        def slice2D(a, b, mult=[1]):
            tm = []
            if type(b) == tuple:
                for i in range(b[0], b[1] + 1):
                    tm.append((i, a))
            elif type(b) == int:
                for i in range(1, b + 1):
                    tm.append((i, a))
            elif type(b) == list:
                tm = [(i, a) for i in b]
            if integerSlices:
                tm2 = sorted(list(set([(i[0], int(
                    float(i[1]) * m)) for i in tm for m in mult])))
            else:
                tm2 = sorted(tm)
            print(tm2)
            return tm2

        def slice3D(a, b, c, mult=[1]):
            tm = []
            for i in range(b[0], b[1] + 1):
                for t in range(c[0], c[1] + 1):
                    tm.append((i, a, t))
            tm2 = sorted(list(set([(i[0], int(
                float(i[1]) * m)) for i in tm for m in mult])))
            print(tm2)
            return tm2

        # sluces actually are defined
        runs = slice2D(cur_slice, sliceParams, multipliers)
        # runs = slice3D(cur_slice, (1,14),(1,10),multipliers)

        for i in runs:
            # filename is replaced in slices
            try:
                files.append(base.replace("DATA1", str(i[0])).replace("DATA2",
                                                                      str(i[1])).replace("DATA3", str(i[2])))
            except:
                files.append(base.replace("DATA1", str(i[0]
                                                       )).replace("DATA2", str(i[1])))

        datas = []

        def newload(i):
            # loads a file
            try:
                data = loadFunction(i, False)
                if len(data) != 3:
                    data = data.T
                if len(data) != 3:
                    raise Exception("Wrong shape of data")
                data = np.asarray(data, order="C", dtype=float)
                return data
            except tuple(exceptionList):
                print("file not found", i)
                return None

        # use this for determining the file size
        datas = [x for x in fmap(newload, files[::
                                            len(files) // 20 + 1], n=3) if x is not None]
        datlen = len(datas[0][0])

        if mode == "chain":
            bins2 = logbins(4, datlen - 100, binstep)
        if mode == "parts":
            bins2 = logbins(4, datlen - 100, binstep)
        if (mode == "ring") or (mode == "intring"):
            b1 = logbins(2, datlen // 4 - 1, binstep)
            bins2 = [2 * i for i in b1]
            print(bins2)
        binsrg = logbins(4, datlen - 100, binstep)

        def give_plots(i):
            data = newload(i)
            if data is None:
                return None
            i = data

            if (mode == "ring") or (mode == "intring"):
                b = give_radius_scaling(i, binsrg, ring=True)
            else:
                b = give_radius_scaling(i, binsrg, ring=False)

            if (mode == "chain"):
                a = giveCpScaling(i, bins2, cutoff, integrate, verbose=verbose)
            if (mode == "ring"):
                a = giveCpScaling(i, bins2, cutoff, integrate, ring=True, verbose=verbose)
            if (mode == "intring"):
                a = giveCpScaling(i, bins2, cutoff, integrate, ring=True,
                                  project=False, intContacts=True, verbose=verbose)
            if (mode == "project"):
                a = giveCpScaling(i, bins2, 1.450, integrate, project=True, verbose=verbose)

            if (mode == "ring") or (mode == "intring"):
                c = give_distance(i, bins2, ring=True)
            else:
                c = give_distance(i, bins2, ring=False)

            if (normalize == True):
                a = np.array(a)
                pos = a[0]
                values = a[1]
                bins = np.r_[1.5 * pos[0] - 0.5 * pos[1],
                             0.5 * (pos[1:] + pos[:-1]), pos[-1]]
                lens = bins[1:] - bins[:-1]
                ints = np.cumsum(lens * values)

                values /= ints[-1]
                ints /= ints[-1]
                a = [pos, values]

            a = np.array(a, dtype=float)
            b = np.array(b, dtype=float)
            c = np.array(c, dtype=float)

            return np.array([a, b, c])
        random.shuffle(files)

        parPlots = fmap(give_plots, files, n=nproc)

        parPlots = [x for x in parPlots if x is not None]

        means = np.mean(parPlots, axis=0)
        plotsBySlice.append([means, {"slice":cur_slice}])

    if tosave is not None:
        pickle.dump(plotsBySlice, open(tosave, 'wb'), -1)
    print("Finished!!!")
    return plotsBySlice

