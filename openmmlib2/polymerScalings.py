# Code written by: Maksim Imakaev (imakaev@mit.edu)

from . import polymerutils
from . import contactmaps
from .contactmaps import load, calculate_contacts
import pickle
from math import sqrt
import random
import numpy as np

from copy import copy



def generate_bins(N, start=4, bins_per_order_magn=10):
    lstart = np.log(start)
    lend = np.log(N)
    num = np.ceil((lend - lstart) * bins_per_order_magn)
    
    

def contact_scaling(data, bins0 = None, cutoff=1.1, integrate=False,
                  ring=False, verbose=False):

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
    (mids, contact probabilities) where "mids" are centers of bins in the input bins0 array in logspace.

    """

    if len(data) != 3:
        data = data.T
    if len(data) != 3:
        raise ValueError("Wrong data shape")

    N = len(data[0])
    
    if bins0 is None: 
        bins0 = generate_bins(N)
    
    bins0 = np.array(bins0)
    bins = [(bins0[i], bins0[i + 1]) for i in range(len(bins0) - 1)]
    contacts = np.array(calculate_contacts(data, cutoff))

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


def R2_scaling(data, bins, ring=False):
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
    return (np.array(bins), rads)




def Rg2_scaling(data, bins=None, ring=False):
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
    return (np.array(bins), rads)



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


