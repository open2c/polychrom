# Code written by: Maksim Imakaev (imakaev@mit.edu)
"""
This file contains a bunch of method to work on contact maps of a Hi-C data.


"""

from __future__ import absolute_import, division, print_function, unicode_literals
import os
import traceback

import numpy as np

from math import sqrt
import sys
from polychrom.polymerutils import load
import warnings
import polychrom.polymerutils as polymerutils
import time
from scipy.spatial import ckdtree
from polychrom.polymer_analyses import calculate_contacts as giveContacts


def rescalePoints(points, bins):
    "converts array of contacts to the reduced resolution contact map"
    a = np.histogram2d(points[:, 0], points[:, 1], bins)[0]
    a = a + np.transpose(a)

    return a


def rescaledMap(data, bins, cutoff=1.7, contactMap=None):
    # print data.sum(), bins.sum(), cutoff
    """calculates a rescaled contact map of a structure
    Parameters
    ----------
    data : Nx3 or 3xN array
        polymer conformation

    bins : Lx1 array
        bin starts

    cutoff : float, optional
        cutoff for contacts

    Returns
    -------
        resXres array with the contact map
    """

    t = giveContacts(data, cutoff)
    x = np.searchsorted(bins, t[:, 0]) - 1
    y = np.searchsorted(bins, t[:, 1]) - 1
    assert x.min() >= 0
    assert y.min() >= 0
    assert x.max() < len(bins) - 1
    assert y.max() < len(bins) - 1
    matrixSize = len(bins) - 1
    index = matrixSize * x + y

    unique, inds = np.unique(index, return_counts=True)
    uniquex = unique // matrixSize
    uniquey = unique % matrixSize

    if contactMap is None:
        contactMap = np.zeros((matrixSize, matrixSize), dtype=int)
    contactMap[uniquex, uniquey] += inds

    return contactMap


def pureMap(data, cutoff=1.7, contactMap=None):
    """calculates an all-by-all contact map of a single polymer chain.
    Doesn't work for multi-chain polymers!!!
    If contact map is supplied, it just updates it

    Parameters
    ----------
    data : Nx3 or 3xN array
        polymer conformation
    cutoff : float
        cutoff for contacts
    contactMap : NxN array, optional
        contact map to update, if averaging is used
    """
    data = np.asarray(data)
    if len(data.shape) != 2:
        raise ValueError("Wrong dimensions of data")
    if 3 not in data.shape:
        raise ValueError("Wrong size of data: %s,%s" % data.shape)
    if data.shape[0] == 3:
        data = data.T
    data = np.asarray(data, float, order="C")

    t = giveContacts(data, cutoff)
    N = data.shape[0]
    if contactMap is None:
        contactMap = np.zeros((N, N), "int32")
    contactMap[t[:, 0], t[:, 1]] += 1
    contactMap[t[:, 1], t[:, 0]] += 1
    return contactMap


def averageBinnedContactMap(
    filenames,
    chains=None,
    binSize=None,
    cutoff=1.7,
    n=4,  # Num threads
    loadFunction=load,
    exceptionsToIgnore=None,
    printProbability=1,
    map_function=map,
):
    """
    Returns an average contact map of a set of conformations.
    Non-existing files are ignored if exceptionsToIgnore is set to IOError.
    example:\n

    An example:

    .. code-block:: python
        >>> filenames = ["myfolder/blockd%d.dat" % i for i in xrange(1000)]
        >>> cmap = averageBinnedContactMap(filenames) + 1  #getting cmap
        #either showing a log of a map (+1 for zeros)
        >>> plt.imshow(numpy.log(cmap +1))
        #or truncating a map
        >>> vmax = np.percentile(cmap, 99.9)
        >>> plt.imshow(cmap, vmax=vmax)
        >>> plt.show()

    Parameters
    ----------
    filenames : list of strings
        Filenames to average map over
    chains : list of tuples or Nx2 array
        (start,end+1) of each chain
    binSize : int
        size of each bin in monomers
    cutoff : float, optional
        Cutoff to calculate contacts
    n : int, optional
        Number of threads to use.
        By default 4 to minimize RAM consumption.
    exceptionsToIgnore : list of Exceptions
        List of exceptions to ignore when finding the contact map.
        Put IOError there if you want it to ignore missing files.

    Returns
    -------
    tuple of two values:
    (i) MxM numpy array with the conntact map binned to binSize resolution.
    (ii) chromosomeStarts a list of start sites for binned map.

    """
    n = min(n, len(filenames))
    subvalues = [filenames[i::n] for i in range(n)]

    getResolution = 0
    fileInd = 0
    while getResolution == 0:
        try:
            data = loadFunction(filenames[fileInd])  # load filename
            getResolution = 1
        except:
            fileInd = fileInd + 1
        if fileInd >= len(filenames):
            print("no valid files found in filenames")
            raise ValueError("no valid files found in filenames")

    if chains is None:
        chains = [[0, len(data)]]
    if binSize is None:
        binSize = int(np.floor(len(data) / 500))

    bins = []
    chains = np.asarray(chains)
    chainBinNums = np.ceil((chains[:, 1] - chains[:, 0]) / (0.0 + binSize))
    for i in range(len(chainBinNums)):
        bins.append(binSize * (np.arange(int(chainBinNums[i]))) + chains[i, 0])
    bins.append(np.array([chains[-1, 1] + 1]))
    bins = np.concatenate(bins)
    bins = bins - 0.5
    Nbase = len(bins) - 1

    if Nbase > 10000:
        warnings.warn(
            UserWarning("very large contact map" " may be difficult to visualize")
        )

    chromosomeStarts = np.cumsum(chainBinNums)
    chromosomeStarts = np.hstack((0, chromosomeStarts))

    def myaction(values):  # our worker receives some filenames
        mysum = None  # future contact map.
        for i in values:
            try:
                data = loadFunction(i)
                if np.random.random() < printProbability:
                    print(i)
            except tuple(exceptionsToIgnore):
                print("file not found", i)
                continue

            if data.shape[0] == 3:
                data = data.T
            if mysum is None:  # if it's the first filename,

                mysum = rescaledMap(data, bins, cutoff)  # create a map

            else:  # if not
                rescaledMap(data, bins, cutoff, mysum)
                # use existing map and fill in contacts

        return mysum

    blocks = list(map_function(myaction, subvalues))
    blocks = [i for i in blocks if i is not None]
    a = blocks[0]
    for i in blocks[1:]:
        a = a + i
    a = a + a.T

    return a, chromosomeStarts


def averagePureContactMap(
    filenames,
    cutoff=1.7,
    n=4,  # Num threads
    loadFunction=load,
    exceptionsToIgnore=[],
    printProbability=0.005,
    map_function=map,
):
    """
        Parameters
    ----------
    filenames : list of strings
        Filenames to average map over
    cutoff : float, optional
        Cutoff to calculate contacts
    n : int, optional
        Number of threads to use.
        By default 4 to minimize RAM consumption with pure maps.
    exceptionsToIgnore : list of Exceptions
        List of exceptions to ignore when finding the contact map.
        Put IOError there if you want it to ignore missing files.

    Returns
    -------

    An NxN (for pure map) numpy array with the contact map.
    """

    """
    Now we actually need to modify our contact map by adding
    contacts from each new file to the contact map.
    We do it this way because our contact map is huge (maybe a gigabyte!),
    so we can't just add many gigabyte-sized arrays together.
    Instead of this each worker creates an empty "average contact map",
    and then loads files one by one and adds contacts from each file to a contact map.
    Maps from different workers are then added together manually.
    """

    n = min(n, len(filenames))
    subvalues = [filenames[i::n] for i in range(n)]

    def myaction(values):  # our worker receives some filenames
        mysum = None  # future contact map.
        for i in values:
            try:
                data = loadFunction(i)
                if np.random.random() < printProbability:
                    print(i)
            except tuple(exceptionsToIgnore):
                print("file not found", i)
                continue
            except:
                print("Unexpected error:", sys.exc_info()[0])
                print("File is: ", i)
                return -1

            if data.shape[0] == 3:
                data = data.T
            if mysum is None:  # if it's the first filename,

                if len(data) > 6000:
                    warnings.warn(
                        UserWarning(
                            "very large contact map"
                            " may cause errors. these may be fixed with n=1 threads."
                        )
                    )
                if len(data) > 20000:
                    warnings.warn(
                        UserWarning(
                            "very large contact map" " may be difficult to visualize."
                        )
                    )

                mysum = pureMap(data, cutoff)  # create a map

            else:  # if not
                pureMap(data, cutoff, mysum)
                # use existing map and fill in contacts

        return mysum

    blocks = list(map_function(myaction, subvalues))
    blocks = [i for i in blocks if i is not None]
    a = blocks[0]
    for i in blocks[1:]:
        a = a + i
    return a
