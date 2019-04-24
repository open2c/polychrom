# Code written by: Maksim Imakaev (imakaev@mit.edu)

from . import polymerutils
from math import sqrt

import random
import numpy as np
import pandas as pd 

from scipy.spatial import ckdtree


def calculate_contacts(data, cutoff=1.7, method="auto"):
    """Returns contacts of a single polymer with a given cutoff

    .. warning:: Use this only to find contacts of a single polymer chain
    with distance between monomers of 1.
    Multiple chains will lead to silent bugs.

    Parameters
    ----------
    data : Nx3 or 3xN array
        Polymer configuration. One chaon only.
    cutoff : float , optional
        Cutoff distance that defines contact

    Returns
    -------
    k by 2 array of contacts. Each row corresponds to a contact.
    """
    if data.shape[1] != 3:
        raise ValueError("Incorrect polymer data shape. Must be Nx3.")

    if np.isnan(data).any():
        raise RuntimeError("Data contains NANs")

    tree = ckdtree.cKDTree(data)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    return pairs


def generate_bins(N, start=4, bins_per_order_magn=10):
    lstart = np.log10(start)
    lend = np.log10(N-1)  + 1e-6
    num = np.ceil((lend - lstart) * bins_per_order_magn)
    bins = np.unique(np.logspace(lstart, lend, dtype=int))
    assert bins[-1] == N - 1
    return bins

    

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

    N = len(data)
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


def R2_scaling(data, bins=None, ring=False):
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

    N = data.shape[1]
    
    if bins is None:
        bins = generate_bins(N)
    if ring == True:
        data = np.concatenate([data, data], axis=1)

    rads = [0. for i in range(len(bins))]
    for i in range(len(bins)):
        length = bins[i]
        if ring == True:
            rads[i] = np.mean((np.sum((data[:, :N]
                                           - data[:, length:length + N]) ** 2, 0)))
        else:
            rads[i] = np.mean((np.sum((data[:, :-
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
    N = data.shape[1]
    if bins is None:
        bins = generate_bins(N)
        
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
        sums = (np.sum(diffs, 0))
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

def ndarray_groupby_aggregate(df, ndarray_cols, aggregate_cols, value_cols=[], 
                              sample_cols=[], preset="sum",
                               ndarray_agg = lambda x:np.sum(x, axis=0), value_agg = lambda x:x.sum()):
    
    """
    A version of pd.groupby that is aware of numpy arrays as values of columns 
    
    It aggregates columns ndarray_cols using ndarray_agg aggregator
    It aggregates value_cols using value_agg aggregator
    It takes the first element in sample_cols
    and aggregates over aggregate_cols 
    
    It has presets for sum, mean and nanmean. 
    """

    if preset == "sum":
        ndarray_agg = lambda x:np.sum(x, axis=0)
        value_agg = lambda x:x.sum()
    elif preset == "mean":
        ndarray_agg = lambda x:np.mean(x, axis=0)
        value_agg = lambda x:x.mean()
    elif preset == "nanmean":
        ndarray_agg = lambda x:np.nanmean(x, axis=0)
        value_agg = lambda x:x.mean()        
    
    def combine_values(in_df):        
        "splits into ndarrays, 'normal' values, and samples; performs aggregation, and returns a Series"
        average_arrs = pd.Series(index=ndarray_cols, 
                data=[ndarray_agg([np.asarray(j) for j in in_df[i].values]) for i in ndarray_cols])
        average_values = value_agg(in_df[value_cols])
        sample_values = in_df[sample_cols].iloc[0]
        agg_series = pd.concat([average_arrs, average_values, sample_values])        
        return agg_series
    
    return  df.groupby(aggregate_cols).apply(combine_values)
    

    
def streaming_ndarray_agg(in_stream,  ndarray_cols, aggregate_cols, value_cols=[],  sample_cols=[], 
                  chunksize=30000, add_count_col=False, divide_by_count=False):
    
    """
    Takes in_stream of dataframes
    
    Applies ndarray-aware groupby-sum or groupby-mean: treats ndarray_cols as numpy arrays, 
    value_cols as normal values, for sample_cols takes the first element. 
    
    Does groupby over aggregate_cols 
    
    if add_count_col is True, adds column "count", if it's a string - adds column with add_count_col name     

    if divide_by_counts is True, divides result by column "count". 
    If it's a string, divides by divide_by_count column
    
    
    """
    value_cols_orig = [i for i in value_cols]
    ndarray_cols, value_cols = list(ndarray_cols), list(value_cols)
    aggregate_cols, sample_cols = list(aggregate_cols), list(sample_cols)
    if add_count_col is not False:
        if add_count_col is True:
            add_count_col = "count"
        value_cols.append(add_count_col)

    
    def agg_one(dfs, aggregate):
        """takes a list of DataFrames and old aggregate
        performs groupby and aggregation  and returns new aggregate"""
        if add_count_col is not False:
            for i in dfs:
                i[add_count_col] = 1
                
        df = pd.concat(dfs + ([aggregate] if aggregate is not None else []) , sort=False)             
        aggregate = ndarray_groupby_aggregate(df, ndarray_cols=ndarray_cols, aggregate_cols=aggregate_cols,
                                    value_cols=value_cols,  sample_cols=sample_cols, preset="sum")                
        return aggregate.reset_index()
    
    aggregate = None
    cur = []
    count = 0 
    for i in in_stream:
        cur.append(i)
        count += len(i)
        if count > chunksize:
            aggregate = agg_one(cur, aggregate)
            cur = []
            count = 0 
    if len(cur) > 0:
        aggregate = agg_one(cur, aggregate)
        
    if divide_by_count is not False:
        if divide_by_count is True:
            divide_by_count = "count"
        for i in ndarray_cols + value_cols_orig:
            aggregate[i] = aggregate[i] / aggregate[divide_by_count]
    
    return aggregate





            

                             
def kabsch_rmsd(P, Q):
    """
    Calculates RMSD between two vectors using Kabash alcorithm 
    Borrowed from https://github.com/charnley/rmsd  with some changes 
    
    rmsd is licenced with  a 2-clause BSD licence 
    
    Copyright (c) 2013, Jimmy Charnley Kromann <jimmy@charnley.dk> & Lars Bratholm
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
    ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
    ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    
    """
    P = P - np.mean(P, axis=0)
    Q = Q - np.mean(Q, axis=0)    

    C = np.dot(np.transpose(P), Q)

    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)
    
    dist = np.mean((np.dot(P, U) - Q)**2)  * 3

    return dist

