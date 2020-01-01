# Code written by: Maksim Imakaev (imakaev@mit.edu)

from . import _polymer_math
from math import sqrt

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
    lend = np.log10(N - 1) + 1e-6
    num = np.ceil((lend - lstart) * bins_per_order_magn)
    bins = np.unique(np.logspace(lstart, lend, dtype=int, num=num))
    assert bins[-1] == N - 1
    return bins


def contact_scaling(
        data, bins0=None, cutoff=1.1, integrate=False,
        ring=False,
        bins_per_order_magn=10):
    """
    Returns contact probability scaling for a given polymer conformation
    Contact between monomers X and X+1 is counted as s=1 
    

    Parameters
    ----------
    data : Nx3 array of ints/floats
        Input polymer conformation
    bins0 : list or None
        Bins to calculate scaling.
        Bins should probably be log-spaced; log-spaced bins can be quickly calculated using mirnylib.numtuis.logbinsnew.
        If None, bins will be calculated automatically
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
    (mids, contact probabilities) where "mids" contains geometric means of bin start/end 
    

    """
    data = np.asarray(data)
    N = data.shape[0]
    assert data.shape[1] == 3

    if bins0 is None:
        bins0 = generate_bins(N, bins_per_order_magn=bins_per_order_magn)

    bins0 = np.array(bins0)
    bins = [(bins0[i], bins0[i + 1]) for i in range(len(bins0) - 1)]
    contacts = np.array(calculate_contacts(data, cutoff))

    contacts = contacts[:, 1] - contacts[:, 0]  # contact lengthes

    if ring == True:
        mask = contacts > N // 2
        contacts[mask] = N - contacts[mask]

    scontacts = np.sort(contacts)  # sorted contact lengthes
    connumbers = np.diff(np.searchsorted(scontacts, bins0, side="left"))  # count of contacts

    if ring == True:
        possible = np.diff(N * bins0)
    else:
        possible = np.diff(N * bins0 + 0.5 * bins0 - 0.5 * (bins0 ** 2))

    connumbers = connumbers / possible

    a = [sqrt(i[0] * (i[1] - 1)) for i in bins]
    return (a, connumbers)


def R2_scaling(data, bins=None, ring=False):
    """
    Returns end-to-end distance scaling of a given polymer conformation.
    ..warning:: This method averages end-to-end scaling over all possible subchains of given length

    Parameters
    ----------

    data: Nx3 array
    bins: the same as in giveCpScaling

    """
    data = np.asarray(data, float)
    N = data.shape[0]
    assert data.shape[1] == 3
    data = data.T

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


def Rg2(data):
    """
    Simply calculates gyration radius of a polymer chain.
    """
    data = np.asarray(data)
    assert data.shape[1] == 3
    return np.mean((data - np.mean(data, axis=0)) ** 2) * 3


def Rg2_matrix(data):
    """
    Uses dynamic programming and vectorizing to calculate Rg for each subchain of the matrix 
    element [i,j] in resulting matrix is Rg of a subchain from i to j including  i and j 
    """

    data = np.asarray(data, float)
    assert data.shape[1] == 3
    N = data.shape[0]
    data = np.concatenate([[[0, 0, 0]], data])

    coms = np.cumsum(data, 0)  # cumulative sum of locations to calculate COM
    coms2 = np.cumsum(data ** 2, 0)  # cumulative sum of locations^2 to calculate RG

    dists = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) + 1
    coms2d = (-coms2[:-1, None, :] + coms2[None, 1::, :]) / dists[:, :, None]
    comsd = ((coms[:-1, None, :] - coms[None, 1:, :]) / dists[:, :, None]) ** 2
    sums = np.sum(coms2d - comsd, 2)
    np.fill_diagonal(sums, 0)
    mask = np.arange(N)[:, None] > np.arange(N)[None, :]
    sums[mask] = sums.T[mask]
    return sums


def Rg2_scaling(data, bins=None, ring=False):
    """Calculates average gyration radius of subchains a function of s
    
    Parameters
    ----------
    
    data: Nx3 array
    bins: subchain lengths at which to calculate Rg
    ring: treat polymer as a ring (default: False) 
    """

    data = np.asarray(data, float)
    N = data.shape[0]
    assert data.shape[1] == 3

    data = np.concatenate([[[0, 0, 0]], data])

    if bins is None:
        bins = generate_bins(N)

    coms = np.cumsum(data, 0)  # cumulative sum of locations to calculate COM
    coms2 = np.cumsum(data ** 2, 0)  # cumulative sum of locations^2 to calculate RG

    def radius_gyration(len2):
        data
        if ring == True:
            comsadd = coms[1:len2, :].copy()
            coms2add = coms2[1:len2, :].copy()
            comsadd += coms[-1, :][None, :]
            coms2add += coms2[-1, :][None, :]
            comsw = np.concatenate([coms, comsadd], axis=0)
            coms2w = np.concatenate([coms2, coms2add], axis=0)
        else:
            comsw = coms
            coms2w = coms2

        coms2d = (-coms2w[:-len2, :] + coms2w[len2:, :]) / (len2)
        comsd = ((comsw[:-len2, :] - comsw[len2:, :]) / (len2)) ** 2
        diffs = coms2d - comsd
        sums = (np.sum(diffs, 1))
        return np.mean(sums)

    rads = [0. for i in range(len(bins))]
    for i in range(len(bins)):
        rads[i] = radius_gyration(int(bins[i]))
    return (np.array(bins), rads)


def _test_Rg_scalings():
    a = np.random.lognormal(1, 1, size=(30, 3))  # array for testing

    gr = Rg2_matrix(a)  # calculate Rg matrix in a normal way 

    for i in range(len(a) + 1):  # fill on eside of it with manually calculated Rg(i:j)
        for j in range(i + 1, len(a)):
            gr[j, i] = Rg2(a[i:j + 1])
            pass

    assert np.allclose(gr, gr.T)

    scal = Rg2_scaling(a, bins=[5])  # 5th diagonal here means s=5 (5-monomer chains)
    d1 = np.diagonal(gr, 4).mean()  # here Nth diagonal means N+1 monomer chain, so that the corner = whole chain    
    assert np.allclose(scal[1][0], d1)  # compare P(s) to manually calculated from Rg matrix 

    scal = Rg2_scaling(a, bins=[3], ring=True)  # now we are testing ring there are (N-s+1) subchains of length s. 
    d1 = (np.diagonal(gr, 2).sum() + Rg2(np.array([a[0], a[-1], a[-2]])) + Rg2(np.array([a[0], a[1], a[-1]]))) / len(a)
    assert np.allclose(scal[1][0], d1)
    # compare with manually calculated rg(i,j) plus Rg of two 3-monomer subchains crossing the boundary


def ndarray_groupby_aggregate(df, ndarray_cols, aggregate_cols, value_cols=[],
                              sample_cols=[], preset="sum",
                              ndarray_agg=lambda x: np.sum(x, axis=0), value_agg=lambda x: x.sum()):
    """
    A version of pd.groupby that is aware of numpy arrays as values of columns 
    
    It aggregates columns ndarray_cols using ndarray_agg aggregator
    It aggregates value_cols using value_agg aggregator
    It takes the first element in sample_cols
    and aggregates over aggregate_cols 
    
    It has presets for sum, mean and nanmean. 
    """

    if preset == "sum":
        ndarray_agg = lambda x: np.sum(x, axis=0)
        value_agg = lambda x: x.sum()
    elif preset == "mean":
        ndarray_agg = lambda x: np.mean(x, axis=0)
        value_agg = lambda x: x.mean()
    elif preset == "nanmean":
        ndarray_agg = lambda x: np.nanmean(x, axis=0)
        value_agg = lambda x: x.mean()

    def combine_values(in_df):
        "splits into ndarrays, 'normal' values, and samples; performs aggregation, and returns a Series"
        average_arrs = pd.Series(index=ndarray_cols,
                                 data=[ndarray_agg([np.asarray(j) for j in in_df[i].values]) for i in ndarray_cols])
        average_values = value_agg(in_df[value_cols])
        sample_values = in_df[sample_cols].iloc[0]
        agg_series = pd.concat([average_arrs, average_values, sample_values])
        return agg_series

    return df.groupby(aggregate_cols).apply(combine_values)


def streaming_ndarray_agg(in_stream, ndarray_cols, aggregate_cols, value_cols=[], sample_cols=[],
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

        df = pd.concat(dfs + ([aggregate] if aggregate is not None else []), sort=False)
        aggregate = ndarray_groupby_aggregate(df, ndarray_cols=ndarray_cols, aggregate_cols=aggregate_cols,
                                              value_cols=value_cols, sample_cols=sample_cols, preset="sum")
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


def kabsch_msd(P, Q):
    """
    Calculates MSD between two vectors using Kabash alcorithm 
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

    dist = np.mean((np.dot(P, U) - Q) ** 2) * 3

    return dist


kabsch_rmsd = kabsch_msd


def _test_Rg_scalings():
    a = np.random.lognormal(1, 1, size=(30, 3))  # array for testing

    gr = Rg2_matrix(a)  # calculate Rg matrix in a normal way 

    for i in range(len(a) + 1):  # fill on eside of it with manually calculated Rg(i:j)
        for j in range(i + 1, len(a)):
            gr[j, i] = Rg2(a[i:j + 1])
            pass

    assert np.allclose(gr, gr.T)

    scal = Rg2_scaling(a, bins=[5])  # 5th diagonal here means s=5 (5-monomer chains)
    d1 = np.diagonal(gr, 4).mean()  # here Nth diagonal means N+1 monomer chain, so that the corner = whole chain    
    assert np.allclose(scal[1][0], d1)  # compare P(s) to manually calculated from Rg matrix 

    scal = Rg2_scaling(a, bins=[3], ring=True)  # now we are testing ring there are (N-s+1) subchains of length s. 
    d1 = (np.diagonal(gr, 2).sum() + Rg2(np.array([a[0], a[-1], a[-2]])) + Rg2(np.array([a[0], a[1], a[-1]]))) / len(a)
    assert np.allclose(scal[1][0], d1)
    # compare with manually calculated rg(i,j) plus Rg of two 3-monomer subchains crossing the boundary


def mutualSimplify(a, b, verbose=False):
    if verbose:
        print("Starting mutual simplification of polymers")
    while True:
        la, lb = len(a), len(b)
        if verbose:
            print(len(a), len(b), "before; ", end=' ')
        a, b = _polymer_math.mutualSimplify(a, b)
        if verbose:
            print(len(a), len(b), "after one; ", end=' ')
        b, a = _polymer_math.mutualSimplify(b, a)
        if verbose:
            print(len(a), len(b), "after two; ")

        if (len(a) == la) and (len(b) == lb):
            if verbose:
                print("Mutual simplification finished")
            return a, b


def getLinkingNumber(data1, data2, simplify=True, randomOffset=True, verbose=False):
    if simplify:
        data1, data2 = mutualSimplify(a=data1, b=data2, verbose=verbose)
    return _polymer_math.getLinkingNumber(data1, data2, randomOffset=randomOffset)
