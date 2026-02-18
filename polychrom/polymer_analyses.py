# Code written by: Maksim Imakaev (imakaev@mit.edu)
"""
Analyses of polymer conformations
------------------------------


This module presents a collection of utils to work with polymer conformations.


Tools for calculating contacts
------------------------------

The main function calculating contacts is: :py:func:`polychrom.polymer_analyses.calculate_contacts`
Right now it is a simple wrapper around scipy.KDTree.

Another function :py:func:`polychrom.polymer_analyses.smart_contacts` was added recently to help build contact maps
with a large contact radius. It randomly sub-samples the monomers; by default selecting N/cutoff monomers. It then
calculates contacts from sub-sampled monomers only. It is especially helpful when the same code needs to calculate
contacts at large and small contact radii.Because of sub-sampling at large contact radius, it avoids the problem of
having way-too-many-contacts at a large contact radius. For ordinary contacts, the number of contacts scales as
contact_radius^3; however, with smart_contacts it would only scale linearly with contact radius, which leads to
significant speedsups.


Tools to calculate P(s) and R(s)
--------------------------------

We provide functions to calculate P(s), Rg^2(s) and R^2(s) for polymers. By default, they use  log-spaced bins on the
X axis, with about 10 bins per order of magnitude, but aligned such that the last bins ends exactly at (N-1). They
output (bin, scaling) for Rg^2 and R^2, and (bin_mid, scaling) for contacts. In either case, the returned values are
ready to plot. The difference is that Rg and R^2 are evaluated at a given value of s, while contacts are aggregated
for (bins[0].. bins[1]), (bins[1]..bins[2]). Therefore, we have to return bin mids for contacts.

"""

from math import sqrt
from typing import Callable, List, Optional, Sequence, Tuple, Union
import warnings
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter1d

try:
    from . import _polymer_math  # type: ignore
except Exception:
    pass


def calculate_contacts(data: np.ndarray, cutoff: float = 1.7) -> np.ndarray:
    """Calculates contacts between points give the contact radius (cutoff)

    Parameters
    ----------
    data : Nx3 array
        Coordinates of points
    cutoff : float , optional
        Cutoff distance (contact radius)

    Returns
    -------
    k by 2 array of contacts. Each row corresponds to a contact.
    """
    if data.shape[1] != 3:
        raise ValueError("Incorrect polymer data shape. Must be Nx3.")

    if np.isnan(data).any():
        raise RuntimeError("Data contains NANs")

    tree = KDTree(data)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")
    return pairs


def smart_contacts(
    data: np.ndarray,
    cutoff: float = 1.7,
    min_cutoff: float = 2.1,
    percent_func: Callable[[float], float] = lambda x: 1 / x,
) -> np.ndarray:
    """Calculates contacts for a polymer, give the contact radius (cutoff)
    This method takes a random fraction of the monomers that is equal to (
    1/cutoff).

    This is done to make contact finding faster, and because if cutoff radius
    is R, and monomer (i,j) are in contact, then monomers (i+a), and (j+b)
    are likely in contact if |a| + |b| <~ R  (the polymer could not run away
    by more than R in R steps)

    This method will have # of contacts grow approximately linearly with
    contact radius, not cubically, which should drastically speed up
    computations of contacts for large (5+) contact radii. This should allow
    using the same code both for small and large contact radius, without the
    need to reduce the # of conformations, subsample the data, or both at
    very large contact radii.


    Parameters
    ----------
    data : Nx3 array
        Polymer coordinates
    cutoff : float , optional
        Cutoff distance that defines contact
    min_cutoff : float, optional
        Apply the "smart" reduction of contacts only when cutoff
        is less than this value
    percent_func : callable, optional
        Function that calculates fraction of monomers to use, as a function of cutoff
        Default is 1/cutoff

    Returns
    -------
    k by 2 array of contacts. Each row corresponds to a contact.
    """
    if data.shape[1] != 3:
        raise ValueError("Incorrect polymer data shape. Must be Nx3.")

    if np.isnan(data).any():
        raise RuntimeError("Data contains NANs")

    if cutoff > min_cutoff:
        frac = percent_func(cutoff)
        inds = np.nonzero(np.random.random(len(data)) < frac)[0]

        conts = calculate_contacts(data[inds], cutoff)
        conts[:, 0] = inds[conts[:, 0]]
        conts[:, 1] = inds[conts[:, 1]]
        return conts

    else:
        return calculate_contacts(data, cutoff)


def generate_bins(N: int, start: int = 4, bins_per_order_magn: int = 10) -> np.ndarray:
    lstart = np.log10(start)
    lend = np.log10(N - 1) + 1e-6
    num = int(np.ceil((lend - lstart) * bins_per_order_magn))
    bins = np.unique(np.logspace(lstart, lend, dtype=int, num=max(num, 0)))
    if len(bins) > 0:
        assert bins[-1] == N - 1
    return bins


def contact_scaling(
    data: np.ndarray,
    bins0: Optional[Union[np.ndarray, Sequence[int]]] = None,
    cutoff: float = 1.1,
    *,
    ring: bool = False,
) -> Tuple[List[float], np.ndarray]:
    """
    Returns contact probability scaling for a given polymer conformation
    Contact between monomers X and X+1 is counted as s=1


    Parameters
    ----------
    data : Nx3 array of ints/floats
        Input polymer conformation
    bins0 : list or None
        Bins to calculate scaling.
        Bins should probably be log-spaced; log-spaced bins can be quickly
        calculated using mirnylib.numtuis.logbinsnew.
        If None, bins will be calculated automatically
    cutoff : float, optional
        Cutoff to calculate scaling
    ring : bool, optional
        If True, will calculate contacts for the ring

    Returns
    -------
    (mids, contact probabilities) where "mids" contains
    geometric means of bin start/end


    """
    data = np.asarray(data)
    N = data.shape[0]
    assert data.shape[1] == 3

    if bins0 is None:
        bins0 = generate_bins(N)

    bins0 = np.array(bins0)
    bins = [(bins0[i], bins0[i + 1]) for i in range(len(bins0) - 1)]
    contacts = np.array(calculate_contacts(data, cutoff))

    contacts = contacts[:, 1] - contacts[:, 0]  # contact lengthes

    if ring:
        mask = contacts > N // 2
        contacts[mask] = N - contacts[mask]

    scontacts = np.sort(contacts)  # sorted contact lengthes
    # count of contacts
    connumbers = np.diff(np.searchsorted(scontacts, bins0, side="left"))

    if ring:
        possible = np.diff(N * bins0)
    else:
        possible = np.diff(N * bins0 + 0.5 * bins0 - 0.5 * (bins0**2))

    connumbers = connumbers / possible

    a = [sqrt(i[0] * (i[1] - 1)) for i in bins]
    return a, connumbers


def _smooth_for_slope(x: np.ndarray, sigma: float) -> np.ndarray:
    """Helper function for slope_contact_scaling to smooth data.
    Extracted to module level for pickle compatibility."""
    return gaussian_filter1d(x, sigma)


def slope_contact_scaling(
    mids: Union[np.ndarray, List[float]], cp: Union[np.ndarray, List[float]], sigma: float = 2.0
) -> Tuple[np.ndarray, np.ndarray]:
    # P(s) has to be smoothed in logspace, and both P and s have to be smoothed.
    # It is discussed in detail here
    # https://gist.github.com/mimakaev/4becf1310ba6ee07f6b91e511c531e73

    # Values sigma=1.5-2 look reasonable for reasonable simulations

    slope = np.diff(_smooth_for_slope(np.log(cp), sigma)) / np.diff(_smooth_for_slope(np.log(mids), sigma))

    return np.array(mids[1:]), np.array(slope)


def _radius_gyration_helper(len2: int, coms: np.ndarray, coms2: np.ndarray, ring: bool = False) -> float:
    """Helper function for Rg2_scaling to calculate radius of gyration.
    Extracted to module level for pickle compatibility.

    Parameters
    ----------
    len2 : int
        Length of the subchain
    coms : ndarray
        Cumulative sum of locations to calculate COM
    coms2 : ndarray
        Cumulative sum of locations^2 to calculate RG
    ring : bool
        Whether to treat polymer as a ring
    """
    if ring:
        comsadd = coms[1:len2, :].copy()
        coms2add = coms2[1:len2, :].copy()
        comsadd += coms[-1, :][None, :]
        coms2add += coms2[-1, :][None, :]
        comsw = np.concatenate([coms, comsadd], axis=0)
        coms2w = np.concatenate([coms2, coms2add], axis=0)
    else:
        comsw = coms
        coms2w = coms2

    coms2d = (-coms2w[:-len2, :] + coms2w[len2:, :]) / len2
    comsd = ((comsw[:-len2, :] - comsw[len2:, :]) / len2) ** 2
    diffs = coms2d - comsd
    sums = np.sum(diffs, 1)
    return np.mean(sums)


def Rg2_scaling(
    data: np.ndarray, bins: Optional[Union[np.ndarray, Sequence[int]]] = None, ring: bool = False
) -> Tuple[np.ndarray, List[float]]:
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
    coms2 = np.cumsum(data**2, 0)  # cumulative sum of locations^2 to calculate RG

    rads = [0.0 for _ in range(len(bins))]
    for i in range(len(bins)):
        rads[i] = _radius_gyration_helper(int(bins[i]), coms, coms2, ring=ring)
    return np.array(bins), rads


def R2_scaling(
    data: np.ndarray, bins: Optional[Union[np.ndarray, Sequence[int]]] = None, ring: bool = False
) -> Tuple[np.ndarray, List[float]]:
    """
    Returns end-to-end distance scaling of a given polymer conformation.
    ..warning:: This method averages end-to-end scaling over all possible
     subchains of given length

    Parameters
    ----------

    data: Nx3 array
    bins: the same as in giveCpScaling
    ring: is the polymer a ring?

    """
    data = np.asarray(data, float)
    N = data.shape[0]
    assert data.shape[1] == 3
    data = data.T

    if bins is None:
        bins = generate_bins(N)
    if ring:
        data = np.concatenate([data, data], axis=1)

    rads = [0.0 for _ in range(len(bins))]
    for i in range(len(bins)):
        length = bins[i]
        if ring:
            rads[i] = np.mean((np.sum((data[:, :N] - data[:, length : length + N]) ** 2, 0)))
        else:
            rads[i] = np.mean((np.sum((data[:, :-length] - data[:, length:]) ** 2, 0)))
    return np.array(bins), rads


def Rg2(data: np.ndarray) -> float:
    """
    Simply calculates gyration radius of a polymer chain.
    """
    data = np.asarray(data)
    assert data.shape[1] == 3
    return np.mean((data - np.mean(data, axis=0)) ** 2) * 3


def Rg2_matrix(data: np.ndarray) -> np.ndarray:
    """
    Uses dynamic programming and vectorizing to calculate Rg for each subchain of the polymer.
    Returns a matrix for which an element [i,j] is Rg of a subchain from i to j including  i and j
    """

    data = np.asarray(data, float)
    assert data.shape[1] == 3
    N = data.shape[0]
    data = np.concatenate([[[0, 0, 0]], data])

    coms = np.cumsum(data, 0)  # cumulative sum of locations to calculate COM
    coms2 = np.cumsum(data**2, 0)  # cumulative sum of locations^2 to calculate RG

    dists = np.abs(np.arange(N)[:, None] - np.arange(N)[None, :]) + 1
    coms2d = (-coms2[:-1, None, :] + coms2[None, 1::, :]) / dists[:, :, None]
    comsd = ((coms[:-1, None, :] - coms[None, 1:, :]) / dists[:, :, None]) ** 2
    sums = np.sum(coms2d - comsd, 2)
    np.fill_diagonal(sums, 0)
    mask = np.arange(N)[:, None] > np.arange(N)[None, :]
    sums[mask] = sums.T[mask]
    return sums


def kabsch_msd(P: np.ndarray, Q: np.ndarray) -> float:
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


def mutualSimplify(a: np.ndarray, b: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simplify two polymer rings while preserving their mutual topology.

    This function performs topology-preserving simplification of two interlinked polymer
    rings simultaneously. It reduces the number of monomers in both polymers while
    maintaining their linking number and individual knot types.

    The algorithm works by iteratively attempting to remove monomers from each polymer.
    A monomer can be removed if doing so doesn't change the linking between the two
    polymers. This is checked by verifying that no segments from either polymer pass
    through the triangle formed by removing the monomer.

    Parameters
    ----------
    a : np.ndarray
        First polymer ring as an Nx3 array of 3D coordinates.
    b : np.ndarray
        Second polymer ring as an Mx3 array of 3D coordinates.
    verbose : bool, optional
        If True, print progress during simplification. Default is False.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Simplified versions of both input polymers that preserve their
        mutual topology (linking number and individual knot types).

    Examples
    --------
    >>> # Create two linked rings
    >>> ring1 = create_ring(center=[0, 0, 0], radius=1, n_points=100)
    >>> ring2 = create_ring(center=[0.5, 0, 0], radius=1, n_points=100)
    >>> simp1, simp2 = mutualSimplify(ring1, ring2)
    >>> # The simplified rings will have the same linking number

    Notes
    -----
    - The simplification alternates between the two polymers to ensure balanced reduction
    - Small random perturbations are added internally to avoid numerical degeneracies
    - The function is particularly useful before calculating linking numbers, as it can
      dramatically reduce computation time
    - Originally ported from openmmlib

    See Also
    --------
    simplifyPolymer : Simplify a single polymer ring
    getLinkingNumber : Calculate the linking number between two rings
    """
    if verbose:
        print("Starting mutual simplification of polymers")
    while True:
        la, lb = len(a), len(b)
        if verbose:
            print(len(a), len(b), "before; ", end=" ")
        a, b = _polymer_math.mutualSimplify(a, b)  # type: ignore
        if verbose:
            print(len(a), len(b), "after one; ", end=" ")
        b, a = _polymer_math.mutualSimplify(b, a)  # type: ignore
        if verbose:
            print(len(a), len(b), "after two; ")

        if (len(a) == la) and (len(b) == lb):
            if verbose:
                print("Mutual simplification finished")
            return a, b


def getLinkingNumber(
    data1: np.ndarray, data2: np.ndarray, simplify: bool = True, randomOffset: bool = True, verbose: bool = False
) -> int:
    """
    Calculate the linking number between two closed polymer rings.

    The linking number is a topological invariant that measures how many times
    two closed curves wind around each other. It is always an integer and remains
    constant under continuous deformations that don't break the curves.

    The algorithm computes the linking number using the Gauss linking integral,
    counting signed crossings when one curve is projected onto a plane perpendicular
    to segments of the other curve.

    Parameters
    ----------
    data1 : np.ndarray
        First polymer ring as an Nx3 array of 3D coordinates.
    data2 : np.ndarray
        Second polymer ring as an Mx3 array of 3D coordinates.
    simplify : bool, optional
        If True, simplify both polymers before calculating linking number.
        This can dramatically speed up the calculation. Default is True.
    randomOffset : bool, optional
        If True, add small random perturbations to avoid numerical degeneracies
        when polymer segments are exactly coplanar. Default is True.
    verbose : bool, optional
        If True, print progress information during calculation. Default is False.

    Returns
    -------
    int
        The linking number between the two polymer rings.
        Positive values indicate right-handed linking, negative for left-handed.

    Examples
    --------
    >>> # Create two unlinked rings
    >>> ring1 = create_ring([0, 0, 0], radius=1)
    >>> ring2 = create_ring([3, 0, 0], radius=1)  # far apart
    >>> L = getLinkingNumber(ring1, ring2)
    >>> print(L)  # Should be 0
    0

    >>> # Create Hopf link (two linked rings)
    >>> ring1 = create_ring([0, 0, 0], radius=1, axis='z')
    >>> ring2 = create_ring([0.5, 0, 0], radius=1, axis='x')
    >>> L = getLinkingNumber(ring1, ring2)
    >>> print(abs(L))  # Should be 1
    1

    Notes
    -----
    - The polymers must be closed rings (the last point connects to the first)
    - The sign of the linking number depends on the orientation of the curves
    - Simplification is highly recommended for long polymers to reduce computation time
    - The linking number is undefined for open chains
    - Originally ported from openmmlib

    See Also
    --------
    mutualSimplify : Simplify two polymers while preserving their linking
    simplifyPolymer : Simplify a single polymer ring
    """
    if simplify:
        data1, data2 = mutualSimplify(a=data1, b=data2, verbose=verbose)
    return _polymer_math.getLinkingNumber(data1, data2, randomOffset=randomOffset)  # type: ignore


def simplifyPolymer(data: np.ndarray, verbose: bool = False) -> np.ndarray:
    """
    Simplify a polymer ring while preserving its topology.

    This function uses a topology-preserving simplification algorithm to reduce the number
    of monomers in a polymer ring while maintaining its knot type. The algorithm iteratively
    removes monomers that can be deleted without changing the topology by checking for
    intersections with the remaining polymer segments.

    The algorithm works by:
    1. Testing each monomer to see if it can be removed
    2. Checking if removing it would cause any segment intersections
    3. If no intersections, replacing the monomer with the midpoint of its neighbors
    4. Repeating until no more simplifications are possible

    This is particularly useful for:
    - Speeding up topological calculations like Alexander polynomial
    - Reducing computational cost of linking number calculations
    - Visualizing complex knots with fewer segments

    Parameters
    ----------
    data : np.ndarray
        Nx3 array of polymer coordinates representing a closed ring.
        The polymer is assumed to be a closed loop (first and last points are connected).
    verbose : bool, optional
        If True, print simplification progress. Default is False.

    Returns
    -------
    np.ndarray
        Simplified polymer coordinates with shape (M, 3) where M <= N.
        The simplified polymer has the same topology as the input.

    Examples
    --------
    >>> # Simplify a complex knot for faster analysis
    >>> polymer = np.random.randn(1000, 3)
    >>> simplified = simplifyPolymer(polymer)
    >>> print(f"Reduced from {len(polymer)} to {len(simplified)} monomers")

    Notes
    -----
    - The function adds small random perturbations to avoid numerical degeneracies
    - The simplification is deterministic up to the random perturbations
    - For unknotted polymers, the result will typically be very short (3-4 monomers)
    - For complex knots, the simplified length depends on the knot complexity
    """
    try:


        if len(data) < 3:
            raise ValueError("Polymer must have at least 3 monomers")

        if data.shape[1] != 3:
            raise ValueError("Data must be Nx3 array of 3D coordinates")

        if verbose:
            print(f"Simplifying polymer with {len(data)} monomers...")

        result = _polymer_math.simplifyPolymer(data)  # type: ignore

        if verbose:
            print(f"Simplified to {len(result)} monomers")

        return result

    except ImportError:
        warnings.warn(
            "C++ simplification module not available. " "Please compile the Cython extensions.", RuntimeWarning
        )
        return data


def calculate_cistrans(
    data: np.ndarray,
    chains: Optional[List[List[int]]],
    chain_id: int = 0,
    cutoff: float = 5.0,
    pbc_box: bool = False,
    box_size: Optional[Union[List[float], np.ndarray]] = None,
) -> Tuple[int, int]:
    """
    Analysis of the territoriality of polymer chains from simulations, using the cis/trans ratio.
    Cis signal is computed for the marked chain ('chain_id') as amount of contacts of the chain with itself
    Trans signal is the total amount of trans contacts for the marked chain with other chains from 'chains'
    (and with all the replicas for 'pbc_box'=True)

    """
    if data.shape[1] != 3:
        raise ValueError("Incorrect polymer data shape. Must be Nx3.")

    if np.isnan(data).any():
        raise RuntimeError("Data contains NANs")

    N = len(data)

    if pbc_box:
        if box_size is None:
            raise ValueError("Box size is not given")
        else:
            data_scaled = np.mod(data, box_size)

    else:
        box_size = None
        data_scaled = np.copy(data)

    if chains is None:
        chains = [[0, N]]
        chain_id = 0

    chain_start = chains[chain_id][0]
    chain_end = chains[chain_id][1]

    # all contact pairs available in the scaled data
    tree = KDTree(data_scaled, boxsize=box_size)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")

    # total number of contacts of the marked chain:
    # each contact is counted twice if both monomers belong to the marked chain and
    # only once if just one of the monomers in the pair belong to the marked chain
    all_signal = len(pairs[pairs < chain_end]) - len(pairs[pairs < chain_start])

    # contact pairs of the marked chain with itself
    tree = KDTree(data[chain_start:chain_end], boxsize=None)
    pairs = tree.query_pairs(cutoff, output_type="ndarray")

    # doubled number of contacts of the marked chain with itself (i.e. cis signal)
    cis_signal = 2 * len(pairs)

    assert all_signal >= cis_signal

    trans_signal = all_signal - cis_signal

    return cis_signal, trans_signal


def rotation_matrix(rotate):
    """Calculates rotation matrix based on three rotation angles"""
    tx, ty, tz = rotate
    Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, -np.sin(ty)], [0, 1, 0], [np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0, 0, 1]])
    return np.dot(Rx, np.dot(Ry, Rz))
