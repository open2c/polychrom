"""
Loading and saving individual conformations
===========================================


The module :py:mod:`polychrom.polymerutils` provides tools for saving and loading individual conformations. Note that
saving and loading trajectories should generally be done using :py:mod:`polychrom.hdf5_format` module. This module
provides tools for loading/saving invividual conformations, or for working with projects that have both  old-style
and new-style trajectories.

For projects using both old-style and new-style trajectories(e.g. in a project that was switched to polychrom,
and new files were added), a function :py:func:`polychrom.polymerutils.fetch_block` can be helpful as it provides the
same interface for fetching a conformation from both old-style and new-style trajectory. Note however that it is not
the fastest way to iterate over conformations in the new-style trajectory, and the
:py:func:`polychrom.hdf5_format.list_URIs` is faster.

A typical workflow with the new-style trajectories should be:

.. code-block:: python

    URIs = polychrom.hdf5_format.list_URIs(folder)
    for URI in URIs:
        data = polychrom.hdf5_format.load_URI(URI)
        xyz = data["pos"]
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import warnings

import numpy as np

from polychrom.hdf5_format import load_URI

from . import hdf5_format


def load(filename):
    """
    A function to load a single conformation from a URI. Deprecated.
    Use load_URI from hdf5_format instead.

    Parameters
    ----------

    filename: str
        filename to load or a URI

    """
    warnings.warn("polymerutils.load is deprecated. Use hdf5_format.load_URI instead.", DeprecationWarning)

    if "::" in filename:
        return hdf5_format.load_URI(filename)["pos"]

    raise ValueError("Only URIs are supported in this version of polychrom")


def fetch_block(folder, ind, full_output=False):
    """
    A function to fetch a single block from a folder with a new-style trajectory.
    Old-style trajectories are deprecated.

    Parameters
    ----------

        folder: str, folder with a trajectory

        ind: str or int, number of a block to fetch

        full_output: bool (default=False)
            If set to true, outputs a dict with positions, eP, eK, time etc.
            if False, outputs just the conformation
            (relevant only for new-style URIs, so default is False)

    Returns
    -------
        data, Nx3 numpy array

        if full_output==True, then dict with data and metadata; XYZ is under key "pos"
    """
    warnings.warn(
        "fetch_block is deprecated. Use hdf5_format.list_uris followed by hdf5_format.load_URI instead.",
        DeprecationWarning,
    )

    blocksh5 = glob.glob(os.path.join(folder, "blocks*.h5"))
    ind = int(ind)
    if len(blocksh5) == 0:
        raise ValueError("no blocks found")

    if len(blocksh5) > 0:
        fnames = [os.path.split(i)[-1] for i in blocksh5]
        inds = [i.split("_")[-1].split(".")[0].split("-") for i in fnames]
        exists = [(int(i[0]) <= ind) and (int(i[1]) >= ind) for i in inds]

        if True not in exists:
            raise ValueError(f"block {ind} not found in files")
        if exists.count(True) > 1:
            raise ValueError("Cannot find the file uniquely: names are wrong")
        pos = exists.index(True)
        block = load_URI(blocksh5[pos] + f"::{ind}")
        if not full_output:
            return block["pos"]
        return block

    raise ValueError(f"Cannot find the block {ind} in the folder {folder}")


def save(data, filename, mode="txt", pdbGroups=None):
    """
    A legacy function, currently only kept for compatibility with PDB saving that is rarely used.
    """
    warnings.warn("polymerutils.save is deprecated. Will be moved to legacy", DeprecationWarning)

    data = np.asarray(data, dtype=np.float32)

    if mode == "pdb":
        data = data - np.minimum(np.min(data, axis=0), np.zeros(3, float) - 100)[None, :]
        retret = ""

        def add(st, n):
            if len(st) > n:
                return st[:n]
            else:
                return st + " " * (n - len(st))

        if pdbGroups is None:
            pdbGroups = ["A" for i in range(len(data))]
        else:
            pdbGroups = [str(int(i)) for i in pdbGroups]

        for i, line, group in zip(list(range(len(data))), data, pdbGroups):
            atomNum = (i + 1) % 9000
            segmentNum = (i + 1) // 9000 + 1
            line = [float(j) for j in line]
            ret = add("ATOM", 6)
            ret = add(ret + "{:5d}".format(atomNum), 11)
            ret = ret + " "
            ret = add(ret + "CA", 17)
            ret = add(ret + "ALA", 21)
            ret = add(ret + group[0], 22)
            ret = add(ret + str(atomNum), 26)
            ret = add(ret + "         ", 30)
            # ret = add(ret + "%i" % (atomNum), 30)
            ret = add(ret + ("%8.3f" % line[0]), 38)
            ret = add(ret + ("%8.3f" % line[1]), 46)
            ret = add(ret + ("%8.3f" % line[2]), 54)
            ret = add(ret + (" 1.00"), 61)
            ret = add(ret + str(float(i % 8 > 4)), 67)
            ret = add(ret, 73)
            ret = add(ret + str(segmentNum), 77)
            retret += ret + "\n"
        with open(filename, "w") as f:
            f.write(retret)
            f.flush()
    elif mode == "pyxyz":
        with open(filename, "w") as f:
            for i in data:
                filename.write("C {0} {1} {2}".format(*i))

    else:
        raise ValueError(f"Unknown mode {mode}. Only 'pdb' and 'pyxyz' are supported.")


def rotation_matrix(rotate):
    warnings.warn("rotation_matrix will be moved to polymer_analyses", DeprecationWarning, stacklevel=2)
    from polychrom.polymer_analyses import rotation_matrix as rm
    return rm(rotate)
