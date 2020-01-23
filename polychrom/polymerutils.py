"""
Loading polychrom trajectories
==============================


The module :py:mod:`polychrom.polymerutils` provides tools for saving and loading individual 
conformations. Note that saving trajectories should be done using :py:mod:`polychrom.hdf5_format` module. 
This module provides tools for loading/saving invividual conformations, or for working with 
projects that have both  old-style and new-style trajectories. 

For projects using both old-style and new-style trajectories(e.g. in a project that was
switched to polychrom, and new files were added), a function :py:func:`polychrom.polymerutils.fetch_block`
can be helpful as it provides the same interface for fetching a conformation from both 
old-style and new-style trajectory. Note however that it is not the fastest way to iterate over conformations
in the new-style trajectory, and the :py:func:`polychrom.hdf5_format.list_URIs` is faster. 

A typical workflow with the new-style trajectories should be: 
``URIs = polychrom.hdf5_format.list_URIs(folder)
for URI in URIs:
    data = polychrom.hdf5_format.load_URI(URI)
    xyz = data["pos"] 
``
"""


from __future__ import absolute_import, division, print_function, unicode_literals
import six
import numpy as np
import os

from . import hdf5_format
from polychrom.hdf5_format import load_URI
import joblib
import glob

import io


def load(filename):
    """Universal load function for any type of data file It always returns just XYZ
    positions - use fetch_block or hdf5_format.load_URI for loading the whole metadata
    
    Accepted file types
    -------------------
    
    New-style URIs (HDF5 based storage)
    
    Text files in openmm-polymer format
    joblib files in openmm-polymer format 
    
    Parameters
    ----------
    
    filename: str 
        filename to load or a URI

    """
    if "::" in filename:
        return hdf5_format.load_URI(filename)["pos"]

    if not os.path.exists(filename):
        raise IOError("File not found :( \n %s" % filename)

    try:  # loading from a joblib file here
        return dict(joblib.load(filename)).pop("data")
    except:  # checking for a text file
        data_file = open(filename)
        line0 = data_file.readline()
        try:
            N = int(line0)
        except (ValueError, UnicodeDecodeError):
            raise TypeError("Could not read the file. Not text or joblib.")
        data = [list(map(float, i.split())) for i in data_file.readlines()]

        if len(data) != N:
            raise ValueError("N does not correspond to the number of lines!")
        return np.array(data)


def fetch_block(folder, ind, full_output=False):
    """
    A more generic function to fetch block number "ind" from a trajectory in a folder
    
    
    This function is useful both if you want to load both "old style" trajectories (block1.dat), 
    and "new style" trajectories ("blocks_1-50.h5")
    
    It will be used in files "show" 
    
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
    blocksh5 = glob.glob(os.path.join(folder, "blocks*.h5"))
    blocksdat = glob.glob(os.path.join(folder, "block*.dat"))
    ind = int(ind)
    if (len(blocksh5) > 0) and (len(blocksdat) > 0):
        raise ValueError("both .h5 and .dat files found in folder - exiting")
    if (len(blocksh5) == 0) and (len(blocksdat) == 0):
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
            block = block["pos"]

    if len(blocksdat) > 0:
        block = load(os.path.join(folder, f"block{ind}.dat"))
    return block


def save(data, filename, mode="txt", pdbGroups=None):
    """
    Basically unchanged polymerutils.save function from openmm-polymer
    
    It can save into txt or joblib formats used by old openmm-polymer
    
    It is also very useful for saving files to PDB format to make them compatible
    with nglview, pymol_show and others
    """
    data = np.asarray(data, dtype=np.float32)

    if mode.lower() == "joblib":
        joblib.dump({"data": data}, filename=filename, compress=9)
        return

    if mode.lower() == "txt":
        lines = [str(len(data)) + "\n"]

        for particle in data:
            lines.append("{0:.3f} {1:.3f} {2:.3f}\n".format(*particle))
        if filename == None:
            return lines

        elif isinstance(filename, six.string_types):
            with open(filename, "w") as myfile:
                myfile.writelines(lines)
        elif hasattr(filename, "writelines"):
            filename.writelines(lines)
        else:
            raise ValueError("Not sure what to do with filename {0}".format(filename))

    elif mode == "pdb":
        data = (
            data - np.minimum(np.min(data, axis=0), np.zeros(3, float) - 100)[None, :]
        )
        retret = ""

        def add(st, n):
            if len(st) > n:
                return st[:n]
            else:
                return st + " " * (n - len(st))

        if pdbGroups == None:
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
        raise ValueError("Unknown mode : %s, use h5dict, joblib, txt or pdb" % mode)


def rotation_matrix(rotate):
    """Calculates rotation matrix based on three rotation angles"""
    tx, ty, tz = rotate
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]]
    )
    Ry = np.array(
        [[np.cos(ty), 0, -np.sin(ty)], [0, 1, 0], [np.sin(ty), 0, np.cos(ty)]]
    )
    Rz = np.array(
        [[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0, 0, 1]]
    )
    return np.dot(Rx, np.dot(Ry, Rz))
