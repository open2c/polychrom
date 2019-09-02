from __future__ import absolute_import, division, print_function, unicode_literals
import six
import warnings
import numpy as np
import joblib
import os
from math import sqrt, sin, cos
import numpy

from . import hdf5_format
import scipy, scipy.stats  # @UnusedImport
from polychrom.hdf5_format import load_block
import numpy as np
import joblib
import gzip
import glob 

import io



def load(filename, h5dictKey=None):
    """Universal load function for any type of data file"""
    if "::" in filename:
        return hdf5_format.load_block(filename)["pos"]
        
    if not os.path.exists(filename):
        raise IOError("File not found :( \n %s" % filename)

    try: #loading from a joblib file here
        return dict(joblib.load(filename)).pop("data")
    except: #checking for a text file
        data_file = open(filename)
        line0 = data_file.readline()
        try:
            N = int(line0)
        except (ValueError, UnicodeDecodeError):
            raise TypeError("Could not read the file. Not text or joblib.")
        # data = Cload(filename, center=False)
        data = [list(map(float, i.split())) for i in data_file.readlines()]

        if len(data) != N:
            raise ValueError("N does not correspond to the number of lines!")
        return np.array(data)

def fetch_block(folder, ind):
    """
    A more generic function to fetch block number "ind" from trajectory 
    
    
    This function is useful if you want to load both "old style" trajectories (block1.dat), 
    and "new style" trajectories ("blocks_1-50.h5")
    
    It will be used in files "show" 
    
    Parameters
    ----------
    
        folder: str, folder with a trajectory

        ind: str or int, number of a block to fetch 
    
    Returns
    -------
        data, Nx3 numpy array     
    """
    blocksh5 = glob.glob(os.path.join(folder,"blocks*.h5"))
    blocksdat = glob.glob(os.path.join(folder, "block*.dat"))
    ind = int(ind)
    if (len(blocksh5) > 0) and (len(blocksdat) > 0):
        raise ValueError("both .h5 and .dat files found in folder - exiting")

    if len(blocksh5) > 0:
        fnames = [os.path.split(i)[-1] for i in blocksh5]
        inds = [i.split("_")[-1].split(".")[0].split("-") for i in fnames]    
        exists = [(int(i[0]) <= ind) and (int(i[1]) >= ind) for i in inds]

        if True not in exists:
            raise ValueError(f"block {ind} not found in files")
        if exists.count(True) > 1:
            raise ValueError("Cannot find the file uniquely: names are wrong")
        pos = exists.index(True)
        block = load_block(blocksh5[pos]+f"::{ind}")["pos"]

    if len(blocksdat) > 0:
        block = load(os.path.join(folder, f"block{ind}.dat"))
    return block

def save(data, filename, mode="txt",  pdbGroups=None):
    data = np.asarray(data, dtype=np.float32)
    
    if mode.lower() == "joblib":                
        joblib.dump({"data":data}, filename=filename, compress=9)
        return

    if mode.lower() == "txt":
        lines = [str(len(data)) + "\n"]

        for particle in data:
            lines.append("{0:.3f} {1:.3f} {2:.3f}\n".format(*particle))
        if filename == None:
            return lines
        
        elif isinstance(filename, six.string_types):
            with open(filename, 'w') as myfile:
                myfile.writelines(lines)
        elif hasattr(filename, "writelines"):
            filename.writelines(lines)
        else:
            raise ValueError("Not sure what to do with filename {0}".format(filename))
                             

    elif mode == 'pdb':
        data = data - np.minimum(np.min(data, axis=0), np.zeros(3, float) - 100)[None, :]
        retret = ""

        def add(st, n):
            if len(st) > n:
                return st[:n]
            else:
                return st + " " * (n - len(st) )

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
            #ret = add(ret + "%i" % (atomNum), 30)
            ret = add(ret + ("%8.3f" % line[0]), 38)
            ret = add(ret + ("%8.3f" % line[1]), 46)
            ret = add(ret + ("%8.3f" % line[2]), 54)
            ret = add(ret + (" 1.00"), 61)
            ret = add(ret + str(float(i % 8 > 4)), 67)
            ret = add(ret, 73)
            ret = add(ret + str(segmentNum), 77)
            retret += (ret + "\n")
        with open(filename, 'w') as f:
            f.write(retret)
            f.flush()
    elif mode == "pyxyz":
        with open(filename, 'w') as f:             
            for i in data: 
                filename.write("C {0} {1} {2}".format(*i))
            

    else:
        raise ValueError("Unknown mode : %s, use h5dict, joblib, txt or pdb" % mode)
                            


def rotation_matrix(rotate):
    """Calculates rotation matrix based on three rotation angles"""
    tx, ty, tz = rotate
    Rx = np.array([[1, 0, 0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, -np.sin(ty)], [0, 1, 0], [np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0, 0, 1]])
    return np.dot(Rx, np.dot(Ry, Rz))
                             

