from __future__ import absolute_import, division, print_function, unicode_literals
import six
import warnings
import numpy as np
import joblib
import os
from math import sqrt, sin, cos
import numpy


import scipy, scipy.stats  # @UnusedImport

import numpy as np
import joblib
import gzip

import io



def scanBlocks(folder, assertContinuous=True):
    if not os.path.exists(folder):
        files = []
    else:
        files = [i for i in os.listdir(folder) if i.startswith("block") and i.endswith("dat")]
        files = sorted(files, key=lambda x: int(x[5:-4]))

    keys = np.array([int(i[5:-4]) for i in files])

    if assertContinuous:
        if len(files) > 0:
            assert np.all(np.diff(np.array(keys)) == 1)

    files = [os.path.join(folder, i) for i in files]
    return {"files": files, "keys": keys}



def load(filename, h5dictKey=None):
    """Universal load function for any type of data file"""
   
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



def persistenceLength(data):
    bonds = np.diff(data, axis=0)
    lens = np.sqrt((bonds ** 2).sum(axis=1))
    bondCosines = np.dot(bonds, bonds.T) / lens[:, None] / lens[:, None].T
    avgCosines = np.array([np.diag(bondCosines, i).mean() for i in range(lens.size)])
    truncCosines = avgCosines[:np.where(avgCosines < 1.0 / np.e / np.e)[0][0]]
    slope, intercept, _, _, _ = scipy.stats.linregress(
            list(range(truncCosines.size)), np.log(truncCosines))
    return -1.0 / slope



def create_spiral(r1, r2, N):
    """
    Creates a "propagating spiral", often used as an easy mitotic-like starting conformation.
    Run it with r1=10, r2 = 13, N=5000, and see what it does.
    """
    Pi = 3.141592
    points = []
    finished = [False]

    def rad(phi):
        return phi / (2 * Pi)

    def ang(rad):
        return 2 * Pi * rad

    def coord(phi):
        r = rad(phi)
        return (r * sin(phi), r * cos(phi))

    def fullcoord(phi, z):
        c = coord(phi)
        return [c[0], c[1], z]

    def dist(phi1, phi2):
        c1 = coord(phi1)
        c2 = coord(phi2)
        d = sqrt((c1[1] - c2[1]) ** 2 + (c1[0] - c2[0]) ** 2)
        return d

    def nextphi(phi):
        phi1 = phi
        phi2 = phi + 0.7 * Pi
        mid = phi2
        while abs(dist(phi, mid) - 1) > 0.00001:
            mid = (phi1 + phi2) / 2.
            if dist(phi, mid) > 1:
                phi2 = mid
            else:
                phi1 = mid
        return mid

    def prevphi(phi):

        phi1 = phi
        phi2 = phi - 0.7 * Pi
        mid = phi2

        while abs(dist(phi, mid) - 1) > 0.00001:
            mid = (phi1 + phi2) / 2.
            if dist(phi, mid) > 1:
                phi2 = mid
            else:
                phi1 = mid
        return mid

    def add_point(point, points=points, finished=finished):
        if (len(points) == N) or (finished[0] == True):
            points = np.array(points)
            finished[0] = True
            print("finished!!!")
        else:
            points.append(point)

    z = 0
    forward = True
    curphi = ang(r1)
    add_point(fullcoord(curphi, z))
    while True:
        if finished[0] == True:
            return np.array(points)
        if forward == True:
            curphi = nextphi(curphi)
            add_point(fullcoord(curphi, z))
            if (rad(curphi) > r2):
                forward = False
                z += 1
                add_point(fullcoord(curphi, z))
        else:
            curphi = prevphi(curphi)
            add_point(fullcoord(curphi, z))
            if (rad(curphi) < r1):
                forward = True
                z += 1
                add_point(fullcoord(curphi, z))


def create_random_walk(step_size, N):
    """
    Creates a freely joined chain of length N with step step_size 
    """
    theta = np.random.uniform(0., 1., N)
    theta = 2.0 * np.pi * theta
    
    u = np.random.uniform(0., 1., N)
    
    u = 2.0 * u - 1.0
    x = step_size * np.sqrt(1. - u * u) * numpy.cos(theta)
    y = step_size * np.sqrt(1. - u * u) * numpy.sin(theta)
    z = step_size * u
    x, y, z = np.cumsum(x), np.cumsum(y), np.cumsum(z)
    return np.vstack([x, y, z]).T


                             
def grow_cubic(N, boxSize, method="standard"):
    """
    This function grows a ring or linear polymer on a cubic lattice 
    in the cubic box of size boxSize. 
    
    If method=="standard, grows a ring starting with a 4-monomer ring in the middle 
    
    if method =="extended", it grows a ring starting with a long ring 
    going from z=0, center of XY face, to z=boxSize center of XY face, and back. 
    
    If method="linear", then it grows a linearly organized chain from 0 to size.
    The chain may stick out of the box by one, (N%2 != boxSize%2), or be flush with the box otherwise

    Parameters
    ----------
    N: chain length. Must be even for rings. 
    boxSize: size of a box where polymer is generated.
    method: "standard", "linear" or "extended"


    """
    if N > boxSize**3:
        raise ValueError("Steps ahs to be less than size^3")
    if N > 0.9 * boxSize**3:
        warnings.warn("N > 0.9 * boxSize**3. It will be slow")                      
    if (N % 2 != 0) and (method != "linear"):
        raise ValueError("N has to be multiple of 2 for rings")    
                                                          
    numpy = np
    t = boxSize // 2
    if method == "standard":
        a = [(t, t, t), (t, t, t + 1), (t, t + 1, t + 1), (t, t + 1, t)]
        
    elif method == "extended":
        a = []
        for i in range(1, boxSize):
            a.append((t, t, i))

        for i in range(boxSize - 1, 0, -1):
            a.append((t, t - 1, i))
        if len(a) > N:
            raise ValueError("polymer too short for the box size")

    elif method == "linear":
        a = []
        for i in range(0, boxSize + 1):
            a.append((t, t, i))
        if (len(a) % 2) != (N % 2):
            a = a[1:]        
        if len(a) > N:
            raise ValueError("polymer too short for the box size")

    else:
        raise ValueError("select methon from standard, extended, or linear")

    b = numpy.zeros((boxSize + 2, boxSize + 2, boxSize + 2), int)
    for i in a:
        b[i] = 1
        
    for i in range((N - len(a)) // 2):
        while True:
            if method == "linear":
                t = numpy.random.randint(0, len(a)-1)
            else: 
                t = numpy.random.randint(0, len(a))
            
            if t != len(a) - 1:
                c = numpy.abs(numpy.array(a[t]) - numpy.array(a[t + 1]))
                t0 = numpy.array(a[t])
                t1 = numpy.array(a[t + 1])
            else:
                c = numpy.abs(numpy.array(a[t]) - numpy.array(a[0]))
                t0 = numpy.array(a[t])
                t1 = numpy.array(a[0])
            cur_direction = numpy.argmax(c)
            while True:
                direction = numpy.random.randint(0, 3)
                if direction != cur_direction:
                    break
            if numpy.random.random() > 0.5:
                shift = 1
            else:
                shift = -1
            shiftar = numpy.array([0, 0, 0])
            shiftar[direction] = shift
            t3 = t0 + shiftar
            t4 = t1 + shiftar
            if (b[tuple(t3)] == 0) and (b[tuple(t4)] == 0) and (numpy.min(t3) >= 1) and (numpy.min(t4) >= 1) and (
                numpy.max(t3) < boxSize+1) and (numpy.max(t4) < boxSize+1):
                a.insert(t + 1, tuple(t3))
                a.insert(t + 2, tuple(t4))
                b[tuple(t3)] = 1
                b[tuple(t4)] = 1
                break
                # print a
    return numpy.array(a) - 1

def grow_rw(step, size, method="line"):
    raise DeprecationWarning("grow_rw is being renamed to grow_cubic")
    return grow_cubic(N, boxSize, method="line")
                         
                         
                             

                             
def kabsch(P, Q):
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
    
    dist = np.sqrt(np.mean((np.dot(P, U) - Q)**2)  * 3)

    return dist
