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
        files = os.listdir(folder)
        files = [i for i in files if i.startswith("block") and i.endswith("dat")]
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

    try:
        "loading from a joblib file here"
        mydict = dict(joblib.load(filename))
        data = mydict.pop("data")
        return data

    except:
        pass
    
    "checking for a text file"
    data_file = open(filename)
    line0 = data_file.readline()
    try:
        N = int(line0)
    except (ValueError, UnicodeDecodeError):
        raise TypeError("Cannot read text file... reading pickle file")
    # data = Cload(filename, center=False)
    data = [list(map(float, i.split())) for i in data_file.readlines()]

    if len(data) != N:
        raise ValueError("N does not correspond to the number of lines!")
    return np.array(data)

    

    #try:
    #    data = loadJson(filename)
    #    return data["data"]
    #except:
    #    print("Could not load json")
    #    pass

    #h5dict loading deleted



def save(data, filename, mode="txt", h5dictKey="1", pdbGroups=None):
    data = np.asarray(data, dtype=np.float32)

    h5dictKey = str(h5dictKey)
    mode = mode.lower()

    if mode == "h5dict":
        from mirnylib.h5dict import h5dict
        mydict = h5dict(filename, mode="w")
        mydict[h5dictKey] = data
        del mydict
        return

    elif mode in ["joblib", "json"]:
        metadata = {}
        metadata["data"] = data
        if mode == "joblib":
            joblib.dump(metadata, filename=filename, compress=9)
        else:

            with gzip.open(filename, 'wb') as myfile:
                mystr = json.dumps(metadata,  cls=NumpyEncoder)
                mybytes = mystr.encode("ascii")
                myfile.write(mybytes)

        return

    elif mode == "txt":
        lines = []
        lines.append(str(len(data)) + "\n")

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
            return lines

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


def bondLengths(data):
    bonds = np.diff(data, axis=0)
    return np.sqrt((bonds ** 2).sum(axis=1))


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
    Creates a "propagating spiral", often used as a starting conformation.
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
            return np.transpose(points)
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


def create_random_walk(step_size, N, segment_length=1):
    theta = np.repeat(np.random.uniform(0., 1., N // segment_length + 1),
                      segment_length)
    theta = 2.0 * np.pi * theta[:N]
    u = np.repeat(np.random.uniform(0., 1., N // segment_length + 1),
                  segment_length)
    u = 2.0 * u[:N] - 1.0
    x = step_size * np.sqrt(1. - u * u) * numpy.cos(theta)
    y = step_size * np.sqrt(1. - u * u) * numpy.sin(theta)
    z = step_size * u
    x, y, z = np.cumsum(x), np.cumsum(y), np.cumsum(z)
    return np.vstack([x, y, z]).T


matlabImported = False



def grow_rw(step, size, method="line"):
    """This does not grow a random walk, but the name stuck.

    What it does - it grows a polymer in the middle of the sizeXsizeXsize box.
    It can start with a small ring in the middle (method="standart"),
    or it can start with a line ("method=line").
    If method="linear", then it grows a linearly organized chain from 0 to size.

    step has to be less than size^3

    """
    numpy = np
    t = size // 2
    if method == "standard":
        a = [(t, t, t), (t, t, t + 1), (t, t + 1, t + 1), (t, t + 1, t)]
    elif method == "line":
        a = []
        for i in range(1, size):
            a.append((t, t, i))

        for i in range(size - 1, 0, -1):
            a.append((t, t - 1, i))

    elif method == "linear":
        a = []
        for i in range(0, size + 1):
            a.append((t, t, i))
        if (len(a) % 2) != (step % 2):
            a = a[:-1]

    else:
        raise ValueError("select methon from line, standard, linear")

    b = numpy.zeros((size + 1, size + 1, size + 1), int)
    for i in a:
        b[i] = 1
    for i in range((step - len(a)) // 2):
        # print len(a)
        while True:
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
                numpy.max(t3) < size) and (numpy.max(t4) < size):
                a.insert(t + 1, tuple(t3))
                a.insert(t + 2, tuple(t4))
                b[tuple(t3)] = 1
                b[tuple(t4)] = 1
                break
                # print a
    return numpy.array(a)


def _test():
    print("testing save/load")
    a = np.random.random((20000, 3))
    save(a, "bla", mode="txt")
    b = load("bla")
    print(a)
    print(b)
    assert abs(b.mean() - a.mean()) < 0.00001

    save(a, "bla", mode="joblib")
    b = load("bla")
    assert abs(b.mean() - a.mean()) < 0.00001

    #save(a, "bla.json", mode="json")
    #b = loadJson("bla.json")["data"]
    #assert abs(b.mean() - a.mean()) < 0.00001


    #save(a, "bla.json.gz", mode="json")
    #b = load("bla.json.gz")
    #assert abs(b.mean() - a.mean()) < 0.00001

    #save(a, "bla.json", mode="json")
    #b = load("bla.json")
    #assert abs(b.mean() - a.mean()) < 0.00001

    #save(a, "bla", mode="h5dict")
    #b = load("bla")
    #assert abs(b.mean() - a.mean()) < 0.00001

    os.remove("bla")
    os.remove("bla.json.gz")

    print("Finished testing save/load, successful")


def createSpiralRing(N, twist, r=0, offsetPerParticle=np.pi, offset=0):
    """
    Creates a ring of length N. Then creates a spiral
    """
    from mirnylib import numutils
    if not numutils.isInteger(N * offsetPerParticle / (2 * np.pi)):
        print(N * offsetPerParticle / (2 * np.pi))
        raise ValueError("offsetPerParticle*N should be multitudes of 2*Pi")
    totalTwist = twist * N
    totalTwist = np.floor(totalTwist / (2 * np.pi)) * 2 * np.pi
    alpha = np.linspace(0, 2 * np.pi, N + 1)[:-1]
    # print alpha
    twistPerParticle = totalTwist / float(N) + offsetPerParticle
    R = float(N) / (2 * np.pi)
    twist = np.cumsum(np.ones(N, dtype=float) * twistPerParticle) + offset
    # print twist
    x0 = R + r * np.cos(twist)
    z = 0 + r * np.sin(twist)
    x = x0 * np.cos(alpha)
    y = x0 * np.sin(alpha)
    return np.array(np.array([x, y, z]).T, order="C")


def smooth_conformation(conformation, n_avg):
    """Smooth a conformation using moving average.
    """
    if conformation.shape[0] == 3:
        conformation = conformation.T
    new_conformation = np.zeros(shape=conformation.shape)
    N = conformation.shape[0]

    for i in range(N):
        if i < n_avg:
            new_conformation[i] = conformation[:i + n_avg].mean(axis=0)
        elif i >= N - n_avg:
            new_conformation[i] = conformation[-(N - i + n_avg):].mean(axis=0)
        else:
            new_conformation[i] = conformation[i - n_avg:i + n_avg].mean(axis=0)
    return new_conformation




def getCloudGeometry(d, frac=0.05, numSegments=1, widthPercentile=50, delta=0):
    """Trace the centerline of an extended cloud of points and determine
    its length and width.

    The function switches to the principal axes of the cloud (e1,e2,e3)
    and applies LOWESS to define the centerline as (x2,x3)=f(x1).
    The length is then determined as the total length of the centerline.
    The width is determined as the median shortest distance from the points of
    clouds to the centerline.
    On top of that, the cloud can be chopped into `numSegments` in the order
    of data entries in `d`. The centerline is then determined independently for
    each segment.

    Parameters
    ----------

    d : np.array, 3xN
        an array of coordinates

    frac : float
        The fraction of all points used to determine the local position and
        slope of the centerline in LOWESS.

    numSegments : int
        The number of segments to split `d` into. The centerline in fit
        independently for each data segment.

    widthPercentile : float
        The width is determined at `widthPercentile` of shortest distances
        from the points to the centerline. The default value is 50, i.e. the
        width is the median distance to the centerline.

    delta : float
        The parameter of LOWESS. According to the documentation:
        "delta can be used to save computations. For each x_i, regressions are
        skipped for points closer than delta. The next regression is fit for the
        farthest point within delta of x_i and all points in between are
        estimated by linearly interpolating between the two regression fits."

    Return
    ------

        (length, width) : (float, float)

    """

    import statsmodels
    import statsmodels.nonparametric
    import statsmodels.nonparametric.smoothers_lowess
    from mirnylib import numutils

    dists = []
    length = 0.0
    for segm in range(numSegments):
        segmd = d[segm * (d.shape[0] // numSegments): (segm + 1) * (d.shape[0] // numSegments)]
        (e1, e2), _ = numutils.PCA(segmd, 2)
        e3 = np.cross(e1, e2)
        xs = np.dot(segmd, e1)
        ys = np.vstack([np.dot(segmd, e2), np.dot(segmd, e3)])
        ys_pred = np.vstack([
            statsmodels.nonparametric.smoothers_lowess.lowess(
                    ys[0], xs, frac=frac, return_sorted=False, delta=10),
            statsmodels.nonparametric.smoothers_lowess.lowess(
                    ys[1], xs, frac=frac, return_sorted=False,
                    delta=10)])
        order = np.argsort(xs)
        fit_d = np.vstack([xs[order],
                           ys_pred[0][order],
                           ys_pred[1][order]]).T

        for i in range(len(xs)):
            dists.append(
                    (((fit_d - np.array([xs[i], ys[0][i], ys[1][i]])) ** 2).sum(axis=1) ** 0.5).min())

        length += (((fit_d[1:] - fit_d[:-1]) ** 2).sum(axis=1) ** 0.5).sum()
    width = np.percentile(dists, widthPercentile)

    return length, width


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


