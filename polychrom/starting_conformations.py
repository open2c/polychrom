from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
from math import sqrt, sin, cos

import numpy as np


def create_spiral(r1, r2, N):
    """
    Creates a "propagating spiral", often used as an easy mitotic-like
    starting conformation.

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
        return r * sin(phi), r * cos(phi)

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
            mid = (phi1 + phi2) / 2.0
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
            mid = (phi1 + phi2) / 2.0
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
            if rad(curphi) > r2:
                forward = False
                z += 1
                add_point(fullcoord(curphi, z))
        else:
            curphi = prevphi(curphi)
            add_point(fullcoord(curphi, z))
            if rad(curphi) < r1:
                forward = True
                z += 1
                add_point(fullcoord(curphi, z))


def create_random_walk(step_size, N):
    """
    Creates a freely joined chain of length N with step step_size 
    """
    theta = np.random.uniform(0.0, 1.0, N)
    theta = 2.0 * np.pi * theta

    u = np.random.uniform(0.0, 1.0, N)

    u = 2.0 * u - 1.0
    x = step_size * np.sqrt(1.0 - u * u) * np.cos(theta)
    y = step_size * np.sqrt(1.0 - u * u) * np.sin(theta)
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
    if N > boxSize ** 3:
        raise ValueError("Steps ahs to be less than size^3")
    if N > 0.9 * boxSize ** 3:
        warnings.warn("N > 0.9 * boxSize**3. It will be slow")
    if (N % 2 != 0) and (method != "linear"):
        raise ValueError("N has to be multiple of 2 for rings")

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

    b = np.zeros((boxSize + 2, boxSize + 2, boxSize + 2), int)
    for i in a:
        b[i] = 1

    for i in range((N - len(a)) // 2):
        while True:
            if method == "linear":
                t = np.random.randint(0, len(a) - 1)
            else:
                t = np.random.randint(0, len(a))

            if t != len(a) - 1:
                c = np.abs(np.array(a[t]) - np.array(a[t + 1]))
                t0 = np.array(a[t])
                t1 = np.array(a[t + 1])
            else:
                c = np.abs(np.array(a[t]) - np.array(a[0]))
                t0 = np.array(a[t])
                t1 = np.array(a[0])
            cur_direction = np.argmax(c)
            while True:
                direction = np.random.randint(0, 3)
                if direction != cur_direction:
                    break
            if np.random.random() > 0.5:
                shift = 1
            else:
                shift = -1
            shiftar = np.array([0, 0, 0])
            shiftar[direction] = shift
            t3 = t0 + shiftar
            t4 = t1 + shiftar
            if (
                (b[tuple(t3)] == 0)
                and (b[tuple(t4)] == 0)
                and (np.min(t3) >= 1)
                and (np.min(t4) >= 1)
                and (np.max(t3) < boxSize + 1)
                and (np.max(t4) < boxSize + 1)
            ):
                a.insert(t + 1, tuple(t3))
                a.insert(t + 2, tuple(t4))
                b[tuple(t3)] = 1
                b[tuple(t4)] = 1
                break
                # print a
    return np.array(a) - 1
