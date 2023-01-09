# Code written by: Maksim Imakaev (imakaev@mit.edu)
#                  Anton Goloborodko (golobor@mit.edu)

"""This class is a collection of functions for showing data with pymol. Note that the limit of pymol is 100k
monomers, therefore interpolateData is useful to collapse the 200k-long simulation into a 100k-long conformation. """
import os
import shutil
import subprocess
import tempfile
import textwrap

import numpy as np
from scipy.interpolate.fitpack2 import InterpolatedUnivariateSpline

from . import polymerutils


def interpolateData(data, targetN=90000, colorArrays=[]):
    """
    Converts a polymer of any length to a smoothed chain with (hopefully) fixed distance between neighboring
    monomers. Does it by cubic spline interpolation as following.

    1. Interpolate the data using cubic spline \n
    2. Evaluate cubic spline at targetN*10 values \n
    3. Rescale the evaluated spline such that total distance is targetN \n
    4. Select targetN points along the path with distance between
    neighboring points _along the chain_ equal to 1.

    Parameters
    ----------
    data : Nx3 array
        Input xyz coordinates
    targetN : int
        Length of output polymer.
        It is not adviced to make it many times less than N

    Returns
    -------
    (about targetN) x 3 array
    """

    fineGrain = 10

    N = len(data)
    numDim = len(data[0])
    targetDataSize = targetN * fineGrain

    evaluateRange = np.arange(N)
    targetRange = np.arange(0, N - 1, N / float(targetDataSize))

    splined = np.zeros((len(targetRange), numDim), float)
    colorsSplined = []
    for coor in range(numDim):
        spline = InterpolatedUnivariateSpline(evaluateRange, data[:, coor], k=3)
        evaled = spline(targetRange)
        splined[:, coor] = evaled

    for color in colorArrays:
        spline = InterpolatedUnivariateSpline(evaluateRange, color, k=2)
        evaled = spline(targetRange)
        colorsSplined.append(evaled)

    dists = np.sqrt(np.sum(np.diff(splined, 1, axis=0) ** 2, axis=1))
    totalDist = np.sum(dists)
    mult = totalDist / targetN
    splined /= mult
    dists /= mult
    cumDists = np.cumsum(dists)
    searched = np.searchsorted(cumDists, np.arange(1, targetN))
    v1 = cumDists[searched]
    v2 = cumDists[searched - 1]
    vals = np.floor(v1)
    p1 = (v1 - vals) / (v1 - v2)
    p2 = 1 - p1

    colorReturn = [i[searched] for i in colorsSplined]

    evaled = p2[:, None] * splined[searched] + p1[:, None] * splined[searched - 1]
    return evaled, colorReturn


def createRegions(a):
    """
    Creates array of non-zero regions of a.
    if a is 0 1 1 0 1 0
    result will be (1,3), (4,5), because elements 1,2 and 4 are non-zero.

    """

    a = a > 0
    a = np.r_[False, a, False]
    a1 = np.nonzero(a[1:] * (1 - a[:-1]))[0]
    a2 = np.nonzero(a[:-1] * (1 - a[1:]))[0]

    return np.transpose(np.array([a1, a2]))


def do_coloring(
    data,
    regions,
    colors,
    transparencies,
    showGui=True,
    saveTo=None,
    showChain="worm",
    returnScriptName=None,
    showMainChain=True,
    chainRadius=0.02,
    subchainRadius=0.04,
    chainTransparency=0.5,
    support="",
    transparentBackground=True,
    multiplier=0.4,
    spherePositions=[],
    pdbGroups=None,
    sphereRadius=0.3,
    sphereColor="grey60",
    force=False,
    miscArguments="",
):

    """
    !!! Please read this completely. Otherwise you'll suck :( !!!

    Creates a PDB file and a rasmol script that shows an XYZ polymer
    using pymol. Polymer consists of two parts: chain and subchain.
    A chain is a grey polymer, that is meant to resemble the main chain.
    It is meant to be thin, gray and transparent (overall, here transparency 1
    means transparent, transparency 0 means fully visible). A subchain
    consists of a certain number of regions, each has it's own color,
    transparency, etc.

    Parameters
    ----------

    data : an Nx3 array of XYZ coordinates

    regions : a list of tuples (start, end)
        Note that rasmol acceps subchains in a format
        (first monomer, last monomer), not the usual python
        convention (first, last+1)!!! An overlap check will watch this.
        If you want two colorings to gradually transition into each other,
        then you should use ((0,10),(10,25),(25,...)).


    colors : a list of colors ("red", "green", "blue", etc.)  for each region

    transparencies : a list of floats between 0 and 1. 0 is fully visible

    chain_radius : radius of a main chain in arbitraty units

    subchain_radius : radius of a subchain in arbitrary units

    chain_transparency : transparency of a main chain

    support : code to put at the end of the script
        put all the "save" or "ray" commands here if you want automation

    multiplier : a number, probably between .1 and 3
        Increasing it makes chain more smooth
        Decreasing it makes it more kinky, but may cause bugs
        or even missing chain regions

    misc_arguments : str
        Misc arguments to pymol command at the very end (mainly >/dev/null)

    .. warning :: Do not call this scripy "pymol.py!"

    .. warning ::
        Please resize the window to the needed size and run
        "ray" command (press "ray" button) to get a nice image.
        Then DO NOT MOVE the image and find "export" in the menu.
        Otherwise your image will be not that high quality

    .. note ::
        performance of "ray" command depends on two things.
        First is resolution : it is more than quadratic in that
        Second is chain complexity. Tens thousand of monomers
        at high resolution may take up to an hour to ray.
        Though it actually looks awesome then!

    Run an example method below to see how the code works.
    See full automation examples below.
    """

    def getSelectionString(start, end):
        if start > end:
            raise ValueError("start should be less than end")
        maxNum = 9000
        atom1 = (start + 1) % maxNum
        seg1 = (start + 1) // maxNum + 1
        atom2 = (end + 1) % maxNum
        seg2 = (end + 1) // maxNum + 1

        if seg1 == seg2:
            return "resi {atom1}-{atom2} and segi {seg1}".format(**locals())
        elif np.abs(seg1 - seg2) == 1:
            return "(resi {atom1}-{maxNum} and segi {seg1}) or (resi 0-{atom2} and segi {seg2})".format(**locals())

        elif np.abs(seg1 - seg2) >= 2:

            line = "(resi {atom1}-{maxNum} and segi {seg1}) or (resi 0-{atom2} and segi {seg2})".format(**locals())
            for i in range(seg1 + 1, seg2):
                line = line + " or (segi {0})".format(i)
            return line
        else:
            raise ValueError("Atoms are too far")

    data = np.array(data)
    data *= multiplier
    chainRadius *= multiplier
    sphereRadius *= multiplier

    if not hasattr(subchainRadius, "__iter__"):
        subchainRadius = [subchainRadius for _ in regions]
    subchainRadius = [i * multiplier for i in subchainRadius]

    tmpPdbFile = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb")
    tmpPdbFilename = tmpPdbFile.name
    pdbname = os.path.split(tmpPdbFilename)[-1].replace(".pdb", "")
    tmpPdbFile.close()
    polymerutils.save(data, tmpPdbFilename, mode="pdb", pdbGroups=pdbGroups)

    # starting background check
    N = len(data)
    nregions = np.array(regions)
    if len(nregions) > 0:
        if nregions.min() < 0 or nregions.max() > N:
            raise ValueError("region boundaries should be between 0 and N-1")
        regions = np.array(regions)
        regions = np.sort(regions, axis=1)

        args = np.argsort(regions[:, 0])[::-1]
        regions = regions[args]
        colors = [colors[i] for i in args]
        subchainRadius = [subchainRadius[i] for i in args]
        transparencies = [transparencies[i] for i in args]

        ends = regions[1:, 1]
        starts = regions[:-1, 0]

        if (force is False) and (starts < ends).any():
            raise ValueError(
                "Overlapped regions detected! Rasmol will not work. "
                "E.g. valid regions are ((0,10),(10,20)), but not ((0,10),(9,20))"
            )

    bgcolor = "grey"
    letters = [i for i in "1234567890abcdefghijklmnopqrstuvwxyz"]
    names = [i + j + k for i in letters for j in letters for k in letters]

    out = tempfile.NamedTemporaryFile(mode="w")

    if returnScriptName is not None:
        pdbname = returnScriptName
    out.write("hide all\n")
    out.write("bg white\n")
    if transparentBackground:
        out.write("set ray_opaque_background, off\n")

    for i in range(len(regions)):
        out.write("select %s\n" % (getSelectionString(*regions[i])))
        out.write("create subchain%s,sele\n" % (names[i]))
        # out.write("remove subchain%s in %s\n"%(names[i],pdbname))

    if showChain == "worm":
        out.write("set cartoon_trace_atoms,1,%s\n" % pdbname)
        out.write("cartoon tube,%s\n" % pdbname)
        out.write("set cartoon_tube_radius,%f,%s\n" % (chainRadius, pdbname))
        out.write("set cartoon_transparency,%f,%s\n" % (chainTransparency, pdbname))
        out.write("color %s,%s\n" % (bgcolor, pdbname))

    elif showChain == "spheres":
        out.write("alter {0}, vdw={1}\n".format(pdbname, 1.5 * chainRadius))
        out.write("show spheres\n")
        out.write("as spheres\n")
        out.write("set sphere_transparency,%f,%s\n" % (chainTransparency, pdbname))
        out.write("color %s,%s\n" % (bgcolor, pdbname))
    elif (showChain == "none") or (not showChain):
        pass
    else:
        raise ValueError("please select showChain to be 'worm' or 'spheres' or 'none'")
    for i in range(len(regions)):

        name = "subchain%s" % names[i]
        if showChain == "worm":
            out.write("set cartoon_trace_atoms,1,%s\n" % name)
            out.write("set cartoon_tube_radius,%f,%s\n" % (subchainRadius[i], name))
            out.write("cartoon tube,%s\n" % name)
            out.write("color %s,subchain%s\n" % (colors[i], names[i]))
            out.write("set cartoon_transparency,%f,%s\n" % (transparencies[i], name))

        elif showChain == "spheres":
            out.write("alter {0}, vdw={1}\n".format(name, 1.5 * subchainRadius[i]))
            out.write("show spheres, %s\n" % name)
            out.write("as spheres\n")
            out.write("color %s,subchain%s\n" % (colors[i], names[i]))
            out.write("set sphere_transparency,%f,%s\n" % (transparencies[i], name))
        elif (showChain == "none") or (not showChain):
            pass
        else:
            raise ValueError("please select showChain to be 'worm' or 'spheres' or 'none'")

    for i in spherePositions:
        out.write("select {0} and  {1}\n".format(name, getSelectionString(i, i)))
        out.write("show spheres, sele\n")
        out.write("alter sele, vdw={0}\n".format(1.5 * sphereRadius))
        out.write("set sphere_color, {0}, sele \n".format(sphereColor))

    if showChain == "worm":
        out.write("show cartoon,name ca\n")
    out.write("zoom %s\n" % pdbname)

    out.write(support)
    out.write("\n")
    out.flush()

    if returnScriptName is not None:
        out.flush()
        return "".join(open(out.name).readlines())

    # out.write("alter all, vdw={0} \n".format(sphereRadius))

    script = "".join(open(out.name).readlines())
    if not (saveTo is None):
        # out.write("viewport 1200,1200\n")
        out.write("ray 800,800\n")
        out.write("png {}\n".format(saveTo))
        print("saved to: ", saveTo)
    if not showGui:
        out.write("quit\n")

    out.flush()

    # saving data

    from time import sleep

    sleep(0.5)

    print(os.system("pymol {1} -u {0} {2}".format(out.name, tmpPdbFilename, miscArguments)))
    return script


def new_coloring(
    data,
    regions,
    colors,
    transparencies,
    showGui=True,
    saveTo=None,
    showChain="worm",
    returnScriptName=None,
    chainRadius=0.02,
    subchainRadius=0.04,
    chainTransparency=0.5,
    support="",
    presupport="",
    transparentBackground=True,
    multiplier=0.4,
    spherePositions=[],
    pdbGroups=None,
    sphereRadius=0.3,
    force=False,
    miscArguments="",
):

    """
    !!! Please read this completely. Otherwise you'll suck :( !!!

    Creates a PDB file and a rasmol script that shows an XYZ polymer
    using pymol. Polymer consists of two parts: chain and subchain.
    A chain is a grey polymer, that is meant to resemble the main chain.
    It is meant to be thin, gray and transparent (overall, here transparency 1
    means transparent, transparency 0 means fully visible). A subchain
    consists of a certain number of regions, each has it's own color,
    transparency, etc.

    Parameters
    ----------

    data : an Nx3 array of XYZ coordinates

    regions : a list of tuples (start, end)
        Note that rasmol acceps subchains in a format
        (first monomer, last monomer), not the usual python
        convention (first, last+1)!!! An overlap check will watch this.
        If you want two colorings to gradually transition into each other,
        then you should use ((0,10),(10,25),(25,...)).


    colors : a list of colors ("red", "green", "blue", etc.)  for each region

    transparencies : a list of floats between 0 and 1. 0 is fully visible

    chain_radius : radius of a main chain in arbitraty units

    subchain_radius : radius of a subchain in arbitrary units

    chain_transparency : transparency of a main chain

    support : code to put at the end of the script
        put all the "save" or "ray" commands here if you want automation

    multiplier : a number, probably between .1 and 3
        Increasing it makes chain more smooth
        Decreasing it makes it more kinky, but may cause bugs
        or even missing chain regions

    misc_arguments : str
        Misc arguments to pymol command at the very end (mainly >/dev/null)

    .. warning :: Do not call this scripy "pymol.py!"

    .. warning ::
        Please resize the window to the needed size and run
        "ray" command (press "ray" button) to get a nice image.
        Then DO NOT MOVE the image and find "export" in the menu.
        Otherwise your image will be not that high quality

    .. note ::
        performance of "ray" command depends on two things.
        First is resolution : it is more than quadratic in that
        Second is chain complexity. Tens thousand of monomers
        at high resolution may take up to an hour to ray.
        Though it actually looks awesome then!

    Run an example method below to see how the code works.
    See full automation examples below.
    """
    data = np.array(data)
    data *= multiplier
    chainRadius *= multiplier
    sphereRadius *= multiplier

    if not hasattr(subchainRadius, "__iter__"):
        subchainRadius = [subchainRadius for _ in regions]
    subchainRadius = [i * multiplier for i in subchainRadius]

    tmpPdbFile = tempfile.NamedTemporaryFile(mode="w", suffix=".pdb")
    tmpPdbFilename = tmpPdbFile.name
    tmpPdbFile.close()
    polymerutils.save(data, tmpPdbFilename, mode="pdb", pdbGroups=pdbGroups)

    letters = [i for i in "1234567890abcdefghijklmnopqrstuvwxyz"]
    names = [i + j + k for i in letters for j in letters for k in letters]

    out = tempfile.NamedTemporaryFile(mode="w")

    out.write(presupport)
    out.write("hide all\n")
    out.write("bg white\n")
    if transparentBackground:
        out.write("set ray_opaque_background, off\n")

    for i in range(len(regions)):
        out.write("select %s, resi %d-%d\n" % (names[i], regions[i][0], regions[i][1]))
        out.write("create subchain%s,%s\n" % (names[i], names[i]))
        # out.write("remove subchain%s in %s\n"%(names[i],pdbname))

    for i in range(len(regions)):

        name = "subchain%s" % names[i]
        if showChain == "worm":
            out.write("set cartoon_trace,1,%s\n" % name)
            out.write("set cartoon_tube_radius,%f,%s\n" % (subchainRadius[i], name))
            out.write("cartoon tube,%s\n" % name)
            out.write("color %s,subchain%s\n" % (colors[i], names[i]))
            out.write("set cartoon_transparency,%f,%s\n" % (transparencies[i], name))
            # out.write("show cartoon,%s\n" % (name))

        elif showChain == "spheres":
            out.write("alter {0}, vdw={1}\n".format(name, 1.5 * subchainRadius[i]))
            out.write("show spheres, %s\n" % name)
            out.write("as spheres\n")
            out.write("color %s,subchain%s\n" % (colors[i], names[i]))
            out.write("set sphere_transparency,%f,%s\n" % (transparencies[i], name))

    # if showChain == "worm":
    #    out.write("show cartoon,name ca\n")
    # out.write("zoom %s\n" % pdbname)

    out.write(support)
    out.write("\n")
    out.flush()

    if returnScriptName is not None:
        out.flush()
        return "".join(open(out.name).readlines())

    # out.write("alter all, vdw={0} \n".format(sphereRadius))

    script = "".join(open(out.name).readlines())
    if not (saveTo is None):
        # out.write("viewport 1200,1200\n")
        out.write("ray 800,800\n")
        out.write("png {}\n".format(saveTo))
        print("saved to: ", saveTo)
    if not showGui:
        out.write("quit\n")

    out.flush()

    # saving data

    from time import sleep

    sleep(0.5)

    print(os.system("pymol {1} -u {0} {2}".format(out.name, tmpPdbFilename, miscArguments)))
    return script


def getTmpPath(folder=None, **kwargs):
    tmpFile = tempfile.NamedTemporaryFile(dir=folder, mode="w", **kwargs)
    tmpPath = tmpFile.name
    tmpFilename = os.path.split(tmpPath)[-1]
    tmpFile.close()
    return tmpPath, tmpFilename


def show_chain(data, showGui=True, saveTo=None, showChain="worm", chains=None, support="", **kwargs):
    """Shows a single rainbow-colored chain using PyMOL.

    Arguments:
    gui - if True then show the GUI.
    save_to - a path to a saved .png figure
    showChain - "worm" or "spheres"

    Keywords arguments:
    chain_radius : the radius of the displayed chain. Default: 0.1
    """
    if isinstance(data, str):
        data = polymerutils.load(data)
    chain_radius = kwargs.get("chain_radius", 0.1)
    data -= np.min(data, axis=0)[None, :]
    print(data.min())

    tmpPdbPath, pdbname = getTmpPath(suffix=".pdb")
    if chains is None:
        polymerutils.save(data, tmpPdbPath, mode="pdb")
    else:
        pdbArray = np.zeros(len(data))
        for j, i in enumerate(chains):
            pdbArray[i[0] : i[1]] = j
        polymerutils.save(data, tmpPdbPath, mode="pdb", pdbGroups=pdbArray)
    pdbname = pdbname.replace(".pdb", "")

    tmpScript = tempfile.NamedTemporaryFile(mode="w")
    tmpScript.write("hide all\n")
    tmpScript.write("bg white\n")
    tmpScript.write("set ray_opaque_background, off\n")
    tmpScript.write("enable {0}\n".format(pdbname))

    # Spectrum coloring.
    tmpScript.write("set cartoon_trace_atoms,1,%s\n" % pdbname)
    tmpScript.write("spectrum\n")

    # Change the size of the spheres.

    if showChain == "worm":
        tmpScript.write("cartoon tube,%s\n" % pdbname)
        tmpScript.write("set cartoon_tube_radius,%f,%s\n" % (chain_radius, pdbname))
        tmpScript.write("show cartoon,name ca\n")

    elif showChain == "spheres":
        tmpScript.write("alter {0}, vdw=1.0\n".format(pdbname))
        tmpScript.write("show spheres\n")
    else:
        raise ValueError("please select showChain to be 'worm' or 'spheres'")
    tmpScript.write("zoom {0}\n".format(pdbname))
    tmpScript.write(kwargs.get("support", ""))
    if not (saveTo is None):
        tmpScript.write("viewport 1200,1200\n")
        tmpScript.write("png {}\n".format(saveTo))
    if not showGui:
        tmpScript.write("quit\n")
    tmpScript.write(support)

    tmpScript.flush()

    os.system("pymol {0} {1} -u {2}".format(tmpPdbPath, "" if showGui else "-c", tmpScript.name))
    tmpScript.close()


def makeMoviePymol(
    fileList,
    destFolder,
    fps=10,
    aviFilename="output.avi",
    rotationPeriod=0.0,
    resolution=(600, 600),
    fiberWidth=1.0,
    rescalingFactor=1.0,
    pymolScript=None,
):
    """
    experimental example script for making a movie...
    """

    numFrames = len(fileList)
    numDigits = int(np.ceil(np.log10(numFrames)))
    pdbPaths = []

    destFolder = os.path.abspath(destFolder)
    pdbFolder = destFolder + "/pdb"
    imgFolder = destFolder + "/img"
    for folder in [destFolder, pdbFolder, imgFolder]:
        if not os.path.isdir(folder):
            os.mkdir(folder)

    for i, dataPath in enumerate(fileList):
        d = polymerutils.load(dataPath)
        d[:, 0], d[:, 2] = d[:, 2].copy(), d[:, 0].copy()
        d *= rescalingFactor
        d -= np.mean(d, axis=0)[None, :]
        pdbFilename = "{0:0{width}}.pdb".format(i, width=numDigits)
        savePath = pdbFolder + "/" + pdbFilename
        polymerutils.save(d, savePath, mode="pdb")
        pdbPaths.append(os.path.abspath(savePath))

    script = "hide all\n"
    for i in pdbPaths:
        script += "load {0}, mov\n".format(i)

    script += "smooth mov\n"

    rotationCode = ""
    if rotationPeriod > 0:
        for i in range(numFrames // rotationPeriod + 1):
            rotationCode += "util.mroll {0},{1},0\n".format(i * rotationPeriod + 1, (i + 1) * rotationPeriod)

    if pymolScript is None:
        script += textwrap.dedent(
            """

        rotate [1,1,1], 90, mov
        turn [1,1,1], 45
        smooth mov
        bg white
        set ray_opaque_background, off
        spectrum count, rainbow, mov
        alter mov, vdw={0}
        show spheres
        as spheres
        zoom mov
        viewport {1}, {2}
        set ray_trace_frames=1


        mview store
        mset -{3}
        {4}
        mview reinterpolate, power=1
        mpng mov
        clip slab, 20000
        """.format(
                fiberWidth,
                resolution[0],
                resolution[1],
                numFrames,
                rotationCode,
                # max(0, len(fileList) - fps * 2),  # unused - why it was here
            )
        )
    else:
        script += pymolScript

    tmpScriptPath = os.path.abspath(destFolder + "/movie.pymol")
    tmpScript = open(tmpScriptPath, "w")
    tmpScript.write(script)
    tmpScript.flush()

    os.system("cd {0}; pymol -c -u {1}; cd -".format(imgFolder, tmpScriptPath))

    _mencoder(imgFolder, fps, aviFilename)
    shutil.move(os.path.join(imgFolder, aviFilename), os.path.join(destFolder, aviFilename))


def _mencoder(imgFolder, fps, aviFilename):
    subprocess.call(
        (
            "cd {0}; ".format(imgFolder)
            + 'mencoder "mf://*.png" -mf fps={0} -o {1} '.format(fps, aviFilename)
            + "-ovc lavc -lavcopts vcodec=mpeg4"
        ),
        shell=True,
    )


def makeMovie(fileList, imgFolder, fps=15, aviFilename="output.avi"):
    if not fileList:
        return
    numFrames = len(fileList)
    numDigits = int(np.ceil(np.log10(numFrames)))
    for i, dataPath in enumerate(fileList):
        d = polymerutils.load(dataPath)
        savePath = imgFolder + "/{0:0{width}}.png".format(i, width=numDigits)
        show_chain(d, showGui=False, saveTo=savePath, showChain="spheres")

    _mencoder(imgFolder, fps, aviFilename)
