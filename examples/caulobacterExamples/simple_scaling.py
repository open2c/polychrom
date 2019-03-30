from polymerScalings import give_slices
import numpy as np
import cPickle
import matplotlib.pyplot as plt
from mirnylib.systemutils import setExceptionHook
import polymerutils
from mirnylib.plotting import niceShow, showPolymerRasmol
from mirnylib.numutils import logbins
import os
from brushManaging import makeBondsForBrush
from scipy.ndimage.filters import gaussian_filter1d
import pymol_show
from pymol_show import do_coloring, interpolateData



def myLoad(filename, dummy=None):
    #ugly workaround for simulating HU:
    if filename[:4] == "_HU_":
        HU = "HU"
        filename = filename[4:]
    elif filename[:4] == "_GY_":
        HU = "GY"
        filename = filename[4:]
    else:
        HU = False




    bonds = makeBondsForBrush()
    bonds.load(os.path.join(os.path.split(filename)[0], "chains"))
    data = polymerutils.load(filename)
    converted = bonds.convertChain(data)

    if HU == "HU":
        converted += gaussian_filter1d(10 * np.random.randn(*converted.shape), sigma=2.8, axis=0)
    elif HU == "GY":
        converted += gaussian_filter1d(7 * np.random.randn(*converted.shape), sigma=2.8, axis=0)

    #using only one arm for scalings
    if np.random.random() < 0.5:
        return converted[:int(0.4 * len(converted))]
    else:
        return converted[int(0.6 * len(converted)):]


folders = [ "realDomains/realDomains_nokick_lambda3.5_L35_gap2_stiff6_domNum15_domLen6_DATA1"]
labelDict = {i:i for i in folders}

for folder in folders:
    #if not os.path.isdir(folder):
    #    continue
    #if not os.path.exists(os.path.join(folder, "block500.dat")):
    #    continue
    names = folder.split("_")

    #if mylambda != 3:
    #    continue




    label = labelDict[folder]



    scalings1 = give_slices(base="{0}/blockDATA2.dat".format(folder),
                           tosave=None, nproc=7,
                           slices=[250],
# For each value in "slices" (i.e. points in time) a separate plot is made.
# The value above (t=250) is multiplied by each value in multipliers,
#(here producing a set of numbers between 125 and 250), then gets
#converted to integers and the plot is being averaged over that
#Note that you want more than one number here only if you want to see a time dependense.
                           sliceParams=range(1, 5),
#For each value in sliceParams, DATA1 gets replaced with this value.
                           cutoff=5.,
                           multipliers=np.arange(0.5, 1.0000000001, 0.01),
                           #multipliers=[1],
                           mode="chain",
                           loadFunction=myLoad,
                           exceptionList=[IOError],
                           normalize=True, verbose=True)

    scalings = list(scalings1)

    values = [i[0] for i in scalings]
    values = np.array(values)

    #Now some manual normalization.
    #Sorry for a messy format, but I wrote the above code few years ago and now I'm sort
    #of stuck with it. It is sort of understandable and very flexible.


    c = 0  # replace me with 1 or 2 to get gyration radius and end-to-end distance.
    values[:, c, 0, :] *= (1600000 / values[:, 0, 0, -1][:, None])
    toint = np.diff(values[:, 0, 0, :], axis=1) * values[:, 0, 1, :-1]

    toint[values[:, c, 0, :-1] < 10000] = 0
    toint[values[:, c, 0, :-1] > 1600000] = 0
    ints = np.sum(toint, axis=1)

    values[:, c, 1, :] /= ints[:, None]

    setExceptionHook()
    for scaling in values:
        plt.plot(*scaling[c], label=label)
        cPickle.dump(scaling[c], open("scaling", 'w'))



    values = [i[0] for i in scalings]
    values = np.array(values)


if c == 0:
    plt.xlim((1e3, 1e6))
    plt.ylim((1e-8, 1e-4))

plt.ylabel("Pc")
plt.xlabel("distance (bp)")
plt.xscale("log")
plt.yscale("log")
niceShow()

exit()
