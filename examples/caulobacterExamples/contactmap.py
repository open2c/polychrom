from polymerScalings import give_slices
import numpy
import numpy as np
import matplotlib.pyplot as plt
from mirnylib.systemutils import setExceptionHook
import polymerutils
from mirnylib.plotting import niceShow, mat_img, showPolymerRasmol
from mirnylib.numutils import logbins, zoomOut
import os.path
import hiclib.fragmentHiC
import joblib
import random
import cPickle
from contactmaps import averageBinnedContactMap, giveContacts
from brushManaging import makeBondsForBrush
from scipy.ndimage.filters import gaussian_filter, gaussian_filter1d
from mirnylib.h5dict import h5dict

filelist = ["newSweep_lambda3.7_L25_gap8_stiff8_ind{0}/block{1}.dat".format(i, j) for i in range(40)  for j in xrange(100, 500, 5)]
saveName = "temp"


print sum([os.path.exists(i) for i in filelist])
# if True not in [os.path.exists(i) for i in filelist]:
#    raise
def myLoad(filename, dummy=None):
    # ugly workaround:
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
    return converted


cm = averageBinnedContactMap(filenames=filelist,
                             binSize=20,
                             cutoff=5.,
                             loadFunction=myLoad,
                             exceptionsToIgnore=[IOError], n=4)
cm = cm[0]

cm = np.hstack([cm, cm])
cm = np.vstack([cm, cm])


setExceptionHook()
mat_img(np.log(cm + 1), trunk=0.0001)

0 / 0

plt.figure(figsize=(8, 6))
plt.xlim((1e-3, 1e2))
plt.ylim((1e-9, 1e-5))
plt.xscale("log")
plt.yscale("log")
for filename in filelist:
    scalings = give_slices(base="{0}/blockDATA2.dat".format(filename),
                           tosave=None, nproc=4,
                           slices=[250], sliceParams=(200),
                           multipliers=numpy.arange(0.850001, 1.0001, 0.001),
                           # multipliers=[1],
                           mode="chain",
                           loadFunction=polymerutils.load)
    setExceptionHook()

    values = [i[0] for i in scalings]
    values = numpy.array(values)
    values[:, 0, :] /= 2500

    labels = ["{0}; time = ".format(
        filename) + str(i[1]["slice"]) for i in scalings]
    for scaling, label in map(None, values, labels):
        plt.plot(*scaling[0], label=label)


a = logbins(10, 10000, 1.2)
a = numpy.array(a)
plt.plot(a / 2500., 1e-5 * a ** (-0.5), label="Proposed -0.5 scaling")
plt.xlabel("distance (MB)")
niceShow()
exit()
