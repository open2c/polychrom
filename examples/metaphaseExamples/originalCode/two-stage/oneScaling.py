from polymerScalings import give_slices
import numpy
import cPickle
import matplotlib.pyplot as plt
from mirnylib.systemutils import setExceptionHook
import polymerutils
from mirnylib.plotting import niceShow
from mirnylib.numutils import logbins
import matplotlib

setExceptionHook()
step = 0.1
slices = [30, 100, 300, 1000, 3000]
par = [1, 2]

scaling1 = cPickle.load(open("/home/magus/HiC2011/natasha/scalings/M-phase-max"))
scaling2 = cPickle.load(open("/home/magus/HiC2011/natasha/scalings/M-phase-min"))

for change in numpy.linspace(0, 1, 50):
    plt.plot(scaling2[0] / 1000000, (change * scaling1[1] + (1 - change) * scaling2[1]) , color="grey", linewidth=3)


def loadFunction(x, y):
    return polymerutils.load(x, y)
    start = numpy.random.randint(0, 80000)
    end = start + 20000
    return polymerutils.load(x, y)[start:end]

colors = [matplotlib.cm.jet(i) for i in numpy.linspace(0.1, 0.9, 5)]
linetypes = [":", "-"]


#for len, color in zip([60, 80, 100, 120, 140], colors):
for len, color in zip([100], colors):
    for num, linestyle in zip([30], linetypes):

        scalings0 = give_slices(base="L_{0}_num_DATA1_len_{1}/blockDATA2.dat".format(len, num),
                               tosave=None, nproc=7,
                               slices=[500], sliceParams=range(1, 30),
                               multipliers=numpy.arange(0.7, 1, 0.03), mode="chain",
                               loadFunction=loadFunction, cutoff=3., binstep=1.23,
                               exceptionList=[IOError], normalize=True)



        values0 = numpy.array([i[0] for i in scalings0])
        labels = [str(i) for i in slices]

        values0[:, 0, 0] /= (1000000. / (600 * (128000 / 30000)))
        values0[:, 0, 1] *= (1. / (600 * (128000 / 30000)))
        for value in values0:
            plt.plot(*value[0], color=color, linestyle=linestyle, label="L={0}; num={1}".format(len, num))

#values1[:, 0, :] /= (1000000. / 600)
#cPickle.dump((scalings, values), open("consScaff", 'wb'))
#exit()

setExceptionHook()




a = logbins(10, 10000, 1.2)
a = numpy.array(a)


plt.xlabel("distance (MB)")
niceShow("log")

exit()

for scaling, label in zip(values, labels):
    plt.plot(*scaling[1], label=label)
a = logbins(10, 10000, 1.2)
a = numpy.array(a)
plt.plot(a, 2 * a ** (1 / 6.))
niceShow("log")
for scaling, label in zip(values, labels):
    plt.plot(*scaling[2], label=label)
niceShow("log")


pc = scalings[0][0][0][1]
rg = scalings[0][0][1][1]
bins = scalings[0][0][0][0]
plt.plot(bins, pc * (rg ** 3))
plt.show()

#plt.show()
#exit()
