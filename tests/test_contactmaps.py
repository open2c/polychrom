import numpy as np
import polychrom.polymer_analyses as polymer_analyses
from polychrom.contactmaps import (
    averageContacts,
    averageContactsSimple,
    monomerResolutionContactMap,
    binnedContactMap,
)
from polychrom.legacy.contactmaps import averagePureContactMap as cmapPureMap
from polychrom.legacy.contactmaps import averageBinnedContactMap as cmapBinnedMap


class DummyContactMap(object):
    "contactmap 'finder' for testing that returns fixed contacts (a) + constant(x)"

    def __init__(self, x, a):
        self.a = a + x
        self.M = 15

    def next(self):
        if self.M == 0:
            raise StopIteration
        self.M -= 1
        return self.a


def test_contactmaps():
    """
    This function performs several tests of contactmaps. It uses an artificial class
    above that returns contacts given to it, adding a number given as an "input
    filename".

    It compares contactmaps calculated using different methods, different number of
    processes, and also compares contactmaps calculated using the old legacy
    contactmap finder.

    """
    ars = [np.random.random((60, 3)) * 4 for _ in range(16)]
    conts = polymer_analyses.calculate_contacts(ars[0], 1)
    args = np.repeat(np.arange(4, dtype=int), 4)
    cmap1 = averageContacts(DummyContactMap, args, 100, classInitArgs=[conts], nproc=4)
    cmap4 = averageContacts(DummyContactMap, args, 100, classInitArgs=[conts], nproc=1)
    cmap2 = averageContactsSimple(DummyContactMap, args, 100, classInitArgs=[conts])
    # manually creating a contact map
    cmap3 = np.zeros((100, 100))
    for i in args:
        cmap3[conts[:, 0] + i, conts[:, 1] + i] += 15
    cmap3 = cmap3 + cmap3.T

    assert np.allclose(cmap1, cmap2)
    assert np.allclose(cmap1, cmap4)
    assert np.allclose(cmap1, cmap3)

    for n in [1, 4]:
        cmap6 = monomerResolutionContactMap(range(8), cutoff=1, loadFunction=lambda x: ars[x], n=n)
        cmap5 = cmapPureMap(
            range(8),
            cutoff=1,
            loadFunction=lambda x: ars[x],
            n=n,
            printProbability=0.000001,
        )
        print(cmap5.sum(), cmap6.sum())
        assert np.allclose(cmap6, cmap5)

        cmap7 = binnedContactMap(
            range(8),
            chains=[(0, 27), (27, 60)],
            binSize=2,
            cutoff=1,
            loadFunction=lambda x: ars[x],
            n=n,
        )[0]

        cmap8 = cmapBinnedMap(
            range(8),
            chains=[(0, 27), (27, 60)],
            binSize=2,
            cutoff=1,
            loadFunction=lambda x: ars[x],
            n=n,
            printProbability=0.000001,
        )[0]
        assert np.allclose(cmap7, cmap8)


if __name__ == "__main__":
    test_contactmaps()
