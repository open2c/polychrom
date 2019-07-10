import numpy as np
import polychrom
import polychrom.starting_conformations
import polychrom.polymer_analyses


def _testMutualSimplify():
    for _ in range(10):
        mat = np.random.random((3, 3))
        a = polychrom.starting_conformations.grow_cubic(2000, 14)
        b = polychrom.starting_conformations.grow_cubic(2000, 14)
        a = np.dot(a, mat)
        b - np.dot(b, mat)
        a = a + np.random.random(a.shape) * 0.0001
        b = b + np.random.random(b.shape) * 0.0001
        c1 = polychrom.polymer_analyses.getLinkingNumber(a, b, simplify=False, randomOffset=False)
        a, b = polychrom.polymer_analyses.mutualSimplify(a, b, verbose=False)
        c2 = polychrom.polymer_analyses.getLinkingNumber(a, b, simplify=False, randomOffset=False)
        print("simplified from 2000 to {0} and {1}".format(len(a), len(b)))
        print("Link before: {0}, link after: {1}".format(c1, c2))
        if c1 != c2:
            print("Test failed! Linking numbers are different")
            return -1

    for _ in range(10):
        mat = np.random.random((3, 3))
        a = polychrom.starting_conformations.create_random_walk(1, 3000)
        b = polychrom.starting_conformations.create_random_walk(1, 1000)

        a = np.dot(a, mat)
        b = np.dot(b, mat)
        a = a + np.random.random(a.shape) * 0.0001
        b = b + np.random.random(b.shape) * 0.0001

        c1 = polychrom.polymer_analyses.getLinkingNumber(a, b, simplify=False, randomOffset=False)
        a, b = polychrom.polymer_analyses.mutualSimplify(a, b, verbose=False)
        c2 = polychrom.polymer_analyses.getLinkingNumber(a, b, simplify=False, randomOffset=False)
        print("simplified from 3000 and 1000 to {0} and {1}".format(len(a), len(b)))
        print("Link before: {0}, link after: {1}".format(c1, c2))
        if c1 != c2:
            print("Test failed! Linking numbers are different")
            return -1



# def testLinkingNumber():
#     a = np.random.random((3, 1000))
#     b = np.random.random((3, 1000))
#     for i in range(100):
#         mat = np.random.random((3, 3))
#         na = np.dot(mat, a)
#         nb = np.dot(mat, b)
#         print(getLinkingNumber(na, nb, randomOffset=False))

_testMutualSimplify()