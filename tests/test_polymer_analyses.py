import numpy as np
import polychrom
import polychrom.starting_conformations
import polychrom.polymer_analyses as polymer_analyses


def test_smart_contacts():
    data = np.random.random((200, 3)) * 10  # generate test data

    conts = polymer_analyses.calculate_contacts(data, 2.5)  # these are regular contacts

    # these are smart contacts - every second monomer is taken
    c2 = polymer_analyses.smart_contacts(data, 2.5)
    # generate unique indices based on contacts; sort them.
    ind_smart = np.sort(c2[:, 0] * 10000 + c2[:, 1])
    ind_regular = np.sort(conts[:, 0] * 10000 + conts[:, 1])

    assert np.in1d(ind_smart, ind_regular).all()


def _testMutualSimplify():
    for _ in range(10):
        mat = np.random.random((3, 3)) * 0.1 + 0.1
        a = polychrom.starting_conformations.grow_cubic(500, 10)
        b = polychrom.starting_conformations.grow_cubic(500, 10)
        a = np.dot(a, mat)
        b = np.dot(b, mat)
        a = a + np.random.random(a.shape) * 0.0001
        b = b + np.random.random(b.shape) * 0.0001
        c1 = polychrom.polymer_analyses.getLinkingNumber(
            a, b, simplify=False, randomOffset=False
        )
        a, b = polychrom.polymer_analyses.mutualSimplify(a, b, verbose=False)
        c2 = polychrom.polymer_analyses.getLinkingNumber(
            a, b, simplify=False, randomOffset=False
        )
        print("simplified from 200 to {0} and {1}".format(len(a), len(b)))
        print("Link before: {0}, link after: {1}".format(c1, c2))
        assert c1 == c2

    for _ in range(10):
        mat = np.random.random((3, 3))
        a = polychrom.starting_conformations.create_random_walk(1, 2000)
        b = polychrom.starting_conformations.create_random_walk(1, 1000)

        a = np.dot(a, mat)
        b = np.dot(b, mat)
        a = a + np.random.random(a.shape) * 0.0001
        b = b + np.random.random(b.shape) * 0.0001

        c1 = polychrom.polymer_analyses.getLinkingNumber(
            a, b, simplify=False, randomOffset=False
        )
        a, b = polychrom.polymer_analyses.mutualSimplify(a, b, verbose=False)
        c2 = polychrom.polymer_analyses.getLinkingNumber(
            a, b, simplify=False, randomOffset=False
        )
        print("simplified from 3000 and 1000 to {0} and {1}".format(len(a), len(b)))
        print("Link before: {0}, link after: {1}".format(c1, c2))
        assert c1 == c2

def test_scalings():
    import polychrom.starting_conformations, polychrom.polymer_analyses
    import numpy as np 
    datas = [polychrom.starting_conformations.create_random_walk(1,100) for _ in range(1000)]

    scals = [polychrom.polymer_analyses.R2_scaling(i) for i in datas]
    scals = np.mean(scals, axis=0)
    assert np.max(np.abs((scals[1]-scals[0])/scals[0])) < 0.1

    scals = [polychrom.polymer_analyses.Rg2_scaling(i) for i in datas]
    scals = np.mean(scals, axis=0)
    assert np.max((scals[1] - scals[0]/6)/ scals[0]) < 0.04

    scals = [polychrom.polymer_analyses.contact_scaling(i,cutoff=2) for i in datas]
    scals = np.mean(scals, axis=0)
    assert np.max(((scals[1]- 8 / scals[0]**(3/2))/scals[0])[3:] < 0.1)
        
        
def _test_Rg_scalings_vs_Rg_matrix():
    a = np.random.lognormal(1, 1, size=(30, 3))  # array for testing

    gr = polymer_analyses.Rg2_matrix(a)  # calculate Rg matrix in a normal way

    for i in range(len(a) + 1):  # fill on eside of it with manually calculated Rg(i:j)
        for j in range(i + 1, len(a)):
            gr[j, i] = polymer_analyses.Rg2(a[i : j + 1])
            pass

    assert np.allclose(gr, gr.T)

    # 5th diagonal here means s=5 (5-monomer chains)
    scal = polymer_analyses.Rg2_scaling(a, bins=[5])
    # here Nth diagonal means N+1 monomer chain, so that the corner = whole chain
    d1 = np.diagonal(gr, 4).mean()
    # compare P(s) to manually calculated from Rg matrix
    assert np.allclose(scal[1][0], d1)
    # now we are testing ring there are (N-s+1) subchains of length s.
    scal = polymer_analyses.Rg2_scaling(a, bins=[3], ring=True)
    d1 = (
        np.diagonal(gr, 2).sum()
        + polymer_analyses.Rg2(np.array([a[0], a[-1], a[-2]]))
        + polymer_analyses.Rg2(np.array([a[0], a[1], a[-1]]))
    ) / len(a)
    assert np.allclose(scal[1][0], d1)


if __name__ == "__main__":
    _test_Rg_scalings()
    _testMutualSimplify()
    test_smart_contacts()
