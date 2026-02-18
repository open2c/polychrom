import pickle
from multiprocessing import Pool

import numpy as np
import pytest

import polychrom
import polychrom.polymer_analyses as polymer_analyses
import polychrom.starting_conformations


def test_calculate_contacts():
    """Test basic contact calculation"""
    # Create a simple test case with known contacts
    data = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],  # 1 unit away from [0,0,0]
            [2, 0, 0],  # 2 units away from [0,0,0], 1 unit from [1,0,0]
            [5, 0, 0],  # 5 units away from [0,0,0]
        ]
    )

    # With cutoff 1.5, should find contacts between (0,1) and (1,2)
    contacts = polymer_analyses.calculate_contacts(data, cutoff=1.5)
    assert len(contacts) == 2
    assert (0, 1) in [(c[0], c[1]) for c in contacts]
    assert (1, 2) in [(c[0], c[1]) for c in contacts]

    # With cutoff 2.5, should find contacts (0,1), (1,2), (0,2)
    contacts = polymer_analyses.calculate_contacts(data, cutoff=2.5)
    assert len(contacts) == 3

    # Test error handling
    with pytest.raises(ValueError):
        polymer_analyses.calculate_contacts(data[:, :2], cutoff=1.5)  # Wrong shape

    with pytest.raises(RuntimeError):
        bad_data = data.astype(float)  # Convert to float first
        bad_data[0, 0] = np.nan
        polymer_analyses.calculate_contacts(bad_data, cutoff=1.5)


def test_smart_contacts():
    data = np.random.random((200, 3)) * 10  # generate test data

    conts = polymer_analyses.calculate_contacts(data, 2.5)  # these are regular contacts

    # these are smart contacts - every second monomer is taken
    c2 = polymer_analyses.smart_contacts(data, 2.5)
    # generate unique indices based on contacts; sort them.
    ind_smart = np.sort(c2[:, 0] * 10000 + c2[:, 1])
    ind_regular = np.sort(conts[:, 0] * 10000 + conts[:, 1])

    assert np.isin(ind_smart, ind_regular).all()

    # Test with different cutoffs
    c3 = polymer_analyses.smart_contacts(data, cutoff=1.5, min_cutoff=2.0)  # Should use regular
    c4 = polymer_analyses.smart_contacts(data, cutoff=3.0, min_cutoff=2.0)  # Should use smart

    assert len(c3) == len(polymer_analyses.calculate_contacts(data, 1.5))
    assert len(c4) <= len(polymer_analyses.calculate_contacts(data, 3.0))


def _testMutualSimplify():
    for _ in range(10):
        mat = np.random.random((3, 3)) * 0.1 + 0.1
        a = polychrom.starting_conformations.grow_cubic(500, 10)
        b = polychrom.starting_conformations.grow_cubic(500, 10)
        a = np.dot(a, mat)
        b = np.dot(b, mat)
        a = a + np.random.random(a.shape) * 0.0001
        b = b + np.random.random(b.shape) * 0.0001
        c1 = polychrom.polymer_analyses.getLinkingNumber(a, b, simplify=False, randomOffset=False)
        a, b = polychrom.polymer_analyses.mutualSimplify(a, b, verbose=False)
        c2 = polychrom.polymer_analyses.getLinkingNumber(a, b, simplify=False, randomOffset=False)
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

        c1 = polychrom.polymer_analyses.getLinkingNumber(a, b, simplify=False, randomOffset=False)
        a, b = polychrom.polymer_analyses.mutualSimplify(a, b, verbose=False)
        c2 = polychrom.polymer_analyses.getLinkingNumber(a, b, simplify=False, randomOffset=False)
        print("simplified from 3000 and 1000 to {0} and {1}".format(len(a), len(b)))
        print("Link before: {0}, link after: {1}".format(c1, c2))
        assert c1 == c2


def test_scalings():
    import numpy as np

    import polychrom.polymer_analyses
    import polychrom.starting_conformations

    datas = [polychrom.starting_conformations.create_random_walk(1, 80) for _ in range(100)]

    scals = [polychrom.polymer_analyses.R2_scaling(i) for i in datas]
    scals = np.mean(scals, axis=0)
    assert np.max(np.abs((scals[1] - scals[0]) / scals[0])) < 0.2

    scals = [polychrom.polymer_analyses.Rg2_scaling(i) for i in datas]
    scals = np.mean(scals, axis=0)
    assert np.max((scals[1] - scals[0] / 6) / scals[0]) < 0.06

    scals = [polychrom.polymer_analyses.contact_scaling(i, cutoff=2) for i in datas]
    scals = np.mean(scals, axis=0)
    assert np.max(((scals[1] - 8 / scals[0] ** (3 / 2)) / scals[0])[3:] < 0.15)

    polychrom.polymer_analyses.contact_scaling(datas[0], cutoff=2, ring=True)
    polychrom.polymer_analyses.Rg2_scaling(datas[0], ring=True)
    polychrom.polymer_analyses.R2_scaling(datas[0], ring=True)
    meanrg = np.mean([polychrom.polymer_analyses.Rg2(x) for x in datas])
    assert (np.abs(meanrg - len(datas[0]) / 6) / len(datas[0])) < 0.15


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


def test_generate_bins():
    """Test bin generation for scaling calculations"""
    # Test basic functionality
    bins = polymer_analyses.generate_bins(100, start=4, bins_per_order_magn=10)
    assert bins[0] == 4
    assert bins[-1] == 99
    assert len(bins) > 0

    # Test with different parameters
    bins2 = polymer_analyses.generate_bins(1000, start=10, bins_per_order_magn=5)
    assert bins2[0] == 10
    assert bins2[-1] == 999

    # Test edge cases
    bins_small = polymer_analyses.generate_bins(5, start=4)
    assert len(bins_small) >= 1


def test_Rg2():
    """Test simple gyration radius calculation"""
    # Test with a simple linear chain
    data = np.array([[i, 0, 0] for i in range(10)])
    rg2 = polymer_analyses.Rg2(data)
    # For a linear chain, theoretical Rg2 = L^2/12 for continuous, but we have discrete
    assert rg2 > 0

    # Test with centered data
    centered_data = data - np.mean(data, axis=0)
    rg2_centered = polymer_analyses.Rg2(centered_data)
    assert np.isclose(rg2, rg2_centered)

    # Test with random data
    random_data = np.random.random((50, 3))
    rg2_random = polymer_analyses.Rg2(random_data)
    assert rg2_random > 0


def test_Rg2_matrix():
    """Test Rg2 matrix calculation"""
    data = np.random.random((20, 3))
    rg_matrix = polymer_analyses.Rg2_matrix(data)

    # Check symmetry
    assert np.allclose(rg_matrix, rg_matrix.T)

    # Check diagonal is zero
    assert np.allclose(np.diag(rg_matrix), 0)

    # Check some specific values match Rg2 function
    for i in range(5):
        for j in range(i + 2, min(i + 10, len(data))):
            expected = polymer_analyses.Rg2(data[i : j + 1])
            assert np.isclose(rg_matrix[i, j], expected, rtol=1e-5)


def test_kabsch_msd():
    """Test Kabsch MSD/RMSD calculation"""
    # Test with identical structures
    P = np.random.random((10, 3))
    Q = P.copy()
    msd = polymer_analyses.kabsch_msd(P, Q)
    assert np.isclose(msd, 0, atol=1e-10)

    # Test with translated structure
    Q_translated = P + np.array([1, 2, 3])
    msd_translated = polymer_analyses.kabsch_msd(P, Q_translated)
    assert np.isclose(msd_translated, 0, atol=1e-10)  # Should be 0 after centering

    # Test with rotated structure
    theta = np.pi / 6
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    Q_rotated = np.dot(P, R)
    msd_rotated = polymer_analyses.kabsch_msd(P, Q_rotated)
    assert np.isclose(msd_rotated, 0, atol=1e-10)

    # Test that kabsch_rmsd is the same function
    assert polymer_analyses.kabsch_rmsd is polymer_analyses.kabsch_msd


def test_contact_scaling():
    """Test contact probability scaling calculation"""
    # Create a compact globule-like structure for testing
    np.random.seed(42)
    data = np.random.random((100, 3)) * 5

    mids, probs = polymer_analyses.contact_scaling(data, cutoff=2.0)

    assert len(mids) == len(probs)
    assert all(p >= 0 and p <= 1 for p in probs)  # Probabilities should be in [0,1]
    assert len(mids) > 0

    # Test with ring
    mids_ring, probs_ring = polymer_analyses.contact_scaling(data, cutoff=2.0, ring=True)
    assert len(mids_ring) == len(probs_ring)

    # Test with custom bins
    custom_bins = [1, 5, 10, 20, 50, 99]
    mids_custom, probs_custom = polymer_analyses.contact_scaling(data, bins0=custom_bins, cutoff=2.0)
    assert len(mids_custom) == len(custom_bins) - 1


def test_slope_contact_scaling():
    """Test slope calculation for contact scaling"""
    # Create mock data
    mids = np.logspace(0, 2, 20)
    # Create a power-law like decay
    cp = 1.0 / (mids**1.5)

    slope_mids, slopes = polymer_analyses.slope_contact_scaling(mids, cp, sigma=1.5)

    assert len(slope_mids) == len(mids) - 1
    assert len(slopes) == len(mids) - 1
    # For a power law, slope should be approximately constant (around -1.5)
    assert np.std(slopes[5:-5]) < 0.5  # Middle values should be relatively stable


def test_R2_scaling():
    """Test end-to-end distance scaling"""
    # Create a random walk for testing
    np.random.seed(42)
    data = np.cumsum(np.random.randn(50, 3) * 0.5, axis=0)

    bins, r2_values = polymer_analyses.R2_scaling(data)

    assert len(bins) == len(r2_values)
    assert all(r2 >= 0 for r2 in r2_values)  # R^2 should be non-negative
    assert r2_values[0] < r2_values[-1]  # Should generally increase with distance

    # Test with ring
    bins_ring, r2_ring = polymer_analyses.R2_scaling(data, ring=True)
    assert len(bins_ring) == len(r2_ring)

    # Test with custom bins
    custom_bins = [1, 5, 10, 20]
    bins_custom, r2_custom = polymer_analyses.R2_scaling(data, bins=custom_bins)
    assert np.array_equal(bins_custom, custom_bins)


def test_Rg2_scaling():
    """Test radius of gyration scaling"""
    # Create a random walk for testing
    np.random.seed(42)
    data = np.cumsum(np.random.randn(50, 3) * 0.5, axis=0)

    bins, rg2_values = polymer_analyses.Rg2_scaling(data)

    assert len(bins) == len(rg2_values)
    assert all(rg2 >= 0 for rg2 in rg2_values)  # Rg^2 should be non-negative
    assert rg2_values[0] < rg2_values[-1]  # Should generally increase with chain length

    # Test relationship with R2 (for random walk, Rg^2 ~ R^2/6)
    bins_r2, r2_values = polymer_analyses.R2_scaling(data, bins=bins)
    ratios = np.array(rg2_values) / np.array(r2_values)
    assert all(0 < r < 1 for r in ratios)  # Rg^2 should be smaller than R^2

    # Test with ring
    bins_ring, rg2_ring = polymer_analyses.Rg2_scaling(data, ring=True)
    assert len(bins_ring) == len(rg2_ring)


def test_calculate_cistrans():
    """Test cis/trans contact calculation"""
    # Create two non-overlapping chains
    chain1 = np.array([[i, 0, 0] for i in range(10)])
    chain2 = np.array([[i, 10, 0] for i in range(10)])
    data = np.vstack([chain1, chain2])

    chains = [[0, 10], [10, 20]]

    # Test for first chain
    cis, trans = polymer_analyses.calculate_cistrans(data, chains, chain_id=0, cutoff=1.5)
    assert cis > 0  # Should have self-contacts
    assert trans == 0  # Chains are far apart

    # Test with closer chains
    chain2_close = np.array([[i, 2, 0] for i in range(10)])
    data_close = np.vstack([chain1, chain2_close])
    cis_close, trans_close = polymer_analyses.calculate_cistrans(data_close, chains, chain_id=0, cutoff=3.0)
    assert trans_close > 0  # Should have inter-chain contacts now

    # Test with PBC
    with pytest.raises(ValueError):
        # Should raise error when pbc_box is True but box_size is None
        polymer_analyses.calculate_cistrans(data, chains, chain_id=0, cutoff=1.5, pbc_box=True)

    # Test with box_size
    cis_pbc, trans_pbc = polymer_analyses.calculate_cistrans(
        data, chains, chain_id=0, cutoff=1.5, pbc_box=True, box_size=[20, 20, 20]
    )
    assert cis_pbc >= 0
    assert trans_pbc >= 0


def test_pickle_compatibility():
    """Test that extracted functions can be pickled (for multiprocessing)"""
    # Test that the extracted helper functions can be pickled
    try:
        # Test _smooth_for_slope
        pickled = pickle.dumps(polymer_analyses._smooth_for_slope)
        unpickled = pickle.loads(pickled)

        # Test _radius_gyration_helper
        pickled = pickle.dumps(polymer_analyses._radius_gyration_helper)
        unpickled = pickle.loads(pickled)

        # Test they work in a pool
        data = np.random.random((30, 3))
        with Pool(2) as pool:
            # Test that we can use these functions in parallel
            results = pool.starmap(
                polymer_analyses._smooth_for_slope, [(np.array([1, 2, 3, 4, 5]), 1.0) for _ in range(4)]
            )
            assert len(results) == 4

    except Exception as e:
        pytest.fail(f"Pickle/multiprocessing test failed: {e}")


if __name__ == "__main__":
    _test_Rg_scalings_vs_Rg_matrix()
    _testMutualSimplify()
    test_smart_contacts()
    test_calculate_contacts()
    test_generate_bins()
    test_Rg2()
    test_Rg2_matrix()
    test_kabsch_msd()
    test_contact_scaling()
    test_slope_contact_scaling()
    test_R2_scaling()
    test_Rg2_scaling()
    test_calculate_cistrans()
    test_pickle_compatibility()
