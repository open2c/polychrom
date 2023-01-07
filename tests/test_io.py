import numpy as np

import polychrom.forcekits
import polychrom.forces
import polychrom.polymerutils
import polychrom.starting_conformations
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI
from polychrom.simulation import Simulation


def test_basic_simulation_and_hdf5(tmp_path):
    data = polychrom.starting_conformations.create_random_walk(1, 100)

    """
    Here we created a hdf5Reporter attached to a foler test, and we are saving 5 blocks per file 
    (you should probalby use 50 here or 100. 5 is just for a showcase)
    """
    reporter = HDF5Reporter(folder=tmp_path, max_data_length=5)

    """
    Passing a reporter to the simulation object - many reporters are possible, and more will be added in a future
    """
    sim = Simulation(
        N=100,
        error_tol=0.001,
        collision_rate=0.1,
        integrator="variableLangevin",
        platform="reference",
        max_Ek=40,
        reporters=[reporter],
    )
    sim.set_data(data)
    sim.add_force(polychrom.forcekits.polymer_chains(sim))
    sim.add_force(polychrom.forces.spherical_confinement(sim, r=4, k=3))
    sim._apply_forces()
    datas = []
    for i in range(19):
        """
        Here we pass two extra records: a string and an array-like object.
        First becomes an attr, and second becomes an HDF5 dataset
        """
        sim.do_block(
            20,
            save_extras={
                "eggs": "I don't eat green eggs and ham!!!",
                "spam": [1, 2, 3],
            },
        )
        datas.append(sim.get_data())

    """
    Here we are not forgetting to dump the last set of blocks that the reporter has. 
    We have to do it at the end of every simulation. 

    I tried adding it to the destructor to make it automatic,
    but some weird interactions with garbage collection made it not very useable. 
    """
    reporter.dump_data()

    files = list_URIs(tmp_path)
    d1 = load_URI(files[1])
    d1_direct = datas[1]

    assert np.abs(d1["pos"] - d1_direct).max() <= 0.0051

    d1_fetch = polychrom.polymerutils.fetch_block(tmp_path, 1)
    assert np.allclose(d1["pos"], d1_fetch)

    assert np.allclose(d1["spam"], [1, 2, 3])  # spam got saved correctly
    assert d1["eggs"] == "I don't eat green eggs and ham!!!"

    del sim
    del reporter

    rep = HDF5Reporter(folder=tmp_path, overwrite=False, check_exists=False)
    ind, data = rep.continue_trajectory()

    # continuing from the last trajectory
    assert np.abs(data["pos"] - datas[-1]).max() <= 0.0054


def run():
    import tempfile

    tmp_dir = tempfile.TemporaryDirectory()
    test_basic_simulation_and_hdf5(tmp_dir.name)


if __name__ == "__main__":
    run()
