import warnings

import h5py
import numpy as np
import pytest

from polychrom.hdf5_format import (
    HDF5Reporter,
    decode_positions,
    encode_positions,
    list_URIs,
    load_hdf5_file,
    load_URI,
    save_hdf5_file,
)


def _walk(n, rng, bond=1.0):
    steps = rng.normal(size=(n, 3))
    steps *= bond / np.linalg.norm(steps, axis=1, keepdims=True)
    return np.cumsum(steps, axis=0)


def _roundtrip(pos, precision):
    enc = encode_positions(pos, precision)
    return enc, decode_positions(enc["pos_delta"], enc["pos_anchor_idx"], enc["pos_anchor_pos"], precision)


@pytest.mark.parametrize("precision,bond", [(0.1, 1), (0.04, 1), (0.01, 1), (0.001, 0.1)])
def test_roundtrip_error_bound_no_accumulation(precision, bond):
    rng = np.random.default_rng(0)
    pos = _walk(200_000, rng, bond) + 100  # long chain away from the origin
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        enc, rec = _roundtrip(pos, precision)
    err = np.abs(rec - pos)
    bound = precision / 2 + np.abs(pos).max() * 2e-7  # grid error + float32 output rounding
    assert err.max() <= bound
    # error at the end of the chain is no worse than at the start
    assert err[-1000:].max() <= bound
    assert enc["pos_delta"].dtype == np.int8


def test_chain_breaks_and_overflow_become_anchors():
    rng = np.random.default_rng(1)
    chains = [_walk(1000, rng) + rng.uniform(-50, 50, 3) for _ in range(20)]
    pos = np.concatenate(chains)
    pos[5000] += 3.0  # one stretched bond inside a chain
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        enc, rec = _roundtrip(pos, 0.01)
    assert np.abs(rec - pos).max() <= 0.005 + 1e-5
    anchors = set(enc["pos_anchor_idx"].tolist())
    assert 0 in anchors and 5000 in anchors and 5001 in anchors
    assert len(anchors) <= 20 + 2


@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_tiny_arrays(n):
    pos = np.arange(n * 3, dtype=float).reshape(n, 3) * 0.37
    _, rec = _roundtrip(pos, 0.04)
    assert rec.shape == (n, 3)
    assert np.abs(rec - pos).max(initial=0) <= 0.02 + 1e-6


def test_invalid_inputs():
    with pytest.raises(ValueError):
        encode_positions(np.ones((10, 3)), 0)
    with pytest.raises(ValueError):
        encode_positions(np.ones(3), 0.01)
    with pytest.raises(ValueError):
        encode_positions(np.full((10, 3), np.nan), 0.01)
    with pytest.raises(ValueError):
        encode_positions(np.full((10, 3), 1e9), 0.01)  # does not fit int32 grid


def test_warning_too_many_anchors():
    pos = _walk(1000, np.random.default_rng(2), bond=10.0)  # bonds of 10 at precision 0.01 -> 1000 steps
    with pytest.warns(UserWarning, match="anchors"):
        _, rec = _roundtrip(pos, 0.01)
    assert np.abs(rec - pos).max() <= 0.005 + 1e-5  # still correct, just inefficient


def test_warning_precision_too_coarse():
    pos = _walk(1000, np.random.default_rng(3), bond=0.01)
    with pytest.warns(UserWarning, match="shorter than"):
        _roundtrip(pos, 0.1)


def test_no_warning_typical_polymer():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _roundtrip(_walk(1000, np.random.default_rng(4)), 0.04)


def test_reporter_roundtrip_and_continue(tmp_path):
    rng = np.random.default_rng(5)
    confs = [_walk(500, rng) for _ in range(7)]
    rep = HDF5Reporter(folder=tmp_path, max_data_length=3)
    for i, c in enumerate(confs):
        rep.report("data", {"pos": c, "pos_precision": 0.04, "block": i, "time": 1.5 * i})
    rep.dump_data()

    uris = list_URIs(tmp_path)
    assert len(uris) == 7
    with h5py.File(uris[0].split("::")[0]) as f:
        g = f["0"]
        assert "pos" not in g and g["pos_delta"].dtype == np.int8
    for i, uri in enumerate(uris):
        d = load_URI(uri)
        assert d["pos"].dtype == np.float32 and d["pos"].shape == (500, 3)
        assert np.abs(d["pos"] - confs[i]).max() <= 0.02 + 1e-5
        assert d["pos_precision"] == 0.04 and d["block"] == i
        assert not any(k.startswith("pos_") and k != "pos_precision" for k in d)

    # continuing re-buffers blocks from a deleted file and re-encodes them losslessly
    before = load_URI(uris[3])["pos"]
    rep2 = HDF5Reporter(folder=tmp_path, check_exists=False)
    ind, data = rep2.continue_trajectory(continue_from=4)
    rep2.dump_data()
    assert ind == 4
    assert np.array_equal(load_URI(list_URIs(tmp_path)[3])["pos"], before)


def test_plain_pos_without_precision_is_unchanged(tmp_path):
    pos = np.random.default_rng(6).random((50, 3))
    save_hdf5_file(tmp_path / "f.h5", {"pos": pos})
    with h5py.File(tmp_path / "f.h5") as f:
        assert "pos" in f
    assert np.array_equal(load_hdf5_file(tmp_path / "f.h5")["pos"], pos)


def test_caller_dict_not_modified(tmp_path):
    pos = np.ones((10, 3))
    d = {"pos": pos, "pos_precision": 0.01}
    save_hdf5_file(tmp_path / "f.h5", d)
    assert d["pos"] is pos and set(d) == {"pos", "pos_precision"}
    assert np.allclose(load_hdf5_file(tmp_path / "f.h5")["pos"], 1)


def test_simulation_save_precision_arguments():
    from polychrom.simulation import Simulation

    def make(**kw):
        return Simulation(
            N=10, platform="reference", integrator="variableLangevin", error_tol=0.01, collision_rate=0.1, **kw
        )

    assert make().kwargs["save_precision"] == pytest.approx(0.04)
    assert make(save_precision=0.01).kwargs["save_precision"] == 0.01
    assert make(save_decimals=2).kwargs["save_precision"] == pytest.approx(0.01)
    assert make(save_decimals=1).kwargs["save_precision"] == pytest.approx(0.1)
    for legacy in (3, 0, False):
        with pytest.warns(UserWarning, match="legacy"):
            sim = make(save_decimals=legacy)
        assert sim.kwargs["save_precision"] is False and sim.kwargs["save_decimals"] is legacy
    with pytest.raises(ValueError):
        make(save_decimals=2, save_precision=0.01)
    with pytest.raises(ValueError):
        make(save_precision=-1)
