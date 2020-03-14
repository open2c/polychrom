#!/usr/bin/env python
import os
import sys
import shutil
import click
import pickle
import glob
import re
import pandas as pd
import numpy as np
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI
from polychrom.polymerutils import load


def _find_matches(pat, filenames):
    """
    Matches pattern to each filename in a list, and returns those that matched.
    Enforces only one match per file. 
    """
    result = {}
    for filename in filenames:
        a = re.search(pat, filename)
        if a is not None:
            if len(a.groups()) != 1:
                raise ValueError(
                    "You should have one group in regex denoting the number of the file"
                )
            assert len(a.groups()) == 1
            gr = int(a.groups()[0])
            result[filename] = gr
    return result


@click.command()
@click.option(
    "--input-style",
    default="old",
    show_default=True,
    help="old (block*.dat) or new (HDF5) style for input files",
)
@click.option(
    "--empty-policy",
    default="copy-limit",
    show_default=True,
    help="empty trajectories: 'copy', 'copy-limit' (enforce file limit), 'raise', 'ignore'",
)
@click.option(
    "--block-pattern",
    default="block([0-9]+).dat",
    show_default=True,
    help="regex to match a block number in blockX.dat",
)
@click.option(
    "--extra-pattern",
    multiple=True,
    default=["SMC([0-9]+).dat"],
    show_default=True,
    help="regex pattern to match 'extra' files with file number in parentheses. This argument can be repeated",
)
@click.option(
    "--extra-pattern-name",
    multiple=True,
    default=["lef_positions"],
    show_default=True,
    help="key under which to store (can be repeated)",
)
@click.option(
    "--extra-loader",
    multiple=True,
    default=["pickle.load(open(filename,'rb'))"],
    show_default=True,
    help="python expression f(filename) that loads data (can be repeated)",
)
@click.option(
    "--extra-require/--extra-not-require",
    multiple=True,
    default=[False],
    show_default=True,
    help="Require or not that extra files are present (can be repeated)",
)
@click.option(
    "--overwrite/--not-overwrite",
    default=False,
    show_default=True,
    help="raise error if files exist in destination",
)
@click.option(
    "--allow-nonconsecutive",
    is_flag=True,
    help="allow blocks to be non-consecutive (1,2,3...)",
)
@click.option(
    "--round-to", default=2, show_default=True, help="round to this number of digits"
)
@click.option(
    "--skip-files", default=1, show_default=True, help="save only every Nth file"
)
@click.option(
    "--HDF5-blocks-per-file",
    default=50,
    show_default=True,
    help="blocks per file for HDF5 reporter",
)
@click.option(
    "--max-unmatched-files",
    default=20,
    show_default=True,
    help="maximum number of extra files found",
)
@click.option(
    "--replace",
    is_flag=True,
    help="use out_dir as temp dir, and replace in_dir with out_dir",
)
@click.option("--force-delete", is_flag=True, help="delete subdirectories")
@click.argument("IN_DIR")
@click.argument("OUT_DIR")
def trajcopy(
    block_pattern,
    extra_pattern,
    extra_pattern_name,
    extra_loader,
    extra_require,
    in_dir,
    out_dir,
    **kwargs,
):
    """
    A function that copies a HDF5 trajectory with several possible features. 
    
    --  It can convert from old-style to new-style trajectories 
    
    --  It by default rounds it to 2 decimal digits (which has space savings)
    
    --  It can "thin" the trajectory by skipping every Nth file (--skip_files)
    
    --  It can integrade information from "extra" files 
    (by default it assumes that there is a file named "SMC<X>.dat" for each "block<x>.dat",
    and that this file is a pickle. This is saved to "lef_positions" key, and this is optional).

    If you have several files like that, you can repeat "--extra-pattern" and other 3 arguments
    several times.     
    """
    in_dir = os.path.abspath(in_dir)
    if not os.path.exists(in_dir):
        raise IOError("input directory doesn't exist")
    
    out_dir = os.path.abspath(out_dir)
    if out_dir == in_dir:
        raise ValueError("Copying to same directory not supported - use replace=True")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    all_files = glob.glob(os.path.join(in_dir, "*"))
    if kwargs["input_style"] == "old":
        blocks = _find_matches(block_pattern, all_files)
    elif kwargs["input_style"] == "new":
        blocks = {i: j for j, i in list_URIs(in_dir, return_dict=True).items()}
    else:
        raise ValueError("input-style should be 'old' or 'new'")
    
    policy = kwargs["empty_policy"]
    if len(blocks) == 0:
        if policy == "copy":
            kwargs["max_unmatched_files"] = 1e9 
        elif policy == "raise":
            print(in_dir)
            raise IOError("Emtpy directory encountered")
        elif policy in ["ignore", "copy-limit"]:
            pass
        else:
            raise ValueError(f"wrong empty policy: {policy}")
        
    
    blocks = pd.Series(data=list(blocks.keys()), index=list(blocks.values()))
    blocks.name = "blocks"
    all_series = [blocks]

    assert len(extra_pattern) == len(extra_pattern_name)
    assert len(extra_loader) == len(extra_pattern_name)
    assert len(extra_loader) == len(extra_require)

    for val_pat, val_name, require in zip(
        extra_pattern, extra_pattern_name, extra_require
    ):
        datas = _find_matches(val_pat, all_files)
        if require:
            if len(datas) != len(blocks):
                raise ValueError(
                    f"files missing for {val_name}: need {len(blocks)} found {len(datas)}"
                )
        if len(datas) > 0:
            datas = pd.Series(data=list(datas.keys()), index=list(datas.values()))
            datas.name = val_name
            all_series.append(datas)

    df = pd.DataFrame(all_series).T

    if (not kwargs["allow_nonconsecutive"]) and (kwargs["skip_files"] == 1):
        assert (np.diff(df.index.values) == 1).all()

    vals = set(df.values.reshape((-1,)))
    other = [i for i in all_files if i not in vals]
    if len(other) > kwargs["max_unmatched_files"]:
        print("example unmatched files found")
        print(other[:: len(other) // 20 + 1])
        print("Verify that none of these should be converted using extra_pattern")
        print("If not, increase max_unmatched_files")
        raise ValueError("Limit exceeded: {0} files did not match anything".format(len(other)))
        
    for i in other:
        dest = os.path.join(out_dir, os.path.split(i)[-1])
        if not kwargs["overwrite"]:
            if os.path.exists(dest):
                raise IOError(f"File exists: {dest}")
        shutil.copy(i, dest)
    
    if len(blocks) > 0:
        rep = HDF5Reporter(
            folder=out_dir,
            max_data_length=kwargs["hdf5_blocks_per_file"],
            h5py_dset_opts=None,
            overwrite=kwargs["overwrite"],
        )

        for i, subdf in df.iloc[:: kwargs["skip_files"]].iterrows():
            cur = {}
            data = subdf["blocks"]
            if kwargs["input_style"] == "old":
                data = load(data)
                data = np.round(np.asarray(data, dtype=np.float32), kwargs["round_to"])
                cur["pos"] = data
                cur["block"] = i
            elif kwargs["input_style"] == "new":
                cur = load_URI(data)
                cur["pos"] = np.round(
                    np.asarray(cur["pos"], dtype=np.float32), kwargs["round_to"]
                )

            for name, ldr in zip(extra_pattern_name, extra_loader):
                if name not in subdf:
                    continue
                filename = subdf[name]
                if filename is not None:
                    cur[name] = eval(ldr)
            rep.report("data", cur)
        rep.dump_data()
    if kwargs["replace"]:
        files = [os.path.join(in_dir, i) for i in os.listdir(in_dir)]
        if not kwargs["force_delete"]:
            for f in files:
                if not os.path.isfile(f):
                    raise IOError(f"I won't delete a sub-directory {f}")
                    exit()
                os.remove(f)
            os.rmdir(in_dir)
        shutil.move(out_dir, in_dir)


if __name__ == "__main__":
    trajcopy()
