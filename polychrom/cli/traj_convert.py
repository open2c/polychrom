#!/usr/bin/env python
"""This is a function that would convert trajectories from the old format "blockXXX.dat" + "SMCXXX.dat" to the new
HDF5-based format.

This glbal docstring has FAQ, general motivation, and examples. To figure out how to use the function, either browse
to the function below and read the click decorators, or just run "traj_convert.py --help".

Installation
------------

Copy the file traj_convert.py to your bin folder (e.g. ~/anaconda3/bin)

Usage
-----

traj_convert.py --help  will print the usage.

Useful arguments to consider

* --dry-run --verbose (do not modify anything, and print what you're doing)
* --empty-policy copy  (for not inplace conversions, will copy non-polymer-simulations trajectories as is)
* --inplace --empty-policy ignore --verbose (once you run it a few times, you can do the rest inplace, it's faster)

FAQ
---

Q: What happens to block numbers from old style trajectories?
A: they are put in the data dict under "block" key, the same way HDF5 trajectories in polychrom do it

Q: What happens to loop extrusion positions?
A: SMC12345.dat are automatically swept in under the key "lef_positions"
and would be returned in a dict returned by polychrom.hdf5_format.load_URI

Q: What is the best way to save space?
A: Rounding to 1 digit (0.05 max error) would save 30-40%.
Picking every second/5th/etc. file would save it by 2x/5x on top of that

Q: How to find how much do folders occupy?
A: `du -sch *`  ; alternatively `du -sc * | sort -n` if you want to sort the output by size.
`find . | wc -l` to find how many files

Default Behavior
----------------

Defaults are fairly conservative, and would use little rounding (to 2 digits, 0.005 maximum error),
would demand the trajectory to be consecutive, and would not do in-place conversions.

Examples
--------

First, run "traj_convert.py --help" to see general usage.

All examples below are real-life examples showing how to convert many trajectories at once.
Examples below convert each sub-folder in a given folder, which is probably the most common usecase.

For very critical data, it is recommended to not convert in place. The script below does this,
and converts each trajectory to a new-style, placed in a "../converted" folder with the same
name. It rounds to 2 digits (max error 0.005) by default, which is very conservative.
It is recommended to round to 1 digit unless you specifically need bond lengths or angles
to a high precision. Contactmaps are not affected by 1-digit rounding.

(put this in a bash script, set -e will take care of not continuing on errors)
set - e
for i in *; do traj_convert.py --empty-policy raise --verbose  "$i" "../converted/$i" ; done

For less critical data, in-place conversion is acceptable. Example below converts every trajectory in-place,
and rounds to 1 digit, and also skips every second file. This gives ~4x space savings. It sets empty-policy to
"ignore" because conversion is in place. You will be notified of all the cases of empty folders because of the
--verbose flag. It will use a temporary folder to copy files to, and then would replace the original with the
temporary folder. It also allows for missing blocks (e.g. block1.dat block2.dat block4.dat).

for i in *; do traj_convert.py --empty-policy ignore --verbose --round-to 1 --skip-files 2 --allow-nonconsecutive
--replace  "$i" `mktemp -d` ; done


Input can be new-style trajectory as well. You would use that for thinning or rounding the data. For example,
the script below would round data to 0.05, and take every 5th file (10x space reduction). It also shows an example of
iterating through all sub-subdirectories, (not sub-directories), which is also a common data layout.

for i in */*; do traj_convert.py --empty-policy ignore --verbose --input-style new --round-to 1 --skip-files 5
--allow-nonconsecutive --replace "$i" `mktemp -d` ; done

"""

import glob
import os
import pickle
import re
import shutil
import sys

import click
import numpy as np
import pandas as pd

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
                raise ValueError("You should have one group in regex denoting the number of the file")
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
    "--dry-run",
    is_flag=True,
    help="do not perform any file operations",
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
@click.option("--verbose", is_flag=True)
@click.option("--round-to", default=2, show_default=True, help="round to this number of digits")
@click.option("--skip-files", default=1, show_default=True, help="save only every Nth file")
@click.option(
    "--HDF5-blocks-per-file",
    default=100,
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

    An example command to replace each subfolder in a folder, and take every second file (4x space saving):

    for i in *; do traj_convert.py --round-to 1 --skip-files 2 --allow-nonconsecutive --replace  $i `mktemp -d` ; done
    """

    # managing input/output directories
    in_dir = os.path.abspath(in_dir)
    if os.path.isfile(in_dir):
        raise IOError("input directory is a file")
    if not os.path.exists(in_dir):
        raise IOError("input directory doesn't exist")
    out_dir = os.path.abspath(out_dir)
    if out_dir == in_dir:
        raise ValueError("Copying to same directory not supported - use replace=True")
    if not kwargs["dry_run"]:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    # getting files/URIs corresponding to blocks
    all_files = glob.glob(os.path.join(in_dir, "*"))
    if kwargs["input_style"] == "old":
        blocks = _find_matches(block_pattern, all_files)
    elif kwargs["input_style"] == "new":
        blocks = {i: j for j, i in list_URIs(in_dir, empty_error=True, return_dict=True).items()}
    else:
        raise ValueError("input-style should be 'old' or 'new'")

    # managing cases when the folder is empty
    policy = kwargs["empty_policy"]
    if len(blocks) == 0:
        if (policy in ["copy", "copy-limit"]) and kwargs["verbose"]:
            if kwargs["replace"]:
                if kwargs["verbose"]:
                    print("no files found; not moving", in_dir)
            else:
                if not kwargs["dry_run"]:
                    shutil.move(in_dir, out_dir)
                if kwargs["verbose"]:
                    print("no files found; simply moving", in_dir)
            exit()

        if policy == "copy":
            kwargs["max_unmatched_files"] = 1e9
        elif policy == "raise":
            print(in_dir)
            raise IOError("Emtpy directory encountered")
        elif policy == "ignore":
            if kwargs["verbose"]:
                print("skipping", in_dir)
            exit()
        elif policy == "copy-limit":
            pass
        else:
            raise ValueError(f"wrong empty policy: {policy}")

    # coverting blocks to pd.Series
    if len(blocks) > 0:
        if kwargs["verbose"]:
            print(f"moving {len(blocks)} blocks in {in_dir}")
        blocks = pd.Series(data=list(blocks.keys()), index=list(blocks.values()))
        blocks.name = "blocks"
        all_series = [blocks]

        # making sure the 4 arguments for extra files are repeated the same number of time
        assert len(extra_pattern) == len(extra_pattern_name)
        assert len(extra_loader) == len(extra_pattern_name)
        assert len(extra_loader) == len(extra_require)

        # matching patterns for extra files, populating the dataframe
        for val_pat, val_name, require in zip(extra_pattern, extra_pattern_name, extra_require):
            datas = _find_matches(val_pat, all_files)
            if require:
                if len(datas) != len(blocks):
                    raise ValueError(f"files missing for {val_name}: need {len(blocks)} found {len(datas)}")
            if len(datas) > 0:
                datas = pd.Series(data=list(datas.keys()), index=list(datas.values()))
                datas.name = val_name
                all_series.append(datas)
        df = pd.DataFrame(all_series).T

        # verifying that index is consecutive
        if not kwargs["allow_nonconsecutive"]:
            assert (np.diff(df.index.values) == 1).all()

    # managing files that are not blocks; raising an error if there are too many of them
    vals = set(df.values.reshape((-1,)))
    other = [i for i in all_files if i not in vals]
    if len(other) > kwargs["max_unmatched_files"]:
        print("example unmatched files found")
        print(other[:: len(other) // 20 + 1])
        print("Verify that none of these should be converted using extra_pattern")
        print("If not, increase max_unmatched_files")
        raise ValueError("Limit exceeded: {0} files did not match anything".format(len(other)))

    # creating the reporter
    if (len(blocks) > 0) and (not kwargs["dry_run"]):
        rep = HDF5Reporter(
            folder=out_dir,
            max_data_length=kwargs["hdf5_blocks_per_file"],
            h5py_dset_opts=None,
            overwrite=kwargs["overwrite"],
        )

    # copying the "other" files
    if not kwargs["dry_run"]:
        for i in other:
            dest = os.path.join(out_dir, os.path.split(i)[-1])
            if not kwargs["overwrite"]:
                if os.path.exists(dest):
                    raise IOError(f"File exists: {dest}")
            shutil.copy(i, dest)

    if (len(blocks) > 0) and (not kwargs["dry_run"]):
        # main loop - skip_files is aplied here
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
                cur["pos"] = np.round(np.asarray(cur["pos"], dtype=np.float32), kwargs["round_to"])

            # adding "extra" data in the dict to save
            for name, ldr in zip(extra_pattern_name, extra_loader):
                if name not in subdf:
                    continue
                filename = subdf[name]
                if not pd.isna(filename):
                    cur[name] = eval(ldr)
            rep.report("data", cur)
        rep.dump_data()

    # replacing the original trajectory if requested
    if kwargs["replace"] and (not kwargs["dry_run"]):
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
