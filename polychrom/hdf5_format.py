"""
New-style HDF5 trajectories 
===========================


The purpose of the HDF5 reporter 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are several reasons for migrating to the new HDF5 storage format: 

* Saving each conformation as individual file is producing too many files
* Using pickle-based approaches (joblib) makes format python-specific and not backwards compatible; text is clumsy
* Would be nice to save metadata, such as starting conformation, forces, or initial parameters. 
* Compression can be benefitial for rounded conformations: can reduce file size by up to 40% 

one file vs  many files  vs  several files 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Saving each conformation as an individual file is undesirable because it will produce too many files: filesystem check or backup on 30,000,000 files takes hours/days. 

Saving all trajectory as a single files is undesirable because 1. backup software will back up a new copy of the file every day as it grows; and 2. if the last write fails, the file will end up in the corrupted state and would need to be recovered. 

Solution is: save groups of conformations as individual files. E.g. save conformations 1-50 as one file, conformations 51-100 as a second file etc. 

This way, we are not risking to lose anything if the power goes out at the end. This way, we are not screwing with backup solutions. This way, we have partial trajectories that can be analyzed. Although partial trajectories are not realtime, @golobor was proposing a solution to it for debug/development. 


Polychrom storage format 
^^^^^^^^^^^^^^^^^^^^^^^^

We chose the HDF5-based storage that roughly mimics the MDTraj HDF5 format. It does not have MDTraj topology because it seemed a little too complicated. However, full MDTraj compatibility may be added in the future 


Separation of simulation and repoter 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Polychrom separates two entities: a simulation object and a reporter. When a simulation object is initialized, a reporter (actually, a list of reporters in case you want to use several) is passed to the simulation object. Simulation object would attempt to save several things: __init__ arguments, starting conformation, energy minimization results, serialized forces, and blocks of conformations together with time, Ek, Ep. 

Each time a simulation object wants to save something, it calls reporter.report(...) for each of the reporters. It passes a string indicating what is being reported, and a dictionary to save. Reporter will have to interpret this and save the data. Reporter is also keeping appropriate counts. Users can pass a dict with extra variables to :py:func:`polychrom.simulation.Simulation.do_block` as ``save_extras`` paramater. This dict will be saved by the reporter. 

.. note:: 
    Generic Python objects are not supported by HDF5 reporter. Data has to be HDF5-compatible, meaning an array of numbers/strings, or a number/string. 

The HDF5 reporter used here saves everything into an HDF5 file. For anything except the conformations, it would immmediately save the data into a single HDF5 file: numpy array compatible structures would be saved as datasets, and regular types (strings, numbers) would be saved as attributes. For conformations, it would wait until a certain number of conformations is received. It will then save them all at once into an HDF5 file under groups /1, /2, /3... /50 for blocks 1,2,3...50 respectively, and save them to `blocks_1-50.h5` file


Multi-stage simulations or loop extrusion
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We frequently have simulations in which a simulation object changes. One example would be changing forces or parameters throughout the simulation. Another example would be loop extrusion simulations. 

In this design, a reporter object can be reused and passed to a new simulation. This would keep counter of conformations, and also save applied forces etc. again. The reporter would create a file "applied_forces_0.h5" the first time it receives forces, and "applied_forces_1.h5" the second time it receives forces from a simulation. Setting `reporter.blocks_only=True` would stop the reporter from saving anything but blocks, which may be helpful for making loop extrusion conformations. This is currently implemented in the examples


URIs to identify individual conformations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because we're saving several conformations into one file, we designed an URI format to quickly fetch a conformation by a unique identifyer. 

URIs are like that: `/path/to/the/trajectory/blocks_1-50.h5::42` 

This URI will fetch block #42 from a file blocks_1-50.h5, which contains blocks 1 through 50 including 1 and 50
:py:func:`polychrom.polymerutils.load` function is compatible with URIs 
Also, to make it easy to load both old-style filenames and new-style URIs, there is a function :py:func:`polychrom.polymerutils.fetch_block`. fetch_block will autodetermine the type of a trajectory folder. So it will fetch both `/path/to/the/trajectory/block42.dat` and  `/path/to/the/trajectory/blocks_x-y.h5::42` automatically 

"""
import numpy as np
import warnings
import h5py
import glob
import os

DEFAULT_OPTS = {"compression_opts": 9, "compression": "gzip"}


def _read_h5_group(gr):
    """
    Reads all attributes of an HDF5 group, and returns a dict of them
    """
    result = {i: j[:] for i, j in gr.items()}
    for i, j in gr.attrs.items():
        result[i] = j
    return result


def _convert_to_hdf5_array(data):
    """
    Attempts to convert data to HDF5 compatible array
    or to an HDF5 attribute compatible entity (str, number)

    Does its best at determining if this is a "normal"
    object (str, int, float), or an array.

    Right now, if something got converted to a numpy object,
    it is discarded and not saved in any way.
    We could think about pickling those cases, or JSONing them...
    """
    if type(data) == str:
        data = np.array(data, dtype="S")
    data = np.array(data)

    if data.dtype == np.dtype("O"):
        return None, None

    if len(data.shape) == 0:
        return ("item", data.item())
    else:
        return ("ndarray", data)


def _write_group(dataDict, group, dset_opts=None):
    """
    Writes a dictionary of elements to an HDF5 group
    Puts all "items" into attrs, and all ndarrays into datasets

    dset_opts is a dictionary of arguments passed to create_dataset function
    (compression would be here for example). By default set to DEFAULT_OPTS
    """
    if dset_opts is None:
        dset_opts = DEFAULT_OPTS
    for name, data in dataDict.items():
        datatype, converted = _convert_to_hdf5_array(data)
        if datatype is None:
            warnings.warn(f"Could not convert record {name}")
        elif datatype == "item":
            group.attrs[name] = data
        elif datatype == "ndarray":
            group.create_dataset(name, data=data, **dset_opts)
        else:
            raise ValueError("Unknown datatype")


def list_URIs(folder, empty_error=True, read_error=True, return_dict=False):
    """
    Makes a list of URIs (path-like records for each block). for a trajectory folder
    Now we store multiple blocks per file, and URI is a
    Universal Resource Identifier for a block.

    It is be compatible with polymerutils.load, and with contactmap finders, and is
    generally treated like a filename.

    This function checks that the HDF5 file is openable (if read_error==True),
    but does not check if individual datasets (blocks) exist in a file.
    If read_error==False, a non-openable file is fully ignored.
    NOTE: This covers the most typical case of corruption due to a terminated write,
    because an HDF5 file becomes invalid in that case.

    It does not check continuity of blocks (blocks_1-10.h5; blocks_20-30.h5 is valid)
    But it does error if one block is listed twice
    (e.g. blocks_1-10.h5; blocks_5-15.h5 is invalid)

    TODO: think about the above checks, and check for readable datasets as well

    Parameters
    ----------

    folder : str
        folder to find conformations in
    empty_error : bool, optional
        Raise error if the folder does not exist or has no files, default True
    read_error : bool, optional
        Raise error if one of the HDF5 files cannot be read, default True
    return_dict : bool, optional
        True: return a dict of {block_number, URI}.
        False: return a list of URIs. This is a default.

    """

    files = glob.glob(os.path.join(folder, "blocks_*-*.h5"))
    if len(files) == 0:
        if empty_error:
            raise ValueError(f"No files found in folder {folder}")
    filenames = {}
    for file in files:
        try:
            f1 = h5py.File(file, "r")
        except:
            if read_error:
                raise ValueError(f"Cannot read file {file}")
        sted = os.path.split(file)[-1].split("_")[1].split(".h5")[0]
        st, end = [int(i) for i in sted.split("-")]
        for i in range(st, end + 1):
            if i in filenames:
                raise ValueError(f"Block {i} exists more than once")
            filenames[i] = file + f"::{i}"
    if not return_dict:
        return [i[1] for i in sorted(filenames.items(), key=lambda x: int(x[0]))]
    else:
        return {int(i[0]): i[1] for i in sorted(filenames.items(), key=lambda x: int(x[0]))}


def load_URI(dset_path):
    """
    Loads a single block of the simulation using address provided by list_filenames
    dset_path should be

    /path/to/trajectory/folder/blocks_X-Y.h5::Z

    where Z is the block number
    """

    fname, group = dset_path.split("::")
    with h5py.File(fname, mode="r") as myfile:
        return _read_h5_group(myfile[group])


def save_hdf5_file(filename, data_dict, dset_opts=None, mode="w"):
    """
    Saves data_dict to filename
    """
    if dset_opts is None:
        dset_opts = DEFAULT_OPTS
    with h5py.File(filename, mode=mode) as file:
        _write_group(data_dict, file, dset_opts=dset_opts)


def load_hdf5_file(fname):
    """
    Loads a saved HDF5 files, reading all datasets and attributes.
    We save arrays as datasets, and regular types as attributes in HDF5
    """
    with h5py.File(fname, mode="r") as myfile:
        return _read_h5_group(myfile)


class HDF5Reporter(object):
    def __init__(
        self,
        folder,
        max_data_length=50,
        h5py_dset_opts=None,
        overwrite=False,
        blocks_only=False,
        check_exists=True,
    ):
        """
        Creates a reporter object that saves a trajectory to a folder

        Parameters
        ----------

        folder : str
            Folder to save data to.

        max_data_length: int, optional (default=50)
            Will save data in groups of max_data_length blocks

        overwrite: bool, optional (default=False)
            Overwrite an existing trajectory in a folder.

        check_exists: bool (optional, default=True)
            Raise an error if previous trajectory exists in the folder

        blocks_only: bool, optional (default=False)
            Only save blocks, do not save any other information



        """

        if h5py_dset_opts is None:
            h5py_dset_opts = DEFAULT_OPTS
        self.prefixes = [
            "blocks",
            "applied_forces",
            "initArgs",
            "starting_conformation",
            "energy_minimization",
            "forcekit_polymer_chains",
        ]  # these are used for inferring if a file belongs to a trajectory or not
        self.counter = {}  # initializing all the options and dictionaries
        self.datas = {}
        self.max_data_length = max_data_length
        self.h5py_dset_opts = h5py_dset_opts
        self.folder = folder
        self.blocks_only = blocks_only

        if not os.path.exists(folder):
            os.mkdir(folder)

        if overwrite:
            for the_file in os.listdir(folder):
                file_path = os.path.join(folder, the_file)
                if os.path.isfile(file_path):
                    for prefix in self.prefixes:
                        if the_file.startswith(prefix):
                            os.remove(file_path)
                else:
                    raise IOError(
                        "Subfolder in traj folder; not deleting. Ensure folder is " "correct and delete manually. "
                    )

        if check_exists:
            if len(os.listdir(folder)) != 0:
                for the_file in os.listdir(folder):
                    for prefix in self.prefixes:
                        if the_file.startswith(prefix):
                            raise RuntimeError(f"folder {folder} is not empty: set check_exists=False to ignore")

    def continue_trajectory(self, continue_from=None, continue_max_delete=5):
        """
        Continues a simulation in a current folder (i.e. continues from the last block, or the block you specify).
        By default, takes the last block. Otherwise, takes the continue_from block

        You should initialize the class with "check_exists=False" to continue a simulation

        NOTE: This funciton does not continue the simulation itself (parameters, bonds, etc.) - it only
        manages counting the blocks and the saved files.


        Returns (block_number, data_dict) - you should start a new simulation with data_dict["pos"]


        Parameters
        ----------

        continue_from: int or None, optional (default=None)
            Block number to continue a simulation from. Default: last block found

        continue_max_delete: int (default = 5)
            Maximum number of blocks to delete if continuing a simulation.
            It is here to avoid accidentally deleting a lot of blocks.

        Returns
        -------

        (block_number, data_dict)

        block_number is a number of a current block

        data_dict is what load_URI would return on the last block of a trajectory.

        """

        uris = list_URIs(self.folder, return_dict=True)
        uri_inds = np.array(list(uris.keys()))  # making a list of all URIs and filenames
        uri_vals = np.array(list(uris.values()))
        uri_fnames = np.array([i.split("::")[0] for i in uris.values()])
        if continue_from is None:
            continue_from = uri_inds[-1]

        if int(continue_from) not in uris:
            raise ValueError(f"block {continue_from} not in folder")

        ind = np.nonzero(uri_inds == continue_from)[0][0]  # position of a starting block in arrays
        newdata = load_URI(uri_vals[ind])

        todelete = np.nonzero(uri_inds >= continue_from)[0]
        if len(todelete) > continue_max_delete:
            raise ValueError("Refusing to delete {uris_delete} blocks - set continue_max_delete accordingly")

        fnames_delete = np.unique(uri_fnames[todelete])
        inds_tosave = np.nonzero((uri_fnames == uri_fnames[ind]) * (uri_inds <= ind))[0]

        for saveind in inds_tosave:  # we are saving some data and deleting the whole last file
            self.datas[uri_inds[saveind]] = load_URI(uri_vals[saveind])
        self.counter["data"] = ind + 1

        files = os.listdir(self.folder)  # some heuristics to infer values of counters - not crucial but maybe useful
        for prefix in self.prefixes:
            if prefix != "data":
                myfiles = [i for i in files if i.startswith(prefix)]
                inds = []
                for i in myfiles:
                    try:
                        inds.append(int(i.split("_")[-1].split(".h5")[0]))
                    except:
                        pass
                    self.counter[prefix] = max(inds, default=-1) + 1

        for fdelete in fnames_delete:  # actually deleting files that we need to delete
            os.remove(fdelete)

        if len(self.datas) >= self.max_data_length:
            self.dump_data()

        return uri_inds[ind], newdata

    def report(self, name, values):
        """
        Semi-internal method to be called when you need to report something

        Parameters
        ----------

        name : str
            Name of what is being reported ("data", "init_args", anything else)
        values : dict
            Dict of what to report. Accepted types are np-array-compatible,
            numbers, strings. No dicts, objects, or what cannot be converted
            to a np-array of numbers or strings/bytes.

        """
        count = self.counter.get(name, 0)

        if name not in ["data"]:
            if not self.blocks_only:
                filename = f"{name}_{count}.h5"
                with h5py.File(os.path.join(self.folder, filename), mode="w") as file:
                    _write_group(values, file, dset_opts=self.h5py_dset_opts)

        else:
            self.datas[count] = values
            if len(self.datas) >= self.max_data_length:
                self.dump_data()
        self.counter[name] = count + 1

    def dump_data(self):
        if len(self.datas) > 0:
            cmin = min(self.datas.keys())
            cmax = max(self.datas.keys())
            filename = f"blocks_{cmin}-{cmax}.h5"
            with h5py.File(os.path.join(self.folder, filename), mode="w") as file:
                for count, values in self.datas.items():
                    gr = file.create_group(str(count))
                    _write_group(values, gr, dset_opts=self.h5py_dset_opts)
            self.datas = {}
