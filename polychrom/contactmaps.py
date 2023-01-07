"""
Building contact maps
=====================


This module is the main workhorse of tools to calculate contactmaps, both from 
polymer simulations and from other simulations (e.g. 1D simulations of loop 
extrusion). All of the functions here are designed to be parallelized, and lots of 
efforts were put into making this possible. 

The reasons we need parallel contactmap code is the following: 

* Calculating contact maps is slow, esp. at large contact radii, and benefits greatly 
  from parallelizing
* Doing regular multiprocesing.map has limitations
* It can only handle heataps up to some size, and transferring gigabyte-sized heatmaps between processes takes minutes 
* It can only do as many heatmaps as fits in RAM, which on 20-core 128GB machine is no more than 5GB/heatmap 

The structure of this class is as follows. 

On the outer level, it  provides three methods to average contactmaps:

* :func:`monomerResolutionContactMap`
* :func:`binnedContactMap`,
* :func:`monomerResolutionContactMapSubchains`.

The first two create contact map from an
entire file: either monomer-resolution or binned. The last one creates contact maps 
from sub-chains in a file, starting at a given set of starting points. It is useful 
when doing contact maps from several copies of a system in one simulation. 

The first two methods have a legacy implementation from the old library that is still 
here to do the tests. 

On the middle level, it provides a method "averageContacts". This method accepts a 
"contact iterator", and can be used to average contacts from both a set of filenames 
and from a simulation of some kind (e.g. averaging positions of loop extruding 
factors from a 1D loop extrusion simulation). All of the outer level functions (
monomerResolutionContactMap for example) are implemented using this method. 

On the lower level, there are internals of the "averageContacts" method and an 
associated "worker" function. There is generally no need to understand the code of 
those functions. There exists a reference implementation of both the worker and the 
:func:`averageContacts` function,  :class:`simpleWorker` and :func:`averageContactsSimple`. They do
all the things that "averageContacts" do, but on only one core. In fact, 
"averageContacts" defaults to "averageContactsSimple" if requested to run on one core 
because it is a little bit faster. 

"""

import ctypes
import multiprocessing as mp
import random
import warnings
from contextlib import closing

import numpy as np

from . import polymer_analyses, polymerutils


def indexing(smaller, larger, M):
    """converts x-y indexes to index in the upper triangular matrix"""
    return larger + smaller * (M - 1) - smaller * (smaller - 1) // 2


def triagToNormal(triag, M):
    """Convert triangular matrix to a regular matrix"""
    ar = np.arange(M)
    mask = ar[:, None] <= ar[None, :]
    x, y = np.nonzero(mask)
    new = np.zeros((M, M), dtype=triag.dtype)
    new[x, y] = triag
    return new + new.T


def tonumpyarray(mp_arr):
    "Converts mp.array to numpy array"
    return np.frombuffer(mp_arr.get_obj(), dtype=np.int32)  # .reshape((N,N))


def findN(filenames, loadFunction, exceptions):
    "Finds length of data in filenames, handling the fact that files could be not loadable"
    for i in range(30):
        if i == 29:
            raise ValueError("Could not load any of the 30 randomly selected files")
        try:
            N = len(loadFunction(random.choice(filenames)))
            break
        except tuple(exceptions):
            continue
    return N


def init(*args):
    """
    Initializes global arguments for the worker

    """

    global sharedArrays__
    global contactIterator__
    global contactBlock__
    global N__
    global classInitArgs__
    global classInitKwargs__
    global contactProcessing__
    contactProcessing__ = args[-6]
    classInitKwargs__ = args[-4]
    classInitArgs__ = args[-5]
    sharedArrays__ = args[:-6]
    contactIterator__ = args[-3]
    contactBlock__ = args[-2]
    N__ = args[-1]


def chunk(mylist, chunksize):
    """

    Args:
        mylist: array
        chunksize: int

    Returns:
        list of chunks of an array len chunksize (last chunk is less)

    """
    N = len(mylist)
    chunks = list(range(0, N, chunksize)) + [N]
    return [mylist[i:j] for i, j in zip(chunks[:-1], chunks[1:])]


def simple_worker(x, uniqueContacts):
    """
    A "reference" version of "worker" function below that runs on only one core.
    Unlike the actual multicore worker, it can write contacts to the matrix directly
    without sorting. This is useful when your contact finding is faster than sorting
    a 1D array fo contacts

     If uniqueContacts True, assume that contactFinder outputs only unique contacts (
     like pure contact map) if False, do not assume that (like in binned contact
     map). Using False is always safe, but True will add a minor speed up, especially
     for very large contact radius.

    """
    my_iterator = contactIterator__(x, *classInitArgs__, **classInitKwargs__)
    while True:
        try:
            contacts = my_iterator.next()
            if contacts is None:
                continue
            contacts = np.asarray(contacts)
            assert len(contacts.shape) == 2
            if contacts.shape[1] != 2:
                raise ValueError("Contacts.shape[1] must be 2. Exiting.")
            contacts = contactProcessing__(contacts)
            ctrue = indexing(contacts[:, 0], contacts[:, 1], N__)
            if not uniqueContacts:
                position, counts = np.unique(ctrue, return_counts=True)
                sharedArrays__[0][position] += counts
            else:
                sharedArrays__[0][ctrue] += 1
        except StopIteration:
            return


def averageContactsSimple(contactIterator, inValues, N, **kwargs):
    """
    This is a reference one-core implementation

    Args:
        contactIterator:
            an iterator. See descriptions of "filenameContactMap" class below for example and explanations
        inValues:
            an array of values to pass to contactIterator. Would be an array of arrays of filenames or something like that.
        N:
            Size of the resulting contactmap

        **kwargs:
            arrayDtype: ctypes dtype (default c_int32) for the contact map
            classInitArgs: args to pass to the constructor of contact iterator as second+ args (first is the file list)
            classInitKwargs: dict of keyword args to pass to the coonstructor
            uniqueContacts: whether contact iterator outputs unique contacts (true) or contacts can be duplicate (False)
            contactProcessing: function f(contacts), should return processed contacts

    Returns:
        contactmap

    """
    arrayDtype = kwargs.get("arrayDtype", ctypes.c_int32)
    contactBlock = 0
    classInitArgs = kwargs.get("classInitArgs", [])
    classInitKwargs = kwargs.get("classInitKwargs", {})
    uniqueContacts = kwargs.get("uniqueContacts", False)
    contactProcessing = kwargs.get("contactProcessing", lambda x: x)
    finalSize = N * (N + 1) // 2
    sharedArrays = [np.zeros(finalSize, dtype=arrayDtype)]  # just an array, not a shared array here bc 1 core
    argset = list(sharedArrays) + [
        contactProcessing,
        classInitArgs,
        classInitKwargs,
        contactIterator,
        contactBlock,
        N,
    ]
    init(*argset)
    [simple_worker(x, uniqueContacts) for x in inValues]  # just calling workers
    final = triagToNormal(sharedArrays[0], N)
    return final


def worker(x):
    """This is a parallel implementation of the worker using shared memory buckets
    This worker is being called by the averageContact method
    It receives contacts from the contactIterator by calling .next()
    And puts contacts into the shared memory buckets

    All the locks etc. for shared memory objects are handeled here as well
    """
    import numpy as np

    np.random.seed()
    import random

    random.seed()  # making sure our bucket selectors are really randomized

    sharedNumpy = list(map(tonumpyarray, sharedArrays__))  # shared numpy arrays
    allContacts = []
    contactSum = 0
    myIterator = contactIterator__(x, *classInitArgs__, **classInitKwargs__)  # acquiring and initializing iterator
    stopped = False
    while True:  # main event loop
        try:
            contacts = myIterator.next()  # fetch contacts
            if contacts is None:
                continue
            contacts = np.asarray(contacts)
            assert len(contacts.shape) == 2
            if contacts.shape[1] != 2:
                raise ValueError("Contacts.shape[1] must be 2. Exiting.")
            contactSum += contacts.shape[0]
            allContacts.append(contacts)
        except StopIteration:
            stopped = True

        if (contactSum > contactBlock__) or stopped:  # we aggregated enough contacts.  ready to dump them.
            if len(allContacts) == 0:
                return  # no contacts found at all - exiting (we must be stopped)
            contactSum = 0
            contacts = np.concatenate(allContacts, axis=0)
            contacts = contactProcessing__(contacts)
            if len(contacts) == 0:
                if stopped:  # contactProcessing killed all contacts? are we done?
                    return  # if yes, exiting
                continue  # if not, going to the next bucket
            ctrue = indexing(contacts[:, 0], contacts[:, 1], N__)  # converting to 1D contacts
            position, counts = np.unique(ctrue, return_counts=True)  # unique contacts
            assert position[0] >= 0
            assert position[-1] < N__ * (N__ + 1) // 2  # boundary check for contacts
            chunks = np.array(np.r_[0, np.cumsum(list(map(len, sharedArrays__)))], dtype=int)
            inds = np.searchsorted(position, chunks)  # assinging contacts to chunks here
            if inds[-1] != len(position):
                raise ValueError  # last chunks boundary must be after all contacts)
            indszip = list(zip(inds[:-1], inds[1:]))  # extents of contact buckets

            indarray = list(range(len(sharedArrays__)))
            random.shuffle(indarray)  # shuffled array of contactmap bucket indices we are going to work with
            for j, (st, end) in enumerate(indszip):
                position[st:end] -= chunks[j]  # pre-subtracting offsets now - not to do it when the lock is being held

            while len(indarray) > 0:  # continue until all contacts are put in buckets
                for i in range(len(indarray)):  # going over all buckets
                    ind = indarray[i]  # select current bucket
                    lock = sharedArrays__[ind].get_lock()  # get lock state
                    if i == len(indarray):  # is this the last bucket?
                        lock.acquire()  # wait for it to be free, and work with it
                    else:
                        if not lock.acquire(0):  # not the last bucket? Try to acquire the lock
                            continue  # if failed, move to the next bucket
                    st, end = indszip[ind]  # succeeded acquiring the lock? Then do things with our bucket
                    sharedNumpy[ind][position[st:end]] += counts[st:end]  # add to the current bucket
                    lock.release()
                    indarray.pop(i)  # remove the index of the bucket because it's finished
                    break  # back to the main loop
            allContacts = []
            if stopped:
                return


def averageContacts(contactIterator, inValues, N, **kwargs):
    """
    A main workhorse for averaging contacts on multiple cores into one shared contact
    map. It mostly does managing the arguments, and initializing the variables. All
    of the logic of how contacts are actually put in shared memory buckets is in the
    worker defined above.

    PARAMETERS
    ----------
        contactIterator : iterator
            an iterator. See descriptions of "filenameContactMap" class below for
            example and explanations
        inValues : iterable
            an array of values to pass to contactIterator. Would be an array of arrays
            of filenames or something like that.
        N : int
            Size of one side of the resulting contactmap

        arrayDtype : ctypes dtype (default c_int32) for the contact map
        classInitArgs : args to pass to the constructor of contact iterator
        classInitKwargs: dict of keyword args to pass to the constructor
        contactProcessing: function f(contacts), should return processed contacts
        nproc : int, number of processors(default 4)
        bucketNum: int (default = nproc) Number of memory buckets to use
        contactBlock: int (default 500k) Number of contacts to aggregate before writing

        useFmap : True, False, or callable
            If True, uses mirnylib.systemutils.fmap
            If False, uses multiprocessing.Pool.map
            Otherwise, uses provided function, assuming it of a fork-map type
            (different initializations are needed for forkmap and
            multiprocessing-style map)

            Sorry, no outside multiprocessing-style maps for now, it's easy to fix
            Let me know if it is needed.


    Code that calcualtes a contactmap from a set of polymer conformation is in the
    methods below (averageMonomerResolutionContactMap, etc.)

    An example code that would run a contactmap from a simulation is below:

    ..code-block:: python

        class simContactMap(object):
            "contactmap 'finder' for a simulation"
            def __init__(self, ind):  # accept a parameter (e.g. random number generator seed)
                self.model = initModel(ind)  # pass parameter to the functon that returns me a model object
                self.count = 10000000   # how many times to run a step of the model
                self.model.steps(10000)   # initial steps of the model to equilibrate it

            def next(self):  # actual realization of the self.next method
                if self.count == 0:   # terminate the simulation if we did self.count iterations
                    raise StopIteration
                self.count -= 1      #decrement the counter
                self.model.steps(30)   # advance model by 30 steps
                return np.array(self.model.getSMCs()).T   # return current LEF positions

        mymap = polychrom.contactmaps.averageContacts(simContactMap, range(20), 30000,  nproc=20 )


    """

    arrayDtype = kwargs.get("arrayDtype", ctypes.c_int32)
    nproc = min(kwargs.get("nproc", 4), len(inValues))
    bucketNum = kwargs.get("bucketNum", nproc)
    if nproc == 1:
        return averageContactsSimple(contactIterator, inValues, N, **kwargs)
    contactBlock = kwargs.get("contactBlock", 5000000)
    useFmap = kwargs.get("useFmap", False)
    classInitArgs = kwargs.get("classInitArgs", [])
    classInitKwargs = kwargs.get("classInitKwargs", {})
    contactProcessing = kwargs.get("contactProcessing", lambda x: x)
    finalSize = N * (N + 1) // 2
    boundaries = np.linspace(0, finalSize, bucketNum + 1, dtype=int)
    chunks = zip(boundaries[:-1], boundaries[1:])
    sharedArrays = [mp.Array(arrayDtype, int(j - i)) for i, j in chunks]
    argset = list(sharedArrays) + [
        contactProcessing,
        classInitArgs,
        classInitKwargs,
        contactIterator,
        contactBlock,
        N,
    ]

    if not useFmap:  # for mp.map we need initializer because shared memory cannot be pickled
        # # or passed as an argument in inValues
        with closing(mp.Pool(processes=nproc, initializer=init, initargs=argset)) as p:
            p.map(worker, inValues)

    # diffent strategy for a local map
    # shared memory is just a global variable created by init()
    else:
        init(*argset)  # creating global variables here
        if callable(useFmap):
            fmap = useFmap
        else:
            from mirnylib.systemutils import fmap
        fmap(worker, inValues, nproc=nproc)

    res = np.concatenate([tonumpyarray(i) for i in sharedArrays])
    del sharedArrays  # save memory
    final = triagToNormal(res, N)
    return final


class filenameContactMap(object):
    """
    This is the iterator for the contact map finder
    """

    def __init__(
        self,
        filenames,
        cutoff=1.7,
        loadFunction=None,
        exceptionsToIgnore=[],
        contactFunction=None,
    ):
        """
        Init accepts arguments to initialize the iterator. filenames will be one of
        the items in the inValues list of the "averageContacts" function cutoff and
        loadFunction should be provided either in classInitArgs or classInitKwargs of
        averageContacts

        When initialized, the iterator should store these args properly and create
        all necessary constructs
        """
        self.filenames = filenames
        self.cutoff = cutoff
        self.exceptionsToIgnore = exceptionsToIgnore
        self.contactFunction = contactFunction
        self.loadFunction = loadFunction
        self.i = 0

    def next(self):
        """
        This is the method which gets called by the worker asking for contacts. This
        method should return new set of contacts each time it is called When there
        are no more contacts to return (all filenames are gone, or simulation is
        over), then this method should raise StopIteration
        """
        if self.i == len(self.filenames):
            raise StopIteration
        try:
            data = self.loadFunction(self.filenames[self.i])
        except tuple(self.exceptionsToIgnore):
            print("contactmap manager could not load file", self.filenames[self.i])
            self.i += 1
            return None
        contacts = self.contactFunction(data, cutoff=self.cutoff)
        self.i += 1
        return contacts


def monomerResolutionContactMap(
    filenames,
    cutoff=5,
    n=8,  # Num threads
    contactFinder=polymer_analyses.calculate_contacts,
    loadFunction=polymerutils.load,
    exceptionsToIgnore=[],
    useFmap=False,
):
    N = findN(filenames, loadFunction, exceptionsToIgnore)
    args = [cutoff, loadFunction, exceptionsToIgnore, contactFinder]
    values = [filenames[i::n] for i in range(n)]
    return averageContacts(
        filenameContactMap,
        values,
        N,
        classInitArgs=args,
        useFmap=useFmap,
        uniqueContacts=True,
        nproc=n,
    )


def binnedContactMap(
    filenames,
    chains=None,
    binSize=5,
    cutoff=5,
    n=8,  # Num threads
    contactFinder=polymer_analyses.calculate_contacts,
    loadFunction=polymerutils.load,
    exceptionsToIgnore=None,
    useFmap=False,
):
    n = min(n, len(filenames))
    N = findN(filenames, loadFunction, exceptionsToIgnore)

    if chains is None:
        chains = [[0, N]]

    bins = []
    chains = np.asarray(chains)
    chainBinNums = np.ceil((chains[:, 1] - chains[:, 0]) / (0.0 + binSize))

    for i in range(len(chainBinNums)):
        bins.append(binSize * (np.arange(int(chainBinNums[i]))) + chains[i, 0])
    bins.append(np.array([chains[-1, 1] + 1]))
    bins = np.concatenate(bins) - 0.5
    Nbase = len(bins) - 1

    if Nbase > 25000:
        warnings.warn(UserWarning("very large contact map" " may be difficult to visualize"))

    chromosomeStarts = np.cumsum(chainBinNums)
    chromosomeStarts = np.hstack((0, chromosomeStarts))

    def contactAction(contacts, myBins=[bins]):
        contacts = np.asarray(contacts, order="C")
        cshape = contacts.shape
        contacts.shape = (-1,)
        contacts = np.searchsorted(myBins[0], contacts) - 1
        contacts.shape = cshape
        return contacts

    args = [cutoff, loadFunction, exceptionsToIgnore, contactFinder]
    values = [filenames[i::n] for i in range(n)]
    mymap = averageContacts(
        filenameContactMap,
        values,
        Nbase,
        classInitArgs=args,
        useFmap=useFmap,
        contactProcessing=contactAction,
        nproc=n,
    )
    return mymap, chromosomeStarts


class filenameContactMapRepeat(object):
    """
    This is a interator for the repeated contact map finder
    """

    def __init__(
        self,
        filenames,
        mapStarts,
        mapN,
        cutoff=1.7,
        loadFunction=None,
        exceptionsToIgnore=[],
        contactFunction=None,
    ):
        """
        Init accepts arguments to initialize the iterator. filenames will be one of
        the items in the inValues list of the "averageContacts" function cutoff and
        loadFunction should be provided either in classInitArgs or classInitKwargs of
        averageContacts

        When initialized, the iterator should store these args properly and create
        all necessary constructs
        """
        self.filenames = filenames
        self.cutoff = cutoff
        self.exceptionsToIgnore = exceptionsToIgnore
        self.mapStarts = mapStarts
        self.mapN = mapN
        self.contactFunction = contactFunction
        self.loadFunction = loadFunction

        self.i = 0
        self.curStarts = []

    def next(self):
        """
        This is the method which gets called by the worker asking for contacts. This
        method should return new set of contacts each time it is called When there
        are no more contacts to return (all filenames are gone, or simulation is
        over), then this method should raise StopIteration
        """

        try:
            if len(self.curStarts) == 0:
                if self.i == len(self.filenames):
                    raise StopIteration
                self.data = self.loadFunction(self.filenames[self.i])
                self.curStarts = list(self.mapStarts)
                self.i += 1
            start = self.curStarts.pop()
            data = self.data[start : start + self.mapN]
            assert len(data) == self.mapN
        except tuple(self.exceptionsToIgnore):
            print("contactmap manager could not load file", self.filenames[self.i])
            self.i += 1
            return None
        contacts = self.contactFunction(data, cutoff=self.cutoff)
        return contacts


def monomerResolutionContactMapSubchains(
    filenames,
    mapStarts,
    mapN,
    cutoff=5,
    n=8,  # Num threads
    method=polymer_analyses.calculate_contacts,
    loadFunction=polymerutils.load,
    exceptionsToIgnore=[],
    useFmap=False,
):
    args = [mapStarts, mapN, cutoff, loadFunction, exceptionsToIgnore, method]
    values = [filenames[i::n] for i in range(n)]
    return averageContacts(
        filenameContactMapRepeat,
        values,
        mapN,
        classInitArgs=args,
        useFmap=useFmap,
        uniqueContacts=True,
        nproc=n,
    )
