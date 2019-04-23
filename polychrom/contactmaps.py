from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import random
import ctypes
import multiprocessing as mp
from contextlib import closing
from . import polymerutils
import warnings
from . import polymer_analyses

"""
This file defines the class that allows averaging contacts from multile processes into one shared contactmap. 

It was not an easy function to write, and as a result it is a little messy. But it is necessary for a set of reasons. 
-- Calculating contact maps is slow, and benefits greatly from parallelizing
-- Doing regular multiprocesing.map has limitations
    -- It can only handle heataps up to some size, and transferring gigabyte-sized heatmaps between processes takes minutes 
    -- It can only do as many heatmaps as fits in RAM, which on 20-core 128GB machine is no more than 5GB/heatmap 
    -- Historically, we had troubles with heatmaps over 6000x6000 monomers being transferred between processes using pickle 
So to combat that we implemented this module. It can create one big heatmap, and keep in in shared memory. 
Different processes can write to that. 


The structure of this class is as follows. 
On the outer lefe, it  provides three methods to average contactmaps: monomerResolutionContactMap, binnedContactMap, 
and monomerResolutionContactMapSubchains
The first two create contact map from an entire file. The last one creates contact maps from sub-chains in a file, starting at a 
given set of starts. It is useful when doing contact maps from several copies of a system in one simulation.

The first two methods have a legacy implementation from the old library that is still here to do the tests. 

It also provides a more generic method "averageContacts". You can use it to average contacts obtained from 
polymer simulations, or from any other simulation. 

It also will provide an example of how to average contacts from a non-polymer simulation, and run this simulation on many cores.
"""
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


def simpleWorker(x, uniqueContacts):
    """
    A "reference" version of "worker" function below that runs on only one core.
     Unlike the actual multicore worker, it can write contacts to the matrix directly without sorting.
     This is useful when your contact finding is faster than sorting a 1D array fo contacts
     uniqueContacts: bool
        if True, assume that contactFinder outputs only unique contacts (like pure contact map)
        if False, do not assume that (like in binned contact map)

    """
    myIterator = contactIterator__(x, *classInitArgs__, **classInitKwargs__)
    while True:
        try:
            contacts = myIterator.next()
            if contacts is None:
                continue
            contacts = np.asarray(contacts)
            assert len(contacts.shape) == 2
            if contacts.shape[1] != 2:
                raise ValueError("Contacts.shape[1] must be 2. Exiting.")
            contacts = contactProcessing__(contacts)
            ctrue = indexing(contacts[:,0], contacts[:,1], N__)
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
    sharedArrays = [np.zeros(finalSize, dtype = arrayDtype)]   # just an array, not a shared array here bc 1 core
    argset = list(sharedArrays) + [contactProcessing, classInitArgs, classInitKwargs, contactIterator, contactBlock, N]
    init(*argset)
    [simpleWorker(x, uniqueContacts) for x in inValues]   # just calling workers 
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
    random.seed()   # making sure our bucket selectors are really randomized

    sharedNumpy = list(map(tonumpyarray, sharedArrays__))  # shared numpy arrays
    allContacts = []
    contactSum = 0
    myIterator = contactIterator__(x, *classInitArgs__, **classInitKwargs__)  # acquiring and initializing iterator
    stopped = False
    while True:   #main event loop
        try:
            contacts = myIterator.next()  # fetch contacts
            if contacts is None:
                continue
            contacts = np.asarray(contacts)
            assert len(contacts.shape) == 2
            if contacts.shape[1] != 2:
                raise ValueError("Contacts.shape[1] must be 2. Exiting.")
            contactSum += contacts.shape[1]
            allContacts.append(contacts)
        except StopIteration:
            stopped = True

        if (contactSum > contactBlock__) or stopped:  # we aggregated enough contacts.  ready to dump them.
            if len(allContacts) == 0:
                return   # no contacts found at all - exiting (we must be stopped)
            contactSum = 0
            contacts = np.concatenate(allContacts, axis=0)
            contacts = contactProcessing__(contacts)
            if len(contacts) == 0:
                if stopped:  # contactProcessing killed all contacts? are we done?
                    return   #if yes, exiting
                continue  # if not, going to the next bucket
            ctrue = indexing(contacts[:,0], contacts[:,1], N__)  # converting to 1D contacts
            position, counts = np.unique(ctrue, return_counts=True) #unique contacts
            assert position[0] >= 0
            assert position[-1] < N__ * (N__+1) // 2   #boundary check for contacts
            chunks = np.array(np.r_[0, np.cumsum(list(map(len, sharedArrays__)))], dtype=int)
            inds = np.searchsorted(position, chunks)  # assinging contacts to chunks here
            if inds[-1] != len(position):
                raise ValueError   # last chunks boundary must be after all contacts)
            indszip = list(zip(inds[:-1], inds[1:]))  # extents of contact buckets

            indarray = list(range(len(sharedArrays__)))
            random.shuffle(indarray)   # shuffled array of contactmap bucket indices we are going to work with 
            for j,(st,end) in enumerate(indszip):
                position[st:end] -= chunks[j]    # pre-subtracting offsets now - not to do it when the lock is being held

            while len(indarray) > 0:   # continue until all contacts are put in buckets
                for i in range(len(indarray)): #going over all buckets
                    ind = indarray[i]      # select current bucket
                    lock = sharedArrays__[ind].get_lock()  # get lock state
                    if i == len(indarray):   #is this the last bucket? 
                        lock.acquire()       # wait for it to be free, and work with it
                    else:
                        if not lock.acquire(0):  # not the last bucket? Try to acquire the lock
                            continue             # if failed, move to the next bucket
                    st, end = indszip[ind]     #succeeded acquiring the lock? Then do things with our bucket
                    sharedNumpy[ind][position[st:end]] += counts[st:end]  # add to the current bucket
                    lock.release()
                    indarray.pop(i)  # remove the index of the bucket because it's finished
                    break            #back to the main loop 
            allContacts = []
            if stopped:
                return


def averageContacts(contactIterator, inValues, N, **kwargs):
    """
    A main workhorse for averaging contacts on multiple cores into one shared contact map. 
    It mostly does managing the arguments, and initializing the variables. All of the logic of 
    how contacts are actually put in shared memory buckets is in the worker defined above. 
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
            classInitKwargs: dict of keyword args to pass to the constructor
            contactProcessing: function f(contacts), should return processed contacts
            nproc : int, number of processors(default 4)
            bucketNum: int (default = nproc) Number of memory bukcets to use
            contactBlock: int (default 500k) Number of contacts to aggregate before writing to memory
            useFmap: True, or callable, fork-map (or regular map) function; not multiprocessing.map! 
                     defaults to mirnylib.systemutils.fmap if true. Otherwise, uses provided function
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
    boundaries = np.linspace(0, finalSize, bucketNum + 1, dtype = int)
    chunks = zip(boundaries[:-1], boundaries[1:])
    sharedArrays = [mp.Array(arrayDtype, int(j - i)) for i, j in chunks]
    argset = list(sharedArrays) + [contactProcessing, classInitArgs, classInitKwargs, contactIterator, contactBlock, N]

    if not useFmap:   # for mp.map we need initializer because shared memory cannot be pickled or passed as an argument in inValues
        with closing(mp.Pool(processes=nproc, initializer=init, initargs=argset)) as p:
            p.map(worker, inValues)
    else:   #diffent strategy for a local map - shared memory is just a global variable created by init() 
        init(*argset)  # creating global variables here
        if callable(useFmap):
            fmap = useFmap
        else: 
            from mirnylib.systemutils import fmap
        fmap(worker, inValues, nproc=nproc)
        
    res = np.concatenate([tonumpyarray(i) for i in sharedArrays])
    del sharedArrays   # save memory 
    final = triagToNormal(res, N)    
    return final


class filenameContactMap(object):
    """
    This is the iterator for the contact map finder
    """
    def __init__(self, filenames, cutoff = 1.7, loadFunction=None, exceptionsToIgnore=[], 
                contactFunction=None):
        """
        Init accepts arguments to initialize the iterator.
        filenames will be one of the items in the inValues list of the "averageContacts" function
        cutoff and loadFunction should be provided either in classInitArgs or classInitKwargs of averageContacts

        When initialized, the iterator should store these args properly and create all necessary constructs
        """
        from openmmlib import contactmaps
        self.contactmaps  = contactmaps
        self.filenames = filenames
        self.cutoff = cutoff
        self.exceptionsToIgnore = exceptionsToIgnore
        self.contactFunction = contactFunction
        self.loadFunction = loadFunction
        self.i = 0

    def next(self):
        """
        This is the method which gets called by the worker asking for contacts.
         This method should return new set of contacts each time it is called
         When there are no more contacts to return (all filenames are gone, or simulation is over),
         then this method should raise StopIteration
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

def monomerResolutionContactMap(filenames,
                          cutoff=1.7,
                          n=4,  # Num threads
                          contactFinder = polymer_analyses.calculate_contacts,
                          loadFunction=polymerutils.load,
                          exceptionsToIgnore=[]):

    N =  findN(filenames, loadFunction, exceptionsToIgnore)        
    args = [cutoff, loadFunction, exceptionsToIgnore, contactFinder]
    values = [filenames[i::n] for i in range(n)]
    return averageContacts(filenameContactMap,values,N, classInitArgs=args, useFmap=True, uniqueContacts = True, nproc=n)


def binnedContactMap(filenames, chains=None, binSize=5, cutoff=1.7,
                            n=4,  # Num threads
                            contactFinder = polymer_analyses.calculate_contacts,
                            loadFunction=polymerutils.load,
                            exceptionsToIgnore=None):
    n = min(n, len(filenames))
    subvalues = [filenames[i::n] for i in range(n)]

    datas = []
    N =  findN(filenames, loadFunction, exceptionsToIgnore)
    
    if chains is None:
        chains = [[0, N]]
        
    bins = []
    chains = np.asarray(chains)
    chainBinNums = (np.ceil((chains[:, 1] - chains[:, 0]) / (0.0 + binSize)))
    
    for i in range(len(chainBinNums)):
        bins.append(binSize * (np.arange(int(chainBinNums[i])))
                    + chains[i, 0])
    bins.append(np.array([chains[-1, 1] + 1]))
    bins = np.concatenate(bins) - 0.5
    Nbase = len(bins) - 1

    if Nbase > 25000:
        warnings.warn(UserWarning('very large contact map'
                                  ' may be difficult to visualize'))

    chromosomeStarts = np.cumsum(chainBinNums)
    chromosomeStarts = np.hstack((0, chromosomeStarts))

    def contactAction(contacts, myBins = [bins]):
        contacts = np.asarray(contacts, order = "C")
        cshape = contacts.shape
        contacts.shape = (-1,)
        contacts = np.searchsorted(myBins[0], contacts) - 1
        contacts.shape = cshape
        return contacts

    args = [cutoff, loadFunction, exceptionsToIgnore, contactFinder]
    values = [filenames[i::n] for i in range(n)]
    mymap =  averageContacts(filenameContactMap,values,Nbase, classInitArgs=args, useFmap=True, contactProcessing=contactAction, nproc=n)
    return mymap, chromosomeStarts


class filenameContactMapRepeat(object):
    """
    This is a interator for the repeated contact map finder
    """
    def __init__(self, filenames, mapStarts, mapN, cutoff = 1.7, loadFunction=None, exceptionsToIgnore=[], 
                contactFunction=None,):
        """
        Init accepts arguments to initialize the iterator.
        filenames will be one of the items in the inValues list of the "averageContacts" function
        cutoff and loadFunction should be provided either in classInitArgs or classInitKwargs of averageContacts

        When initialized, the iterator should store these args properly and create all necessary constructs
        """
        from openmmlib import contactmaps
        self.contactmaps  = contactmaps
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
        This is the method which gets called by the worker asking for contacts.
         This method should return new set of contacts each time it is called
         When there are no more contacts to return (all filenames are gone, or simulation is over),
         then this method should raise StopIteration
        """
        if self.i == len(self.filenames):
            raise StopIteration
            
        try:
            if len(self.curStarts) == 0:
                self.data = self.loadFunction(self.filenames[self.i])
                self.curStarts = list(self.mapStarts)
            start = self.curStarts.pop()
            data = self.data[start:start+self.mapN]
            assert len(data) == self.mapN
        except tuple(self.exceptionsToIgnore): 
            print("contactmap manager could not load file", self.filenames[self.i])
            self.i += 1
            return None
        contacts = self.contactFunction(data, cutoff=self.cutoff)
        self.i += 1
        return contacts

def monomerResolutionContactMapSubchains(filenames,
                          mapStarts, 
                          mapN,
                          cutoff=1.7,
                          n=4,  # Num threads
                          method = polymer_analyses.calculate_contacts,                          
                          loadFunction=polymerutils.load,
                          exceptionsToIgnore=[]):

    args = [ mapStarts, mapN,cutoff, loadFunction, exceptionsToIgnore, method]
    values = [filenames[i::n] for i in range(n)]
    return averageContacts(filenameContactMapRepeat,values,mapN, classInitArgs=args, useFmap=True, uniqueContacts = True, nproc=n)






class dummyContactMap(object):
    "contactmap 'finder' for testing that returns fixed contacts (a) + constant(x)"
    def __init__(self, x, a):
        self.a = a + x
        self.M = 15
    def next(self):
        if self.M == 0:
            raise StopIteration
        self.M -= 1
        return self.a

  
def _test():
    ars = [np.random.random((60,3)) * 4 for _ in range(200)]
    import openmmlib.contactmaps
    conts = polymer_analyses.calculate_contacts(ars[0],1)
    print(conts.shape)
    args = np.repeat(np.arange(20, dtype=int), 10)
    cmap1 = averageContacts(dummyContactMap, args, 100, classInitArgs=[conts], nproc=20)
    cmap4 = averageContacts(dummyContactMap, args, 100, classInitArgs=[conts], nproc=1)
    cmap2 = averageContactsSimple(dummyContactMap, args, 100, classInitArgs=[conts])
    cmap3 = np.zeros((100,100))
    for i in args:
        cmap3[conts[:,0] + i , conts[:,1] + i] += 15
    cmap3 = cmap3 + cmap3.T

    print(cmap1.sum())
    print(cmap2.sum())
    print(cmap3.sum())
    print(cmap4.sum())
    assert np.allclose(cmap1, cmap2)
    assert np.allclose(cmap1, cmap4)
    assert np.allclose(cmap1, cmap3)

    from .legacy_contactmaps import averagePureContactMap as cmapPureMap
    from .legacy_contactmaps import averageBinnedContactMap as cmapBinnedMap

    for n in [1, 5, 20]:
        cmap6 = monomerResolutionContactMap(range(200), cutoff = 1, loadFunction=lambda x:ars[x], n=n)
        cmap5 = cmapPureMap(range(200), cutoff = 1, loadFunction=lambda x:ars[x], n=n, printProbability=0.001)
        print(cmap5.sum(), cmap6.sum())
        assert np.allclose(cmap6, cmap5)

        cmap7 = binnedContactMap(range(200), chains= [(0,27),(27,60)], binSize = 2, cutoff = 1, loadFunction=lambda x:ars[x], n=n)[0]

        cmap8 = cmapBinnedMap(range(200), chains= [(0,27),(27,60)], binSize = 2, cutoff = 1, loadFunction=lambda x:ars[x], n=n, printProbability=0.001)[0]
        print(cmap7.sum(), cmap8.sum())
        assert np.allclose(cmap7, cmap8)
       
    
    print("All tests passed")










