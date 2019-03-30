import numpy as np
numpy = np
from openmmlib.pymol_show import createRegions
from mirnylib.numutils import continuousRegions
from scipy import sparse
from scipy.sparse import csgraph as graph
import matplotlib.pyplot as plt
import pickle
from mirnylib.systemutils import setExceptionHook

class makeBondsForBrush(object):
    def __init__(self, chainLength=10000,
                diameterNm=400,
                lengthNm=1000,
                monNm=8,
                monBp=100,
                genomeLength=4000000, mode="ring"):

        self.N = chainLength
        self.diameterNm = diameterNm
        self.lengthNm = lengthNm
        self.monNm = monNm
        self.monBp = monBp
        self.genomeLength = genomeLength
        self.segments = []
        self.bristleCodes = [1, 2]
        monsIdeal = genomeLength / monBp
        monsReal = self.N
        scalingFactor = (monsIdeal / monsReal) ** (1 / 3.)

        radiusMon = diameterNm / (2 * monNm * scalingFactor)
        halfLengthMon = lengthNm / (2 * monNm * scalingFactor)

        self.radiusMon = radiusMon
        self.halfLengthMon = halfLengthMon
        self.bonds = []
        self.bondStatus = np.zeros(self.N, np.int)
        self.mode = mode.lower()

    def save(self, filename):
        struct = {"N":self.N,
                  "diamNm": self.diameterNm,
                  "lengthNm":self.lengthNm,
                  "monNm":self.monNm,
                  "monBp":self.monBp,
                  "genomeLength":self.genomeLength,
                  "segments": self.segments,
                  "bondStatus":self.bondStatus,
                  "bonds": self.bonds}
        pickle.dump(struct, open(filename, "wb"))
    def load(self, filename):
        s = pickle.load(open(filename,'rb'))
        self.__init__(s["N"], s["diamNm"], s["lengthNm"], s["monNm"], s.get("monBp", 100), s["genomeLength"])
        self.segments = s["segments"]
        self.bondStatus = s["bondStatus"]
        self.bonds = s["bonds"]



    def sortSegments(self):
        self.segments = [i for i in self.segments if i[1] != i[0]]
        self.segments = sorted(self.segments, key=lambda x:x[0])
    def checkConnectivity(self, showComponents=True, throwException=True):
        bonds = np.array(self.bonds).T
        matrix = sparse.csc_matrix((np.ones(len(bonds[0])), bonds), (self.N, self.N))
        tmatrix = matrix + matrix.transpose()
        numComp, comp = graph.connected_components(tmatrix)
        if numComp > 1:
            if showComponents:
                plt.plot(comp)
                plt.ylabel("component number")
                plt.xlabel("monomer number")
                plt.title("Your bonds are not connected!")
                plt.show()
            if throwException:
                raise RuntimeError("Bonds are not connected")
            return False
        return True

    def addGap(self, beg, end):
        "Adds a PFR"
        beg, end = sorted((beg, end))
        if beg < 0:
            raise ValueError("Start has to be more than zero!")
        if end >= self.N:
            raise ValueError("End should be less than chain length")
        self.bondStatus[beg:end] = 1
        self.segments.append((beg, end, 0))

    def addClearGap(self, beg, end):
        beg, end = sorted((beg, end))
        if beg < 0:
            raise ValueError("Start has to be more than zero!")
        if end >= self.N:
            raise ValueError("End should be less than chain length")
        self.bondStatus[beg:end] = 1



    def createBonds(self):
        """
        Actually create a set of bonds to realize a given configuration
        """
        values, starts, end = continuousRegions(self.bondStatus)
        for i in xrange(len(values)):
            val, start, end = values[i], starts[i], ends[i]
            pval, pstart, pend = values[i - 1], starts[i - 1], ends[i - 1]
            for k in xrange(start, end):
                self.bonds.append((k, k + 1))

    def addBristlesInRegion(self, start, end, minimumBristleSize, averageBristleSize,
                            bristleGapMin=1, bristleGapMax=2, code=1, bristleFunction=np.random.exponential):
        "Add bristles with a given distribution of bristle sizes, in a given region"
        cur = start
        segments = []
        while True:
            while True:
                bristleLen = int(bristleFunction(averageBristleSize))
                if bristleLen >= minimumBristleSize:
                    break
            bristleEnd = cur + bristleLen
            if bristleEnd > end:
                bristleEnd = end
                segments.append((cur, bristleEnd, code))
                break
            segments.append((cur, bristleEnd, code))
            if bristleGapMax > 0:
                if bristleGapMax > bristleGapMin:
                    bristleGapLen = np.random.randint(bristleGapMin, bristleGapMax)
                else:
                    bristleGapLen = bristleGapMax
                if bristleGapLen == 0:
                    cur = bristleEnd
                    continue
                bristleGapEnd = bristleEnd + bristleGapLen
                if bristleGapEnd > end:
                    bristleGapEnd = end
                    segments.append((bristleEnd, bristleGapEnd, 0))
                    break
                else:
                    segments.append((bristleEnd, bristleGapEnd, 0))
                cur = bristleGapEnd
            else:
                cur = bristleEnd
        self.segments = self.segments + segments


    def addBristles(self, minimumBristleSize, averageBristleSize,
                    bristleGapMin=1, bristleGapMax=2,
                    bristleFunction=np.random.exponential):
        "Add bristles in all regions which are not PFRs"
        values, starts, ends = continuousRegions(self.bondStatus)

        for value, start, end in zip(values, starts, ends):
            if value == 0:
                self.addBristlesInRegion(start, end,
                                         minimumBristleSize=minimumBristleSize,
                                         averageBristleSize=averageBristleSize,
                                         bristleGapMin=bristleGapMin,
                                         bristleGapMax=bristleGapMax,
                                          bristleFunction=bristleFunction)



    def createBonds(self):
        self.sortSegments()
        segments = self.segments

        all = segments

        for segment in segments:
            if segment[2] in self.bristleCodes:
                self.bonds.append((segment[0], segment[1] if segment[1] < self.N else 0))
            else:
                self.bonds.append((segment[1] - 1, segment[1] if segment[1] < self.N else 0))
            for j in range(segment[0], segment[1] - 1):
                self.bonds.append((j, j + 1))
        if self.mode.lower() == "ring":
            self.bonds.append((0, segments[-1][0]))
        elif self.mode.lower() == "chain":
            self.bonds = [i for i in self.bonds if abs(i[1] - i[0]) < self.N / 2]
            pass
        else:
            raise Exception("Please set mode to ring or chain")

    def isRing(self):
        bonds = numpy.array(self.bonds)
        maxN = bonds.max()
        lens = np.abs(bonds[:, 0] - bonds[:, 1])
        if lens.max() > maxN / 2:
            return True
        return False
    def getChains(self):
        self.sortSegments()
        return [(i[0], i[1], False) for i in self.segments]

    def convertChain(self, data):
        "Converts a set of particle coordinates to a continuous chain"
        self.checkConnectivity()
        self.sortSegments()
        segments = self.segments
        ret = []
        for segment in segments:

            if segment[2] in [0, 1]:
                ret.append(data[segment[0]:segment[1]])
            if segment[2] == 2:
                ret.append(data[segment[0]:segment[0] + 1])
            if segment[2] == 1:
                ret.append(data[segment[1] - 2:(segment[0] - 1) if segment[0] > 0 else None:-1])
        return np.concatenate(ret, axis=0)


