import numpy as np
import matplotlib.pyplot as plt
import subprocess
import os
import pymol_show
import polymerutils
from polymerutils import load
from pymol_show import interpolateData, show_chain
import cPickle
import textwrap
from mirnylib.systemutils import setExceptionHook, fmap
import shutil


setExceptionHook()
def showOneChain():
    data = load("L_201_num_1.0_final_protocol/block500.dat")
    #data = load("../globules_expanded/crumpled1.dat_expanded")
    balls = cPickle.load(open("L_201_num_1.0_final_protocol/coreParticles"))
    balls = np.array(balls)
    balls = balls[(balls > 47000) * (balls < 53000)] - 47000
    data = data[47000:53000]

    pymol_show.show_chain(data, dataMult=1.1,
                          spherePositions=balls, sphereRadius=.02)


def duplicateData(x):
    x = np.array(x)
    shape = x.shape
    shape = list(shape)
    shape[0] = shape[0] * 2 - 1
    newlist = np.zeros(shape, dtype=float)
    newlist[::2] = x
    newlist[1::2] = 0.5 * x[:-1] + 0.5 * x[1:]
    return newlist



def showAllChain():
    data = load("L_200_num_5.0_final_protocol/block500.dat")
    #data = data[25000:50000]

    pymol_show.show_chain(data, dataMult=1.1)
#showAllChain()
#exit()


def highlightThreeRegions():
    folder = "L_200_num_8.0_final_protocol"
    filenum = 500
    offset = 2

    data = load(folder + "/block%d.dat" % filenum)
    data = data[30000:50000]

    newData = interpolateData(data, targetN=20000)



    print pymol_show.do_coloring(
        newData[0], [(4000, 4040), (4400, 4440), (4800, 4840),
                                     (5200, 5240), (5600, 5640)],
        ["green", "red", "blue", "brown", "orange"],
        [0, 0, 0, 0, 0],
        chainRadius=.15, subchainRadius=.2,
        chainTransparency=0.5,
        multiplier=.8,
        showChain="spheres")

    exit()

#highlightThreeRegions()

def showCoreParticles():
    folder = "L_100_num_12_len_30"
    filenum = 25
    offset = 1

    coreParticles = cPickle.load(open(os.path.join(folder,
                                                   "coreParticles")))
    data = load(folder + "/block%d.dat" % filenum)

    #data = load("../globules_expanded/crumpled1.dat_expanded")
    colorArray = np.zeros(len(data), int)
    for j, i in enumerate(coreParticles):
        if (j < len(coreParticles) - 1) and (j % 20 == 1):
            colorArray[max(i , 0): min(coreParticles[j + 1] - 1 , len(data))] = 2

        colorArray[max(i - offset, 0):min(i + offset + 1, len(data))] = 1


    loopArray = np.zeros(len(data), int)

    coords = data

    regions1 = pymol_show.createRegions(colorArray == 1)
    M = len(regions1)
    print M
    colors1 = ["brown" for i in range(M)]
    transparencies1 = [0 for i in colors1]

    regions2 = pymol_show.createRegions(colorArray == 2)
    M = len(regions2)
    print M
    allColors = ["br{0}".format(i) for i in range(10)]
    S = len(allColors)
    colors2 = [allColors[int(float(i) * float(S) / float(M))] for i in range(M)]
    transparencies2 = [0 for i in colors2]


    return pymol_show.do_coloring(coords, list(regions2),
                           list(colors2),
                           list(transparencies2),
                           chainRadius=.25, subchainRadius=.35,
                           chainTransparency=0.85,
                           returnScriptName="mov",
                           showChain="worm",
                           pdbGroups=colorArray,
                           showGui=True,
                           # saveTo="bla.png",
                           multiplier=.8,
                           support="""
                           create back, chain 1
                           set cartoon_transparency,0.000000,back
                           set cartoon_trace_atoms,1, back
                           set cartoon_tube_radius,0.280000, back
                           cartoon tube, back
                           color brown, back
                           set depth_cue, 0
                           set_view (\
_     0.572576821,   -0.411692470,   -0.708985031,\
_    -0.745704532,    0.097847328,   -0.659046650,\
_     0.340698749,    0.906049132,   -0.250979245,\
_     0.000131026,   -0.000255849, -401.058715820,\
_    49.410499573,   50.416854858,   58.617347717,\
_   315.981750488,  458.882690430,  -20.000000000 )
                           """), colorArray




def _mencoder(imgFolder, fps, aviFilename):
    subprocess.call(
        ("cd {0}; ".format(imgFolder) +
         "mencoder \"mf://*.png\" -mf fps={0} -o {1} ".format(fps, aviFilename) +
          "-mc 0 -noskip -skiplimit 0 -ovc lavc -lavcopts "
          "vcodec=mpeg4:vhq:trell:mbd=2:vmax_b_frames=1:v4mv:vb_strategy=0:"
          "vlelim=0:vcelim=0:cmp=6:subcmp=6:precmp=6:predia=3:dia=3:vme=4:vqscale=1"),
         #"type=png -ovc raw -oac copy"),
         #"-ovc lavc -lavcopts vcodec=mpeg4:mbd=2:trell:vbitrate=200000"),
         #"-sameq"),
        shell=True)


#script, colorArray = showCoreParticles()



def makeMoviePymol(fileList, destFolder, fps=15, aviFilename='output.avi', pymolScript=""):
    if False in [os.path.exists(i) for i in fileList]:
        raise IOError("Some files are not in filelist")
    numFrames = len(fileList)
    numDigits = int(np.ceil(np.log10(numFrames)))

    destFolder = os.path.abspath(destFolder)
    pdbFolder = destFolder + '/pdb'
    imgFolder = destFolder + '/img'
    if os.path.exists(imgFolder):
        shutil.rmtree(imgFolder)
    for folder in [destFolder, pdbFolder, imgFolder]:
        if not os.path.isdir(folder):
            os.mkdir(folder)

    def saveToPdb(input):
        i, dataPath = input
        d = polymerutils.load(dataPath)
        pdbFilename = '{0:0{width}}.pdb'.format(i, width=numDigits)
        savePath = pdbFolder + '/' + pdbFilename
        polymerutils.save(d, savePath, mode='pdb', pdbGroups=colorArray)
        return os.path.abspath(savePath)



    pdbPaths = fmap(saveToPdb, enumerate(fileList))

    script = 'hide all\n'
    for i in pdbPaths:
        script += 'load {0}, mov\n'.format(i)



    script += textwrap.dedent("""
    smooth mov
    """)
    script += pymolScript
    script += "\n"

    script += textwrap.dedent("""
    zoom mov
    """)


    tmpScriptPath = os.path.abspath(destFolder + '/movie.pymol')
    tmpScript = open(tmpScriptPath, 'w')
    tmpScript.write(script)
    tmpScript.flush()
    tmpScript.close()

    os.system("cd {0}; pymol  -u {1}; cd -".format(imgFolder, tmpScriptPath))
    _mencoder(imgFolder, fps, aviFilename)

filelist = ["L_100_num_16_len_30/formovie/block%d.dat" % i for i in range(0, 1213)]
#makeMoviePymol(filelist, "mymovie")


def makeMovie(fileList, imgFolder, fps=20, aviFilename='output.avi'):
    offset = 2

    if not fileList:
        return
    numFrames = len(fileList)
    numDigits = int(np.ceil(np.log10(numFrames)))


    def smallFunction(x):
        i, dataPath = x


        savePath = imgFolder + '/{0:0{width}}.png'.format(i, width=numDigits)
        coreParticles = cPickle.load(open(os.path.join(os.path.split(dataPath)[0],
                                                       "coreParticles")))
        data = load(dataPath)

        #data = load("../globules_expanded/crumpled1.dat_expanded")
        colorArray = np.zeros(len(data), int)
        for j, i in enumerate(coreParticles):
            if (j < len(coreParticles) - 1) and (j % 20 == 1):
                colorArray[max(i , 0): min(coreParticles[j + 1] - 1 , len(data))] = 2

            colorArray[max(i - offset, 0):min(i + offset + 1, len(data))] = 1


        loopArray = np.zeros(len(data), int)

        coords = data

        regions1 = pymol_show.createRegions(colorArray == 1)
        M = len(regions1)
        print M
        colors1 = ["brown" for i in range(M)]
        transparencies1 = [0 for i in colors1]

        regions2 = pymol_show.createRegions(colorArray == 2)
        M = len(regions2)
        print M
        allColors = ["br{0}".format(i) for i in range(10)]
        S = len(allColors)
        colors2 = [allColors[int(float(i) * float(S) / float(M))] for i in range(M)]
        transparencies2 = [0 for i in colors2]

        pymol_show.do_coloring(coords, list(regions2),
                               list(colors2),
                               list(transparencies2),
                               chainRadius=.25, subchainRadius=.35,
                               chainTransparency=0.9,
                               #returnScriptName="mov",
                               showChain="worm",
                               pdbGroups=colorArray,
                               showGui=True,
                               #saveTo=savePath,
                               multiplier=.8,
                               support="""
                               create back, chain 1
                               set cartoon_transparency,0.000000,back
                               set cartoon_trace_atoms,1, back
                               set cartoon_tube_radius,0.280000, back
                               cartoon tube, back
                               color brown, back
                               set depth_cue, 0
                               set field_of_view, 10
                            set_view (\
                                 0.362947434,   -0.752908945,    0.548998356,\
                                -0.843832374,   -0.015661031,    0.536378920,\
                                -0.395252228,   -0.657936990,   -0.641013563,\
                                -0.000659317,    0.000253409, -788.783874512,\
                                81.561027527,   81.701515198,  121.653610229,\
                               615.844299316,  961.749450684,  -10.000000000 )
                                png {savepath}
                                quit


                               """.format(savepath=savePath))

    fmap(smallFunction, enumerate(fileList), n=8)
    _mencoder(imgFolder, fps, aviFilename)

makeMovie(filelist, "mymovie")
