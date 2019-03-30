import numpy as np
import matplotlib.pyplot as plt
import os
import pymol_show

from polymerutils import load
from pymol_show import interpolateData
import cPickle
from mirnylib.systemutils import setExceptionHook


def showOneChain():

    #data = load("../globules_expanded/crumpled1.dat_expanded")
    balls = cPickle.load(open("L_201_num_1.0_final_protocol/coreParticles"))
    balls = np.array(balls)
    balls = balls[(balls > 47000) * (balls < 53000)] - 47000
    data = data[47000:53000]

    pymol_show.show_chain(data, dataMult=1.1,
                          spherePositions=balls, sphereRadius=.02)


def showAllChain():
    data = load("L_100_num_1_len_30/block500.dat")
    #data = data[25000:50000]

    pymol_show.show_chain(data, dataMult=1.1)
#showAllChain()
#exit()


def highlightThreeRegions():
    folder = "L_200_num_8.0_final_protocol"
    filenum = 500
    offset = 3

    data = load(folder + "/block%d.dat" % filenum)
    #data = data[30000:80000]

    newData = interpolateData(data, targetN=90000)

    pymol_show.do_coloring(
        newData[0], [(40000, 40400), (44000, 44400), (48000, 48400),
                                     (52000, 52400), (56000, 56400)],
        ["green", "red", "blue", "brown", "orange"],
        [0, 0, 0, 0, 0],
        chain_radius=.07, subchain_radius=.2,
        chain_transparency=0.5,
        multiplier=.8)
    exit()

#highlightThreeRegions()
#exit()


def showCoreParticles():
    folder = "L_100_num_1_len_30"
    #"L_100_num_11_final_protocol"
    filenum = 400
    offset = 40

    coreParticles = cPickle.load(open(os.path.join(folder,
                                                   "coreParticles")))
    data = load(folder + "/block%d.dat" % filenum)


    #data = load("../globules_expanded/crumpled1.dat_expanded")
    colorArray = np.zeros(len(data), int)
    loopArray = np.zeros(len(data), int)
    for j, i in enumerate(coreParticles):
        if (j < len(coreParticles) - 1) and (j % 25 == 2):
            colorArray[max(i + offset + 1, 0): min(coreParticles[j + 1] - offset - 1 , len(data))] = 2

        colorArray[max(i - offset, 0):min(i + offset + 1, len(data))] = 1




    newData = interpolateData(data, targetN=90000, colorArrays=[colorArray])

    coords = newData[0]
    colorArray = newData[1][0]
    colorArray = np.ceil(colorArray)


    """
    greenColors = np.zeros_like(colors)
    greenColors[40000:65000] = 1
    greenColors[colors] = 0
    greenRegions = pymol_show.create_regions(greenColors)
    greenRegions = list(greenRegions)
    greenColors = ["green"] * len(greenRegions)
    greenTransparencies = [0] * len(greenColors)
    """
    regions = pymol_show.createRegions(colorArray == 1)
    regions = list(regions)
    colors = ["blue"] * len(regions)
    transparencies = [0] * len(regions)

    regions2 = pymol_show.createRegions(colorArray == 2)
    M = len(regions2)
    print M
    allColors = ["br{0}".format(i) for i in range(10)]
    S = len(allColors)
    colors2 = [allColors[int(float(i) * float(S) / float(M))] for i in range(M)]
    transparencies2 = [0 for i in colors2]


    pymol_show.do_coloring(coords, list(regions) + list(regions2),
                           colors + colors2,
                           transparencies + transparencies2,
                           showChain="worm",
                           chainRadius=.4, subchainRadius=.4,
                           chainTransparency=0.6,
                           multiplier=.8,
                           pdbGroups=colorArray,
                           support="""
                           bg black
                           set orthoscopic, 0
                           set field_of_view, 70
                           create back, chain 1
                           color brown, back
                           show surface, back
                           """)
    exit()

    pymol_show.do_coloring(coords, regions, colors, transparencies,
                           chain_radius=.1, subchain_radius=.2,
                           chain_transparency=0.2,
                           multiplier=.8)



def makeCoverFigure():
    folder = "L_100_num_16_len_30"
    filenum = 550


    #folder = "../026_controls_loopSize/ConsScaff133_3"
    #filenum = 250

    offset = 1

    coreParticles = cPickle.load(open(os.path.join(folder,
                                                   "coreParticles")))
    print len(coreParticles)
    data = load(folder + "/block%d.dat" % filenum)

    coreArray = np.zeros(len(data), dtype=float)
    coreArray[coreParticles] = 2

    #data = data[::-1]
    #coreParticles = len(data) - coreParticles
    #data = data[:30000]
    #coreParticles = coreParticles[coreParticles < 30000 ]

    if len(data) > 90000:
        data, (coreArray,) = interpolateData(data, 90000, colorArrays=[coreArray])
        coreArray[np.nonzero(coreArray > 0.4)[0] + 1] = 0
        coreParticles = np.nonzero(coreArray > .4)[0]

    data = np.dot(data, np.array([[0.782993197, -0.090055801, 0.615476012, ],
     [0.042767350, 0.994917035, 0.091168165, ],
    [-0.620556772, -0.045062274, 0.782866657, ]]))
    #data = np.dot(data, np.array([[0.829197228, -0.461252242, -0.315715969, ],
    # [0.031428415, 0.602414608, -0.797561765, ],
    # [0.558069587, 0.651413143, 0.514018297, ]]
     #))




    #data = load("../globules_expanded/crumpled1.dat_expanded")
    colorArray = np.zeros(len(data), int)
    loopArray = np.zeros(len(data), int)
    for S, offset in enumerate(range(5, 0, -2)):
        for j, i in enumerate(coreParticles):
            colorArray[max(i - offset, 0):min(i + offset + 1, len(data))] = S + 1

    inds = np.diff(colorArray) != 0
    print inds
    indCumsum = np.r_[0, np.cumsum(inds)]
    newdata = np.zeros(shape=(len(data) + indCumsum[-1], 3), dtype=float)
    newcolorarray = np.zeros(shape=(len(data) + indCumsum[-1],), dtype=float)

    newdata[np.arange(len(data), dtype=int) + indCumsum] = data
    newcolorarray[np.arange(len(data), dtype=int) + indCumsum] = colorArray

    mask = newdata[:, 0] == 0
    newdata[mask] = data[np.nonzero(inds)[0] + 1]
    newcolorarray[mask] = colorArray[np.nonzero(inds)[0] ]



    coords, colorArray = newdata, newcolorarray


    colorStrings = []

    for colorInd in range(0, int(max(colorArray) + 1))[::-1]:
        colorFloat = colorInd / (float(max(colorArray) + 1))
        colorString = "[{0}, {0}, {1}]".format(max(1. - colorFloat * 1.2, 0)  , 1.)
        colorName = "bubu{0}".format(colorInd)
        chainName = "subset{0}".format(colorInd)
        #transp = [0.55, 0.55, 0.55, 0.55, 0., 0., 0., 0.][colorInd]
        transp = [0.75, 0.75, 0.75, 0., 0., 0., 0.][colorInd]

        colorStrings.append(
        """
       set_color {color}, {colorString}
        create {chain}, chain {chainNum}
       color {color}, {chain}
       set cartoon_trace_atoms,1, {chain}
       set cartoon_tube_radius, 0.3, {chain}
       set cartoon_transparency, {transp}, {chain}
       set cartoon_tube_quality, 6

       cartoon tube, {chain}
       show cartoon, {chain}


       """.format(chain=chainName, color=colorName,
                  colorString=colorString, chainNum=colorInd,
                  transp=transp)
                  )



    colorStrings = "".join(colorStrings)



    pymol_show.new_coloring(coords, [],
                           [],
                           [],
                           showChain="none",
                           chainRadius=.4, subchainRadius=.4,
                           chainTransparency=1.0,
                           pdbGroups=colorArray,
                           transparentBackground=True,
                           multiplier=.8,
                           support="""
                           bg white
                           """
                           + colorStrings +

                           """
                            run axes.py
                            axes
                            set spec_direct, 0.1
                            set spec_reflect, 2.
                            set spec_power, 100
                            set spec_direct_power, 100
                            set ray_max_passes, 2000
                            set ray_shadow, off
                            zoom
                            set antialias, 1
                            set light, [0,-1,-0.3]


                           """,
                           presupport="""

                            """)
    exit()



makeCoverFigure()



showCoreParticles()
