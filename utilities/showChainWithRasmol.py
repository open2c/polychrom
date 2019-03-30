#!/usr/bin/env python
import  os, tempfile, sys
import numpy as np
import joblib
import textwrap

if len(sys.argv) < 2:
    print(textwrap.dedent("""
            Usage: show filename [start end pace]
                 show filenum [start end pace] 

                 filenum is a number of files of the type block123.dat

                 start, end, pace will convert data to data[start:end:pace]"""))
       
  

def showData(data):
    if len(sys.argv) == 3:
        start = int(sys.argv[2])
        data = data[start:]
    if len(sys.argv) == 4:
        start = int(sys.argv[2])
        end = int(sys.argv[3])
        data = data[start:end]
    if len(sys.argv) == 5:
        start = int(sys.argv[2])
        end = int(sys.argv[3])
        pace = int(sys.argv[4])
        data = data[start:end:pace]
    


    #if you want to change positions of the spheres along each segment, change these numbers
    #e.g. [0,.1, .2 ...  .9] will draw 10 spheres, and this will look better
    shifts = [0., 0.2, 0.4, 0.6, 0.8]

    #determining the 95 percentile distance between particles,
    dists = np.sqrt(np.sum(np.diff(data, axis=0) ** 2, axis=1))
    meandist = np.percentile(dists, 95)

    #Finding boundaries between chains, appending zero and chain length to it
    breaks = np.nonzero(dists > 2 * meandist)[0]
    breaks = np.r_[0, breaks + 1, len(data)]


    #rescaling the data, so that bonds are of the order of 1. This is because rasmol spheres are of the fixed diameter.
    data /= meandist
    diffs = data[:1] - data[1:]


    #writing the rasmol script. Spacefill controls radius of the sphere.
    rascript = tempfile.NamedTemporaryFile(mode='w')
    rascript.write("""wireframe off
    color temperature
    spacefill 100
    background white
    """)
    rascript.flush()


    #creating the array, linearly chanhing from -225 to 225, to serve as an array of colors
    #(rasmol color space is -250 to 250, but it  still sets blue to the minimum color it found and red to the maximum).
    colors = np.array([int((j * 450.) / (len(data))) - 225 for j in range(len(data))])

    #creating spheres along the trajectory
    #for speedup I just create a Nx4 array, where first three columns are coordinates, and fourth is the color

    def convertData(data, colors):
        "Returns an somethingx4 array for each subchain"
        newData = np.zeros((len(data) * len(shifts) - (len(shifts) - 1) , 4))
        for i in range(len(shifts)):
            #filling in the array like 0,5,10,15; then 1,6,11,16; then 2,7,12,17, etc.
            #this is just very fast
            newData[i:-1:len(shifts), :3] = data[:-1] * shifts[i] + data[1:] * (1 - shifts[i])
            newData[i:-1:len(shifts), 3] = colors[:-1]
        newData[-1, :3] = data[-1]
        newData[-1, 3] = colors[-1]
        return newData

    newDatas = [convertData(data[breaks[i]:breaks[i + 1]],
                              colors[breaks[i]:breaks[i + 1]])
                 for i in range(len(breaks) - 1)]
    newData = np.concatenate(newDatas)

    towrite = tempfile.NamedTemporaryFile(mode='w')
    towrite.write("%d\n\n" % (len(newData)))  #number of atoms and a blank line after is a requirement of rasmol

    for i in newData:
        towrite.write("CA\t%lf\t%lf\t%lf\t%d\n" % tuple(i))
    towrite.flush()
    #For windows you might need to change the place where your rasmol file is
    if os.name == "posix":  #if linux
        os.system("rasmol -xyz %s -script %s" % (towrite.name, rascript.name))
    else:  #if windows
        os.system("C:/RasWin/raswin.exe -xyz %s -script %s" % (towrite.name, rascript.name))
    exit()



def load(filename):
    from openmmlib import polymerutils
    return polymerutils.load(filename)




try:
    showData(load(sys.argv[1]))
    exit()

except IOError:
    try: 
        showData(load("expanded%s.dat" % sys.argv[1]))

    except IOError:
        showData(load("block%s.dat" % sys.argv[1]))




