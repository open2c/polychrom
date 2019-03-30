import warnings
import numpy as np
from openmmlib import Simulation
import os 
import joblib
import matplotlib
import matplotlib.pyplot as plt
import sys
import polymerutils


#----------------- all simulation parameters are defined below------
CHAIN_LENGTH = 3000   #chain length, monomers
STIFFNESS = 3  #chain stiffness; 1-2 = fiexible, 3 = medium, 4-8 = stiff
DENSITY = 0.02  #physiological density is 0.01 to 0.3 for mammals, some estimates give as low as 0.001 for Drosophila (see Grosberg, 2014 review on chromosome territoreies) 
PLATFORM = "cuda"  #change to OpenCL on machines which don't have cuda; you may use CPU for simulations on machines without GPU; see new release of OpenMM (6.0) 
SAVE_FOLDER = "Simulation"
PHANTOM_CHAIN = False  #change to True to simulate phantom chain. Phantom chain was used only for one control simulation in the paper. 
ENERGY_TRUNCATION = 2  #penalty for chain overlap. Setting it below 1 would make simulations unphysical, very close to phantom chain.
NUM_OF_CONFORMATIONS = 80000 # 80000 conformations were used for each simulation in paper
#---------------------------------------------------------------------


#----------------- loop(s) are defined below -----------------------------
# Change the code below to make different arrangements of loops. 
#To get a nicely looking heatmap, try to keep loops in the center of the chain
LOOPS = [(1485,1515)]  #adding a single 30-monomer loop between monomers 1485 and 1515. 

"""
LOOPS = [(1470,1500),(1500,1530)]  #adding a two loops in the center of the system
LOOPS = [(1455,1485),(1515,1545)]  #adding a two loops with a gap between them - hypothetical example, not considered in the paper
"""
#-----------------------------------------------------------------------------

#self-check that loop anchors are far enough from chain boundary
loopAnchors = sum(map(list,LOOPS),[])
if (min(loopAnchors) < 0.2 * CHAIN_LENGTH) or (max(loopAnchors) > 0.8 * CHAIN_LENGTH):
	print """
	-------------------ERROR-------------------
	Loop(s) are too close to the end of the chain, 
	and will not display nicely on the heatmap.
	Please place loop(s) at least {0} monomers away from the chain length ({1})	
	-------------------------------------------
	""".format(int(0.2*CHAIN_LENGTH), CHAIN_LENGTH)
	raise RuntimeError("Loops to close to chain boundary") 

if STIFFNESS > 8:
    raise ValueError("Stiffness is too high. If you really want to explore this regime, remove this error")
if ENERGY_TRUNCATION > 6:
    raise ValueError("Energy truncation is too high; this will make chain non pass itself, and slow down exploration of the conformation space")
if NUM_OF_CONFORMATIONS < 5000:
    warnings.warn("Too few conformations; may not be enough to make a reasonable heatmap") 
if CHAIN_LENGTH < 100:
    raise ValueError("Chain length too short")
if CHAIN_LENGTH < 1000:
    warnings.warn("Note that you won't get a significant speedup from considering chains shorter than 1000") 
if DENSITY > 0.85:
    raise ValueError("Density should be less than 0.85; otherwise the dynamics will be too slow") 
if PLATFORM.lower() not in ["cuda","opencl", "cpu"]:
    raise ValueError('Wrong platform. Please select "cuda", "opencl" or "cpu"')
   


a = Simulation(timestep = 100, thermostat = 0.002)  # initializing simulations 
a.setup(platform = PLATFORM, verbose = True)  # anoter initialization step 
a.saveFolder(SAVE_FOLDER)  #Defining folder to save conformations 
a.load(polymerutils.grow_rw(CHAIN_LENGTH, 80))  # creating initial conformation and loading it
a.setLayout(mode = "ring")  # making the chain into a ring 
a.addSphericalConfinement(density = DENSITY)  # adding spherical confinement
a.addHarmonicPolymerBonds()  # add polymer 

if not PHANTOM_CHAIN:
	a.addGrosbergRepulsiveForce()
else:
	a.addGrosbergRepulsiveForce(ENERGY_TRUNCATION)
a.addGrosbergStiffness(k = STIFFNESS)
print 
print "-----Adding loops as defined by LOOPS array---------"

for loop in LOOPS:
	a.addBond(i=loop[0], j=loop[1], bondWiggleDistance=.05, distance=1, bondType='Harmonic', verbose=True)	

print "-----finished adding loops-------" 
print 
a.localEnergyMinimization()
print "Running for 1000 blocks to achieve equilibration; data are not saved"
for _ in xrange(1000):
	a.doBlock(3000, increment=False)
print "Data collection begins" 
for _ in xrange(NUM_OF_CONFORMATIONS):
	a.doBlock(3000) # runs 3000 steps per block    
	a.save()

a.printStats()
