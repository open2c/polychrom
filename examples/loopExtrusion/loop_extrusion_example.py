import pickle
import os
import time
import numpy as np
import polychrom

from polychrom import polymerutils
from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.starting_conformations import grow_cubic
from polychrom.hdf5_format import HDF5Reporter, list_URIs, load_URI, load_hdf5_file

import simtk.openmm 
import os 
import shutil


import pyximport; pyximport.install()
from smcTranslocator import smcTranslocatorDirectional

import warnings
import h5py 
import glob




# -------defining parameters----------
#  -- basic loop extrusion parameters
SEPARATION = 200
LIFETIME = 300
N = 10000   # number of monomers
smcStepsPerBlock = 1  # now doing 1 SMC step per block 
steps = 250   # steps per block (now extrusion advances by one step per block)
stiff = 2
dens = 0.2
box = (N / dens) ** 0.33  # density = 0.1.
data = grow_cubic(N, int(box) - 2)  # creates a compact conformation 
block = 0  # starting block 

#folder 
folder = "trajectory"

# new parameters because some things changed 
saveEveryBlocks = 10   # save every 10 blocks (saving every block is now too much almost)
skipSavedBlocksBeginning = 20  # how many blocks (saved) to skip after you restart LEF positions
totalSavedBlocks = 25000  # how many blocks to save (number of blocks done is totalSavedBlocks * saveEveryBlocks)
restartMilkerEveryBlocks = 100 

# parameters for smc bonds 
smcBondWiggleDist = 0.2
smcBondDist = 0.5


# assertions for easy managing code below 
assert restartMilkerEveryBlocks % saveEveryBlocks == 0 
assert (skipSavedBlocksBeginning * saveEveryBlocks) % restartMilkerEveryBlocks == 0 
assert (totalSavedBlocks * saveEveryBlocks) % restartMilkerEveryBlocks == 0 

savesPerMilker = restartMilkerEveryBlocks // saveEveryBlocks
milkerInitsSkip = saveEveryBlocks * skipSavedBlocksBeginning  // restartMilkerEveryBlocks
milkerInitsTotal  = (totalSavedBlocks + skipSavedBlocksBeginning) * saveEveryBlocks // restartMilkerEveryBlocks
print("Milker will be initialized {0} times, first {1} will be skipped".format(milkerInitsTotal, milkerInitsSkip))


class smcTranslocatorMilker(object):

    def __init__(self, smcTransObject):
        """
        :param smcTransObject: smc translocator object to work with
        """
        self.smcObject = smcTransObject
        self.allBonds = []

    def setParams(self, activeParamDict, inactiveParamDict):
        """
        A method to set parameters for bonds.
        It is a separate method because you may want to have a Simulation object already existing

        :param activeParamDict: a dict (argument:value) of addBond arguments for active bonds
        :param inactiveParamDict:  a dict (argument:value) of addBond arguments for inactive bonds

        """
        self.activeParamDict = activeParamDict
        self.inactiveParamDict = inactiveParamDict


    def setup(self, bondForce,  blocks = 100, smcStepsPerBlock = 1):
        """
        A method that milks smcTranslocator object
        and creates a set of unique bonds, etc.

        :param bondForce: a bondforce object (new after simulation restart!)
        :param blocks: number of blocks to precalculate
        :param smcStepsPerBlock: number of smcTranslocator steps per block
        :return:
        """


        if len(self.allBonds) != 0:
            raise ValueError("Not all bonds were used; {0} sets left".format(len(self.allBonds)))

        self.bondForce = bondForce

        #precalculating all bonds
        allBonds = []
        for dummy in range(blocks):
            self.smcObject.steps(smcStepsPerBlock)
            left, right = self.smcObject.getSMCs()
            bonds = [(int(i), int(j)) for i,j in zip(left, right)]
            allBonds.append(bonds)

        self.allBonds = allBonds
        self.uniqueBonds = list(set(sum(allBonds, [])))

        #adding forces and getting bond indices
        self.bondInds = []
        self.curBonds = allBonds.pop(0)

        for bond in self.uniqueBonds:
            paramset = self.activeParamDict if (bond in self.curBonds) else self.inactiveParamDict
            ind = bondForce.addBond(bond[0], bond[1], **paramset) # changed from addBond
            self.bondInds.append(ind)
        self.bondToInd = {i:j for i,j in zip(self.uniqueBonds, self.bondInds)}
        return self.curBonds,[]


    def step(self, context, verbose=False):
        """
        Update the bonds to the next step.
        It sets bonds for you automatically!
        :param context:  context
        :return: (current bonds, previous step bonds); just for reference
        """
        if len(self.allBonds) == 0:
            raise ValueError("No bonds left to run; you should restart simulation and run setup  again")

        pastBonds = self.curBonds
        self.curBonds = self.allBonds.pop(0)  # getting current bonds
        bondsRemove = [i for i in pastBonds if i not in self.curBonds]
        bondsAdd = [i for i in self.curBonds if i not in pastBonds]
        bondsStay = [i for i in pastBonds if i in self.curBonds]
        if verbose:
            print("{0} bonds stay, {1} new bonds, {2} bonds removed".format(len(bondsStay),
                                                                            len(bondsAdd), len(bondsRemove)))
        bondsToChange = bondsAdd + bondsRemove
        bondsIsAdd = [True] * len(bondsAdd) + [False] * len(bondsRemove)
        for bond, isAdd in zip(bondsToChange, bondsIsAdd):
            ind = self.bondToInd[bond]
            paramset = self.activeParamDict if isAdd else self.inactiveParamDict
            self.bondForce.setBondParameters(ind, bond[0], bond[1], **paramset)  # actually updating bonds
        self.bondForce.updateParametersInContext(context)  # now run this to update things in the context
        return self.curBonds, pastBonds
    
    
def initModel():
    # this jsut inits the simulation model. Put your previous init code here 
    birthArray = np.zeros(N, dtype=np.double) + 0.1
    deathArray = np.zeros(N, dtype=np.double) + 1. / LIFETIME
    stallDeathArray = np.zeros(N, dtype=np.double) + 1 / LIFETIME
    pauseArray = np.zeros(N, dtype=np.double)

    stallList = [1000, 3000, 5000, 7000, 9000]
    stallLeftArray = np.zeros(N, dtype = np.double)
    stallRightARray = np.zeros(N, dtype = np.double)
    for i in stallList:
        stallLeftArray[i] = 0.8
        stallRightARray[i] = 0.8

    smcNum = N // SEPARATION
    SMCTran = smcTranslocatorDirectional(birthArray, deathArray, stallLeftArray, stallRightARray, pauseArray,
                                         stallDeathArray, smcNum)
    return SMCTran


SMCTran = initModel()  # defining actual smc translocator object 

# ------------feed smcTran to the milker---
SMCTran.steps(1000000)  # first steps to "equilibrate" SMC dynamics. If desired of course. 


milker = smcTranslocatorMilker(SMCTran)   # now feed this thing to milker (do it once!)
#--------- end new code ------------


#!rm -r $folder
if os.path.exists(folder):
    shutil.rmtree(folder)
reporter = HDF5Reporter(folder=folder, max_data_length=100)



for milkerCount in range(milkerInitsTotal):
    doSave = milkerCount >= milkerInitsSkip
    
    # simulation parameters are defined below 
    a = Simulation(
            platform="cuda",
            integrator="variableLangevin", 
            error_tol=0.0002, 
            GPU = "0", 
            collision_rate=0.01, 
            N = len(data),
            reporters=[reporter],
            PBCbox=[box, box, box],
            precision="mixed")  # timestep not necessary for variableLangevin

    
    
    ############################## New code ##############################
    a.set_data(data)  # loads a polymer, puts a center of mass at zero
    
    
    # -----------Adding forces ---------------
    # sim.addForce(forces.sphericalConfinement(sim, density=0.3, k=1))
    #sim.addForce(forces.cylindricalConfinement(sim,cellRadius,bottom=-halfLength,top=halfLength,k=5))


    a.add_force(
        forcekits.polymer_chains(
            a,
            chains=[(0, None, None)],

                # By default the library assumes you have one polymer chain
                # If you want to make it a ring, or more than one chain, use self.setChains
                # self.setChains([(0,50,1),(50,None,0)]) will set a 50-monomer ring and a chain from monomer 50 to the end

            bond_force_func=forces.harmonic_bonds,
            bond_force_kwargs={
                'bondLength':1.0,
                'bondWiggleDistance':0.01, # Bond distance will fluctuate +- 0.05 on average
             },

            angle_force_func=forces.angle_force,
            angle_force_kwargs={
                'k':0.05
                # K is more or less arbitrary, k=4 corresponds to presistence length of 4,
                # k=1.5 is recommended to make polymer realistically flexible; k=8 is very stiff
            },

            nonbonded_force_func=forces.polynomial_repulsive,
            nonbonded_force_kwargs={
                'trunc':1.5, # this will let chains cross sometimes
                'radiusMult':1.05, # this is from old code
                #'trunc':10.0, # this will resolve chain crossings and will not let chain cross anymore
            },

            except_bonds=True,
             
        )
    )

    

#     ############################## OLD code ##############################
    a.step = block

    # ------------ initializing milker; adding bonds ---------
    # copied from addBond
    kbond = a.kbondScalingFactor / (smcBondWiggleDist ** 2)
    bondDist = smcBondDist * a.length_scale

    activeParams = {"length":bondDist,"k":kbond}
    inactiveParams = {"length":bondDist, "k":0}
    milker.setParams(activeParams, inactiveParams)
     
    # this step actually puts all bonds in and sets first bonds to be what they should be
    milker.setup(bondForce=a.force_dict['harmonic_bonds'],
                blocks=restartMilkerEveryBlocks,   # default value; milk for 100 blocks
                 smcStepsPerBlock=smcStepsPerBlock)  # now only one step of SMC per step
    print("Restarting milker")

    
    # If your simulation does not start, consider using energy minimization below
    a.local_energy_minimization() 
    
    
#     a.do_block(steps=steps,)#, increment=False)  # THIS RESULTS IN EK>10!!
    for i in range(restartMilkerEveryBlocks - 1):
        curBonds, pastBonds = milker.step(a.context)  # this updates bonds. You can do something with bonds here
        if i % saveEveryBlocks == (saveEveryBlocks - 2):  
            a.do_block(steps=steps)#, increment = doSave)    
            if doSave: 
#                 a.save()
                pickle.dump(curBonds, open(os.path.join(folder, "SMC{0}.dat".format(a.step)),'wb'))
        else:
            a.integrator.step(steps)  # do steps without getting the positions from the GPU (faster)

    data = a.get_data()  # save data and step, and delete the simulation
    block = a.step
    del a

    time.sleep(0.2)  # wait 200ms for sanity (to let garbage collector do its magic)
    
    
reporter.dump_data()
