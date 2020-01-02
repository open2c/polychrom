from polychrom import polymer_analyses
import numpy as np


# testing smart contacts
data = np.random.random((200,3)) * 10 # generate test data

conts = polymer_analyses.calculate_contacts(data, 2.5)  # these are regular contacts
c2 = polymer_analyses.smart_contacts(data, 2.5)        #these are smart contacts - every second monomer is taken

ind = [c2[0][0] % 2]                                   #figure out which every second monomer is it
mask = ((conts[:,0] % 2) == ind) * ((conts[:,1] % 2) == ind)   # manually take every second monomer

subconts = conts[mask]   # select the right contacts

ind_smart = np.sort(c2[:,0] * 10000 + c2[:,1])   # generate unique indices based on contacts; sort them.
ind_regular = np.sort(subconts[:,0] * 10000 + subconts[:,1])

assert np.all(ind_smart == ind_regular)      #assert we got all the same contacts