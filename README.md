# polychrom

[![DOI](https://zenodo.org/badge/178608195.svg)](https://zenodo.org/badge/latestdoi/178608195)

## Open2C polymer simulation library

This library is designed to easily set up polymer simulations of chromosomes subject to a set of forces or constraints. 
Polychrom is designed to build mechanistic models - i.e. models that simulates a biological process. 
We then compare results of the simulation to Hi-C maps or microscopy, and to other sources of data. 

Polychrom is not designed to make a simulation based on a Hi-C map alone, without a hypothesis in mind. 
For distinction between mechanistic and data-driven models see https://pubmed.ncbi.nlm.nih.gov/26364723/ 

Polychrom documentation is hosted on readthedocs. 
https://polychrom.readthedocs.io/en/latest/

One of the main uses of polychrom now is to simulate loop extrusion. An example loop extrusion simulation is presented here.
https://github.com/mirnylab/polychrom/tree/master/examples/loopExtrusion . It is a good starting point for loop extrusion simulations. 

Simplest example simulation that does nothing, which is a good starting point for novel simulations or polymer physics projects.  
https://github.com/mirnylab/polychrom/blob/master/examples/example/example.py

### Transitioning from openmm-polymer ("openmmlib")
Compared to openmm-polymer, it has a new storage format described here: https://github.com/mirnylab/polychrom/blob/master/examples/storage_formats/hdf5_reporter.ipynb
For backwards compatibility with analysis routines, we have legacy reporter that saves into openmmlib format 
https://github.com/mirnylab/polychrom/blob/master/examples/storage_formats/legacy_reporter.ipynb

polychrom.polymerutils.load function  is backwards compatible with both new and old style 
format. https://github.com/mirnylab/polychrom/blob/master/polychrom/polymerutils.py#L22
