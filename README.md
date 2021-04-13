# polychrom

[![DOI](https://zenodo.org/badge/178608195.svg)](https://zenodo.org/badge/latestdoi/178608195)

## Open2C polymer simulation library

This library is designed to easily set up polymer simulations of chromosomes subject to a set of forces or constraints. 
Polychrom is designed to build mechanistic models - i.e. models that simulates a biological process. 
We then compare results of the simulation to Hi-C maps or microscopy, and to other sources of data. 

Polychrom is not designed to make a simulation based on a Hi-C map alone, without a hypothesis in mind. 
For distinction between mechanistic and data-driven models see https://pubmed.ncbi.nlm.nih.gov/26364723/ 

## Requirements

The following are required before installing polychrom:

* Python 3.7+
* `numpy`
* `cython`
* OpenMM

Polychrom requires OpenMM, which we currently recommend installing via conda: 
```sh
conda install -c omnia openmm
```
See http://docs.openmm.org/latest/userguide/application.html#installing-openmm. 
In our experience, adding ``-c conda-forge`` listed in the link above is optional. 
CUDA is the fastest GPU-assisted backend to OpenMM. To use this feature with polychrom, see OpenMM CUDA installation instructions. 

Other polychrom dependencies are simple, and are listed in requirements.txt. 
All but joblib are installable from either conda/pip, and joblib installs well with pip. 


## Installation

After installing cython and OpenMM, we recommend cloning the polychrom repository, going to its folder, and installing via:
```sh
pip install -e ./
```

## Getting Started 
Polychrom documentation is hosted on readthedocs. 
https://polychrom.readthedocs.io/en/latest/

A good starting point for novel simulations or polymer physics projects is a simulation of a single un-modified polymer chain:  
https://github.com/mirnylab/polychrom/blob/master/examples/example/example.py

One of the current main uses of polychrom is to simulate loop extrusion. An example loop extrusion simulation is presented here:
https://github.com/mirnylab/polychrom/tree/master/examples/loopExtrusion, and provides a good starting point for loop extrusion simulations. 


### Transitioning from openmm-polymer ("openmmlib")
A main difference between polychrom and openmm-polymer is its new storage format, described here:
https://github.com/mirnylab/polychrom/blob/master/examples/storage_formats/hdf5_reporter.ipynb

For backwards compatibility with analysis routines, a legacy reporter can save to openmmlib format 
https://github.com/mirnylab/polychrom/blob/master/examples/storage_formats/legacy_reporter.ipynb

polychrom.polymerutils.load function is backwards compatible with both new and old formats: https://github.com/mirnylab/polychrom/blob/master/polychrom/polymerutils.py#L22
