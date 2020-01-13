.. polychrom documentation master file, created by
   sphinx-quickstart on Sat Jan  4 15:17:36 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the documentation for the polychrom package
======================================================

Polychrom is a package for setting up, performing and analyzing polymer simulations of chromosomes. 
The simulation part is based around VJ Pande's OpenMM library - a GPU-assisted framework for general molecular dynamics simulations. 
The analysis part is written by the mirnylab. 

Installation
------------

Polychrom requires OpenMM, which can be installed through conda: ``conda install -c omnia openmm``. See http://docs.openmm.org/latest/userguide/application.html#installing-openmm . In our experience, adding ``-c conda-forge`` listed in the link above is optional. 

CUDA is the fastest GPU-assisted backend to OpenMM. You would need to have the required version of CUDA, or install OpenMM compiled for your version of CUDA. 

Other dependencies are simple, and are listed in requirements.txt. All but joblib are installable from either conda/pip, and joblib installs well with pip. 


Structure
---------

Polychrom is an API, and each simulation has to be set up as a Python script. Simulations are done using a :ref:`polychrom-simulation` submodule (:class:`polychrom.simulation.simulation`)



.. toctree::
   :maxdepth: 2   
   
   polychrom.simulation
   polychrom.contactmaps


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
