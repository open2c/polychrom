polychrom package
=================

Installation
------------

Polychrom requires OpenMM, which can be installed through conda: ``conda install -c omnia openmm``. See http://docs.openmm.org/latest/userguide/application.html#installing-openmm . In our experience, adding ``-c conda-forge`` listed in the link above is optional. 

CUDA is the fastest GPU-assisted backend to OpenMM. You would need to have the required version of CUDA, or install OpenMM compiled for your version of CUDA. 

Other dependencies are simple, and are listed in requirements.txt. All but joblib are installable from either conda/pip, and joblib installs well with pip. 


Structure
---------

Polychrom is an API, and each simulation has to be set up as a Python script. Simulations are done using a :ref:`polychrom-simulation` submodule (:class:`polychrom.simulation.simulation`)

Submodules
----------

.. toctree::

   polychrom.simulation
   polychrom.forcekits
   polychrom.forces
   polychrom.contactmaps
   polychrom.hdf5_format
   polychrom.polymer_analyses
   polychrom.polymerutils
   polychrom.pymol_show
   polychrom.starting_conformations

Module contents
---------------

.. automodule:: polychrom
   :members:
   :undoc-members:
   :show-inheritance:
