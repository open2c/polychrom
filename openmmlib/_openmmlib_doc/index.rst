.. openmm-lib documentation master file, created by
   sphinx-quickstart on Mon Mar 26 21:38:46 2012.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to openmm-lib's documentation!
======================================

This is a wrapper to OpenMM designed to run and analyze polymer simulations (Brownyan dynamics). 

You can use this library if:

-You need to run a coarse-grained polymer simulation
-You have equally sized monomers
-You need to have ability to make monomers transparent
-You want a fast and efficient way to calculate things like Rg(s) and Pc(s) (Pc - contact probability)
-You want to deal with knot complexity (Alexander's polynomial) or linking number 
-You want to analyze contact maps
-You want to make cool pictures in pymol
-You want to replicate simulations from the Mirnylab papers. 

The main library, openmmlib, is a class to perform simulations. 

A file knotAnalysis.py contains code needed to calculate Alexander's polynomials,
a measure of knot complexity. It also uses OpenMM to help unwrap polymer and
reduce time needed to calculate Alexander's polynomial

A file polymerScalings.py has some utilities to calculate Rg(s) and Pc(s) for
polymer conformations in a fast and efficient way.

A file contactmaps.py has code to quickly calculate contacts within a polymer structure,
and organize them as contactmaps. It is used by polymerScalings.

A file pymol_show has utilities to visualize polymer conformations using pymol
 
Contents:

.. toctree::
   :maxdepth: 3

   openmmlib
   contactmaps
   knotAnalysis
   polymerutils
   polymerScalings
 


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

