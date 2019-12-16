# polychrom

[![DOI](https://zenodo.org/badge/178608195.svg)](https://zenodo.org/badge/latestdoi/178608195)

## This is a library in development. 


It is replacing openmm-polymer soon. 


It has a new storage format that is described here: 
https://github.com/mirnylab/polychrom/blob/master/examples/storage_formats/hdf5_reporter.ipynb

Example file is here: 
https://github.com/mirnylab/polychrom/blob/master/examples/example/example.py

And for backwards compatibility with analysis routines, we have legacy reporter
https://github.com/mirnylab/polychrom/blob/master/examples/storage_formats/legacy_reporter.ipynb


Although polymerutils.load can load new-style URIs (filename-like paths to individual blocks of simulations), and there is a new function polymerutils.fetch_block that should make loading individual files from the trajectory easier. 

