#!/usr/bin/env python
import sys 
from openmmlib import pymol_show
from openmmlib.polymerutils import load 

inFile = sys.argv[1] 
data = load(inFile) 

pymol_show.show_chain(data)

