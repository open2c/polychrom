#!/bin/bash
sphinx-build . _build
cp -r  _build/*  ../../manual_pages/mimakaev.bitbucket.org/
cd ../../manual_pages/mimakaev.bitbucket.org
hg add * 
hg commit -m "Automatic commit from a new build" 
hg push 

