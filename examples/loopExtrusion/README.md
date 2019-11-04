This example contains a draft of a new extrusion simulation code. 

Now we separate extrusion code into two parts: 1D part that records a trajctory, and a 3D part that does a simulation

We have two 1D parts: 

* extrusion_1D_translocator.ipynb   - an old code using "SMCTranslocator" class 
* extrusion_1D_newCode.ipynb  - a draft of a new code using pure python 

Two 1D simulations are different, use different methodology, and are not intended to be the same. 

Old 1D simulation uses CTCFs that capture LEF with some probability and never release it.
New 1D simulation uses CTCFs that capture and release LEF with some probability. 
Thus, new 1D simulation is more general than the old 1D code. 

What is new for both 1D simulations is that they both simulate 10 copies of a 4000-monomer system. 
This is done to speed up simulations: simulations of a 4000-monomer system go not much faster than simulations 
of a 40,000 monomer system, but in the latter case we get 10 times more statistics.
It is generally advisable to simulate systems with at least 20,000 monomers to get the most out of the GPU. 

3D simulation uses a trajectory recorded by one of the 1D parts (it is saved into folder "trajectory" by either of them)
It then performs a 3D simulation and puts it in the same folder 

sample_contactmap.ipynb notebook shows an example of how to generate a contactmap. 