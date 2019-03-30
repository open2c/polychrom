from openmmlib import pymol_show
import numpy as np

rw = .4 * np.cumsum(np.random.random((1000, 3)) - 0.5, axis=0)

def example_pymol():
    # Creating a random walk


    # Highlighting first 100 monomers and then 200-400
    regions = [(000, 100), (100, 200)]

    # Coloring them red and blue
    colors = ["red", "green"]

    # Making red semi-transparent
    transp = [0.7, 0]

    # Running the script with default chain radiuses
    pymol_show.do_coloring(
                data=rw,
                regions=regions,
                colors=colors,
                transparencies=transp,
                spherePositions=[500, 600],
                sphereRadius=0.3)


#this runs example with doColoring
example_pymol()

#this is just showing a chain
pymol_show.show_chain(rw, chain_radius=0.08)
