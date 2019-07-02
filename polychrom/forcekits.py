import numpy as np
from . import forces

def polymerChains(
    sim_object,
    chains=[(0, None, False)],

    bondForceFunc=forces.harmonicBonds,
    bondForceKwargs={'bondWiggleDistance':0.05,
                     'bondLength':1.0},

    angleForceFunc=forces.angleForce,
    angleForceKwargs={'k':0.05},

    nonbondedForceFunc=forces.polynomialRepulsiveForce,
    nonbondedForceKwargs={'trunc':3.0, 
                          'radiusMult':1.},

    exceptBonds=True,
):
    """Adds harmonic bonds connecting polymer chains

    Parameters
    ----------
    chains: list of tuples
        The list of chains in format [(start, end, isRing)]. The particle
        range should be semi-open, i.e. a chain (0,3,0) links
        the particles 0, 1 and 2. If bool(isRing) is True than the first
        and the last particles of the chain are linked into a ring.
        The default value links all particles of the system into one chain.

    exceptBonds : bool
        If True then do not calculate non-bonded forces between the
        particles connected by a bond. True by default.
    """

    force_list = []

    bonds = []
    triplets = []
    for start, end, is_ring in chains:
        end = sim_object.N if (end is None) else end
        
        bonds += [(j, j+1) for j in range(start, end - 1)]
        triplets += [(j - 1, j, j + 1) for j in range(start + 1, end - 1)]

        if is_ring:
            bonds.append((start, end-1))
            triplets.append((int(end - 2), int(end - 1), int(start)))
            triplets.append((int(end - 1), int(start), int(start + 1)))
    
    force_list.append(
        bondForceFunc(sim_object, bonds, **bondForceKwargs)
    )
    
    if angleForceFunc is not None:
        force_list.append(
            angleForceFunc(sim_object, triplets, **angleForceKwargs)
        )
    
    if nonbondedForceFunc is not None:
        nb_force = nonbondedForceFunc(sim_object, **nonbondedForceKwargs)
        
    if exceptBonds:
        exc = list(set([tuple(i) for i in np.sort(np.array(bonds), axis=1)]))
        if hasattr(nb_force, "addException"):
            print('Exclude neighbouring chain particles from {}'.format(nb_force.name))
            for pair in exc:
                nb_force.addException(int(pair[0]), int(pair[1]), 0, 0, 0, True)
                
        # The built-in LJ nonbonded force uses "exclusions" instead of "exceptions"
        elif hasattr(nb_force, "addExclusion"):
            print('Exclude neighbouring chain particles from {}'.format(nb_force.name))
            for pair in exc:
                nb_force.addExclusion(int(pair[0]), int(pair[1]))
                
        print("Number of exceptions:", len(bonds))

        force_list.append(nb_force)

    return force_list

            
