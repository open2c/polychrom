"""
Forcekits (new in polychrom)
----------------------------

The goal of the forcekits is two-fold. First, sometimes communication between forces is required. 
Since explicit is better than implicit, according to The Zen of Python, we are trying to 
avoid communication between forces using hidden variables (as done in openmm-polymer), and make it explicit. 
Forcekits are the tool to implement groups of forces that go together, as to avoid hidden communication between forces. 
Second, some structures can be realized using many different forces: e.g. polymer chain connectivity
can be done using harmonic bond force, FENE bonds, etc. Forcekits help avoid duplicating code, and allow swapping 
one force for another and keeping the topology/geometry of the system the same

The only forcekit that we have now implements polymer chain connectivity. It then explicitly adds exclusions 
for all the polymer bonds into the nonbonded force, without using hidden variables for communication between forces. 
It also allows using any bond force, any angle force, and any nonbonded force, allowing for easy swapping of 
one force for another without duplicating code. 

"""

import numpy as np
from . import forces



def polymer_chains(
    sim_object,
    chains=[(0, None, False)],
    bond_force_func=forces.harmonic_bonds,
    bond_force_kwargs={"bondWiggleDistance": 0.05, "bondLength": 1.0},
    angle_force_func=forces.angle_force,
    angle_force_kwargs={"k": 0.05},
    nonbonded_force_func=forces.polynomial_repulsive,
    nonbonded_force_kwargs={"trunc": 3.0, "radiusMult": 1.0},
    except_bonds=True,
    extra_bonds=None,
    extra_triplets=None,
    override_checks=False,
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

    except_bonds : bool
        If True then do not calculate non-bonded forces between the
        particles connected by a bond. True by default.
        
    extra_bonds : None or list
        [(i,j)] list of extra bonds. Same for extra_triplets. 
    
    override_checks: bool
        If True then do not check that all monomers are a member of exactly
        one chain. False by default. Note that overriding checks does not
        get automatically "passed on" to bond/angle force functions so you
        may need to specify override_checks=True in the respective kwargs
        as well.
    """

    force_list = []

    bonds = [] if ((extra_bonds is None) or len(extra_bonds) == 0) else [tuple(b) for b in extra_bonds]
    triplets = extra_triplets if extra_triplets else []
    newchains = []

    for start, end, is_ring in chains:
        end = sim_object.N if (end is None) else end
        newchains.append((start, end, is_ring))

        bonds += [(j, j + 1) for j in range(start, end - 1)]
        triplets += [(j - 1, j, j + 1) for j in range(start + 1, end - 1)]

        if is_ring:
            bonds.append((start, end - 1))
            triplets.append((int(end - 2), int(end - 1), int(start)))
            triplets.append((int(end - 1), int(start), int(start + 1)))

    # check that all monomers are a member of exactly one chain
    if not override_checks:
        num_chains_for_monomer = np.zeros(sim_object.N, dtype=int)
        for chain in newchains:
            start, end, _ = chain
            num_chains_for_monomer[start:end] += 1

        errs = np.where(num_chains_for_monomer != 1)[0]
        if len(errs) != 0:
            raise ValueError(
                f"Monomer {errs[0]} is a member of {num_chains_for_monomer[errs[0]]} chains. Set override_checks=True to override this check."
            )

    report_dict = {
        "chains": np.array(newchains, dtype=int),
        "bonds": np.array(bonds, dtype=int),
        "angles": np.array(triplets),
    }
    for reporter in sim_object.reporters:
        reporter.report("forcekit_polymer_chains", report_dict)

    if bond_force_func is not None:
        force_list.append(bond_force_func(sim_object, bonds, **bond_force_kwargs))

    if angle_force_func is not None:
        force_list.append(angle_force_func(sim_object, triplets, **angle_force_kwargs))

    if nonbonded_force_func is not None:
        nb_force = nonbonded_force_func(sim_object, **nonbonded_force_kwargs)

        if except_bonds:
            exc = list(set([tuple(i) for i in np.sort(np.array(bonds), axis=1)]))
            if hasattr(nb_force, "addException"):
                print(
                    "Exclude neighbouring chain particles from {}".format(nb_force.name)
                )
                for pair in exc:
                    nb_force.addException(int(pair[0]), int(pair[1]), 0, 0, 0, True)

                num_exc = nb_force.getNumExceptions()

            # The built-in LJ nonbonded force uses "exclusions" instead of "exceptions"
            elif hasattr(nb_force, "addExclusion"):
                print(
                    "Exclude neighbouring chain particles from {}".format(nb_force.name)
                )
                nb_force.createExclusionsFromBonds([(int(b[0]), int(b[1])) for b in bonds], int(except_bonds))
                    # for pair in exc:
                    #     nb_force.addExclusion(int(pair[0]), int(pair[1]))
                num_exc = nb_force.getNumExclusions()

            print("Number of exceptions:", num_exc)

        force_list.append(nb_force)

    return force_list
