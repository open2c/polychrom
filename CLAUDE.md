# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Polychrom is an Open2C polymer simulation library designed to build mechanistic models of chromosomes. It simulates biological processes subject to forces or constraints, which are then compared to Hi-C maps, microscopy, and other data sources.

## Development Commands

### Installation
```bash
pip install cython  # Required dependency
pip install -r requirements.txt
pip install -e .  # Install in development mode
```

### Testing
```bash
pytest  # Run all tests
```

### Building Cython Extensions
```bash
python setup.py build_ext --inplace
```

## Architecture

### Core Components

**Simulation Module** (`polychrom/simulation.py`)
- Central `Simulation` class manages the entire simulation lifecycle
- Handles platform setup (CUDA/OpenCL/CPU), integrators, and parameters
- Methods: `set_data()` for loading conformations, `add_force()` for adding forces, `doBlock()` for running simulation steps

**Forces System** (`polychrom/forces.py`, `polychrom/forcekits.py`)
- Forces define polymer behavior: connectivity, confinement, crosslinks, tethering
- Individual forces in `forces.py` are functions that create OpenMM force objects
- Complex force combinations are packaged as "forcekits" (e.g., `polymer_chains`)
- Legacy forces available in `polychrom/legacy/forces.py`

**Data Storage** (`polychrom/hdf5_format.py`)
- HDF5Reporter handles simulation output in HDF5 format
- Backwards compatibility with legacy format via legacy reporter
- `polymerutils.load()` function reads both new and old formats

**Starting Conformations** (`polychrom/starting_conformations.py`)
- Functions to generate initial polymer configurations
- Example: `grow_cubic()` creates a cubic lattice conformation

### Key Design Patterns

1. **Force Architecture**: Forces are simple functions that wrap OpenMM force objects, returning the force with a `.name` attribute
2. **Simulation Flow**: Initialize Simulation → Load data → Add forces → Run blocks in loop → Save via reporter
3. **Extensibility**: Users can define custom forces in their scripts following the pattern in `forces.py`

## Important Notes

- OpenMM is the underlying engine (required dependency not in requirements.txt)
- Cython extensions in `_polymer_math.pyx` require compilation
- Main use case is loop extrusion simulations (see `examples/loopExtrusion/`)
- Testing uses pytest with configuration in `pytest.ini`
