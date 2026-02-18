__version__ = "0.1.1"

# Check for OpenMM installation
try:
    import openmm
except ImportError:
    import warnings
    warnings.warn(
        "\n"
        "OpenMM is not installed. Polychrom requires OpenMM for molecular dynamics simulations.\n"
        "\n"
        "Please install OpenMM with the appropriate backend for your system:\n"
        "  - For CUDA: pip install openmm[cuda13]\n"
        "  - For CPU:  pip install openmm\n"
        "\n"
        "Visit https://openmm.org for more information.",
        ImportWarning,
        stacklevel=2
    )
