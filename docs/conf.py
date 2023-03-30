# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import re
import shlex
import sys

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# sys.path.insert(0, os.path.abspath('.'))

## We shall consider removing the sys path changes according to https://www.sphinx-doc.org/en/master/tutorial/describing-code.html#including-doctests-in-your-documentation
## if using pyproject.toml
sys.path.insert(0, os.path.abspath(".."))


def skip(app, what, name, obj, would_skip, options):
    if name == "__init__":
        return False
    return would_skip


def setup(app):
    app.connect("autodoc-skip-member", skip)


# autodoc_mock_imports = [
#     'numpy',
#     'scipy',
#     'pandas',
#     'h5py',
#     'dask',
#     'cytoolz',
# ]
import mock

MOCK_MODULES = [
    "numpy",
    "scipy",
    "scipy.sparse",
    "scipy.spatial",
    "scipy.interpolate",
    "scipy.ndimage",
    "pandas",
    "pandas.algos",
    "pandas.api",
    "pandas.api.types",
    "h5py",
    "dask",
    "dask.base",
    "dask.array",
    "dask.dataframe",
    "dask.dataframe.core",
    "dask.dataframe.utils",
    "simtk",
    "simtk.unit",
    "simtk.unit.nanometer",
    "simtk.openmm",
    "joblib",
    "scipy.interpolate.fitpack2",
]
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # 'numpydoc'
]

numpydoc_show_class_members = False
napoleon_use_rtype = False

# -- Project information -----------------------------------------------------

project = "polychrom"
copyright = "2023, Mirny lab"
author = "Mirny lab"
master_doc = "index"

# The full version, including alpha/beta/rc tags
# Read the version from the __init__ file
import io
import re


def _read(*parts, **kwargs):
    filepath = os.path.join(os.path.dirname(__file__), *parts)
    encoding = kwargs.pop("encoding", "utf-8")
    with io.open(filepath, encoding=encoding) as fh:
        text = fh.read()
    return text


def get_version():
    version = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        _read("../polychrom", "__init__.py"),
        re.MULTILINE,
    ).group(1)
    return version


version = get_version()
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The encoding of source files.
# source_encoding = 'utf-8-sig'

source_parsers = {".md": "recommonmark.parser.CommonMarkParser"}
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}
# html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []  # removed the path to static files
