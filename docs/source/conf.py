# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import os, sys
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../marabou'))


# -- Project information -----------------------------------------------------

project = 'contextual-robustness'
copyright = '2021, DeepCert'
author = 'DeepCert'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc'
    ]

templates_path = ['_templates']

exclude_patterns = [
    '../../venv',
    '../../docs',
    '../../sandbox'
    ]

# -- Options for HTML output -------------------------------------------------

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']
