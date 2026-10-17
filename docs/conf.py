"""Sphinx configuration; install the local package before building."""

from kmodes import __version__

project = 'kmodes'
author = 'Nelis J. de Vos'
copyright = '2015–2026, Nelis J. de Vos'
release = __version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
]
autodoc_member_order = 'bysource'
napoleon_numpy_docstring = True
napoleon_google_docstring = False
# Keep descriptive NumPy types (including array shapes) as readable text.
napoleon_use_param = False
napoleon_use_rtype = False
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
html_theme = 'alabaster'
html_title = f'kmodes {release} documentation'
