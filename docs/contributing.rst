Building and checking the documentation
=======================================

From the repository root, using Python 3.12 in an activated virtual environment::

    python -m pip install -e ".[dev]" -r docs/requirements.txt
    python -m sphinx -b html -n -W --keep-going docs docs/_build/html
    python -m sphinx -b doctest -W --keep-going docs docs/_build/doctest
    python examples/quickstart_kmodes.py
    python examples/quickstart_kprototypes.py
    python -m pytest

Open ``docs/_build/html/index.html`` to view the result. The HTML build fails on
warnings, including unresolved references. The doctest build checks the Python
sessions in the parameter guide; the quickstart scripts check cluster grouping,
predictions, centroid values or shape, and cost without relying on cluster IDs.
These documentation checks also run in the documentation CI workflow.

The API reference imports the installed package and renders its NumPy-style
docstrings with Sphinx autodoc and Napoleon. Update the docstrings when changing
parameters. The quickstart pages include the example scripts directly, so the
displayed code and executable examples stay together.

Read the Docs
-------------

The root ``.readthedocs.yaml`` selects Python 3.12, installs the local package
and ``docs/requirements.txt``, and builds with warnings treated as errors.
A repository maintainer must import the GitHub repository into Read the Docs
and enable its webhook to publish builds. Adding the configuration alone does
not create a hosted project. See the
`Read the Docs configuration reference
<https://docs.readthedocs.com/platform/stable/config-file/v2.html>`_.
