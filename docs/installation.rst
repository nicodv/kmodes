Installation
============

Install the released package in your Python environment::

    python -m pip install kmodes

This also installs NumPy, SciPy, scikit-learn, and joblib. Pandas is optional;
the quickstarts use only NumPy and do not download datasets.

To work on the current development version::

    git clone https://github.com/nicodv/kmodes.git
    cd kmodes
    python -m venv .venv

Activate the environment (``source .venv/bin/activate`` on Linux/macOS or
``.venv\Scripts\activate`` on Windows), then install the checkout::

    python -m pip install -e .

Run Python from outside a file named ``kmodes.py`` so it does not hide the
installed package. See :doc:`contributing` to build these docs.
