
from setuptools import setup
import sys
import os
sys.path.insert(0, os.path.join('.', 'kmodes'))
from kmodes import __version__
setup(
    name='kmodes',
    version=__version__
)
