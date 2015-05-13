
import os
import sys
from setuptools import setup
from distutils.util import convert_path

sys.path.insert(0, os.path.join('.', 'kmodes'))

main_ns = {}
ver_path = convert_path('kmodes/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name='kmodes',
    version=main_ns['__version__']
)
