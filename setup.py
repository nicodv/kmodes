
from setuptools import setup
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('kmodes/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name='kmodes',
    version=main_ns['__version__'],
    url='https://github.com/nicodv/kmodes',
    author='Nico de Vos',
    author_email='njdevos@gmail.com',
    packages=['kmodes'],
    license='MIT',
    description='A Python implementation of the k-modes/k-prototypes clustering algorithms.',
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        'numpy>=1.10.4',
        'scikit-learn>=0.17.1',
        'scipy>=0.17.0',
    ],
)
