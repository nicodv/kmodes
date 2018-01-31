from setuptools import setup, find_packages

import kmodes

DESCRIPTION = kmodes.__doc__
VERSION = kmodes.__version__

setup(
    name='kmodes',
    packages=find_packages(exclude=[
        '*.tests',
        '*.tests.*',
    ]),
    version=VERSION,
    url='https://github.com/nicodv/kmodes',
    author='Nico de Vos',
    author_email='njdevos@gmail.com',
    license='MIT',
    description=DESCRIPTION,
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        # Note: minimum numpy and scipy versions should ideally be the same
        # as what scikit-learn uses, but versions of numpy<1.10.4
        # give import problems.
        # scikit-learn version is capped to avoid compatibility issues.
        'numpy>=1.10.4',
        'scikit-learn>=0.19.0, <0.20.0',
        'scipy>=0.13.3',
        'matching'
    ],
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 2',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Scientific/Engineering'],
)
