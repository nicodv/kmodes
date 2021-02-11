from setuptools import setup, find_packages

import kmodes

VERSION = kmodes.__version__

setup(
    name='kmodes',
    packages=find_packages(exclude=[
        '*.tests',
        '*.tests.*',
    ]),
    version=VERSION,
    url='https://github.com/nicodv/kmodes',
    author='Nelis J de Vos',
    author_email='njdevos@gmail.com',
    license='MIT',
    description='Python implementations of the k-modes and k-prototypes '
                'clustering algorithms for clustering categorical data.',
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        # Note: minimum numpy and scipy versions should ideally be the same
        # as what scikit-learn uses, but versions of numpy<1.10.4
        # give import problems.
        'numpy>=1.10.4',
        'scikit-learn>=0.22.0',
        'scipy>=0.13.3',
        'joblib>=0.11'
    ],
    extras_requires=[
        'nose',
    ],
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 'Topic :: Scientific/Engineering'],
)
