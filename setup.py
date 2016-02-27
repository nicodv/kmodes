from setuptools import setup

import kmodes

DESCRIPTION = __doc__
VERSION = kmodes.__version__

setup(
    name='kmodes',
    packages=['kmodes'],
    version=VERSION,
    url='https://github.com/nicodv/kmodes',
    author='Nico de Vos',
    author_email='njdevos@gmail.com',
    license='MIT',
    description=DESCRIPTION,
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        'numpy>=1.10.4',
        'scikit-learn>=0.17.1',
        'scipy>=0.17.0',
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
                 'Programming Language :: Python :: 3.3',
                 'Programming Language :: Python :: 3.4',
                 'Programming Language :: Python :: 3.5',
                 'Topic :: Scientific/Engineering'],
)
