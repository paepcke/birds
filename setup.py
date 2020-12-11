import multiprocessing
from setuptools import setup, find_packages
import os
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "birdsong",
    version = "0.1",
    packages = find_packages(),

    # Dependencies on other packages:
    # Couldn't get numpy install to work without
    # an out-of-band: sudo apt-get install python-dev
    setup_requires   = ['pytest-runner'],
    install_requires = ['scikit-learn>=0.23.1',
                        'pandas>=1.1.3',
                        'numpy>=1.19.1',           
                        'torch>=1.5.1',       
                        'torchvision>=0.6.1', 
                        'seaborn>=0.11.0',
                        'requests>=2.24.0',
                        'torchaudio>=0.5.1',
                        'PyQt5>=5.15.1',
                        'matplotlib>=3.3.0',
                        'librosa>=0.7.2',
                        'scipy>=1.5.2',
                        'SoundFile>=0.10.3.post1',
                        'logging-singleton>=0.1',
                        ],

    tests_require    =['pytest',
                       'testfixtures>=6.14.1',
                       ],

    # metadata for upload to PyPI
    author = "Leo Glikbarg",
    author_email = "lglik@stanford.edu",
    description = "classify birdsong",
    long_description_content_type = "text/markdown",
    long_description = long_description,
    license = "BSD",
    keywords = "birdsong",
    url = "git@github.com:paepcke/birds.git",   # project home page, if any
)
