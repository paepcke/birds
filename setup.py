import multiprocessing
from setuptools import setup, find_packages
from setuptools.command.test import test as setup_test
import os
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "birdsong",
    version = "0.1",
    #****packages = find_packages(),
    test_suite='nose2.collector.collector',
    # Dependencies on other packages:
    # Couldn't get numpy install to work without
    # an out-of-band: sudo apt-get install python-dev
    setup_requires   = ['pytest-runner'],
    install_requires = ['scikit-learn>=0.23.1',
                        'pandas>=1.1.3',
                        'numpy>=1.19.1',           
                        'torch>=1.7.1',       
                        'torchvision>=0.8.2', 
                        'seaborn>=0.11.0',
                        'requests>=2.24.0',
                        'torchaudio>=0.7.2',
                        'PyQt5>=5.15.2',
                        'matplotlib>=3.3.0',
                        'librosa>=0.7.2',
                        'scipy>=1.5.2',
                        'SoundFile>=0.10.3.post1',
                        'logging-singleton>=1.0',
                        'GPUtil>=1.4.0',
                        'tensorboard>=2.4.0',
                        'natsort>=7.1.0',
                        'Pillow>=8.0.1',
                        'json5>=0.9.5',
                        'nose2>=0.9.2',     # For testing
                        'tensorflow-plot>=0.3.2',
                        'psutil>=5.8.0',
                        'adjustText>=0.7.3',
                        ],

    tests_require    =[
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

print("To run tests, type 'nose2'")

