import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="sift2d",
    version="0.2",
    author="Stefan Mordalski",
    author_email="stefanm@if-pan.krakow.pl",
    description=("A library to generate and manipulate 2D-SIFt interaction matrices."),
    license="BSD",
    packages=['sift2d',],
    long_description=read('README.md'),
)
