#! /usr/bin/env python
"""
Setup for sidelobe_finder
"""
import os
import sys
from setuptools import setup


def read(fname):
	"""Read a file"""
	return open(os.path.join(os.path.dirname(__file__), fname)).read()


def get_version():
	""" Get the package version number """
	import sidelobe_finder
	return sidelobe_finder.__version__



PY_MAJOR_VERSION=sys.version_info.major
PY_MINOR_VERSION=sys.version_info.minor
print("PY VERSION: maj=%s, min=%s" % (PY_MAJOR_VERSION,PY_MINOR_VERSION))

reqs= []
reqs.append('numpy>=1.10')
reqs.append('astropy>=2.0, <3')


if PY_MAJOR_VERSION<=2:
	print("PYTHON 2 detected")
	reqs.append('future')
	reqs.append('scipy<=1.2.1')
	reqs.append('scikit-learn>=0.20')
	reqs.append('pyparsing>=2.0.1')
	reqs.append('matplotlib<=2.2.4')
else:
	print("PYTHON 3 detected")
	reqs.append('scipy')
	reqs.append('scikit-learn')
	reqs.append('pyparsing')
	reqs.append('matplotlib')

reqs.append('keras>=2.0')
reqs.append('tensorflow>=1.13')
reqs.append('opencv-python')
reqs.append('h5py')

data_dir = 'data'

setup(
	name="sidelobe_finder",
	version=get_version(),
	author="Simone Riggi",
	author_email="simone.riggi@gmail.com",
	description="Tool to detect radio source sidelobes from astronomical FITS images",
	license = "GPL3",
	url="https://github.com/SKA-INAF/sidelobe-finder",
	long_description=read('README.md'),
	packages=['sidelobe_finder'],
	install_requires=reqs,
	scripts=['scripts/train_frcnn.py','scripts/test_frcnn.py','scripts/train.py'],
)
