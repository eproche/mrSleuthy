#!/usr/bin/env python

import os
from setuptools import setup

setup(
	name="mrSleuthy",
	version='1.0',
	py_modules=['sleuthin'],
	install_requires=['Click', 'nltk', 'gensim', 'python-docx', 'logging', 
	'pillow', 'numpy', 'scipy', 'networkx', 'matplotlib',
	'scikit-learn', 'pandas', 'simplejson', 'py'
	],
	entry_points='''
		[console_scripts]
		sleuth=sleuthin:cli 
	'''
)