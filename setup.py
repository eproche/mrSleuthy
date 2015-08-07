#!/usr/bin/env python

import os
from setuptools import setup

setup(
	name="mrSleuthy",
	version='1.0',
	authors='Evan Roche, Clinton Burgos',
	author_email='eroche@lclark.edu',
	license='MIT',
	py_modules=['sleuthin'],
	install_requires=['Click', 'nltk', 'gensim', 'python-docx', 'logging', 
	'pillow', 'numpy', 'scipy', 'networkx', 'matplotlib',
	'scikit-learn', 'pandas', 'simplejson', 'py', 'scikit-image'
	],
	entry_points='''
		[console_scripts]
		sleuth=sleuthin:cli 
	'''
)