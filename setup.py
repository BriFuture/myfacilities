# -*- coding: utf-8 -*-

# from distutils.core import setup
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

requires = []
with open('requirements.txt', "r", encoding="utf-8") as f:
    for line in f:
        if line.startswith("#"): continue
        requires.append(line)
requires = ",".join(requires)

from bffacilities import __version__

setup( 
    name = "bffacilities", 
    packages = find_packages(where='.'), 
    version = __version__,

    entry_points = {
        "console_scripts": [
            'bffacility  = bffacilities.__main__:main',
            'bff  = bffacilities.__main__:main',
        ],

    },

    description = "BriFuture's scripts set, all scripts will be written with Python3",
    author = "BriFuture",
    author_email = "jw.brifuture@gmail.com",
    license = "GPLv3",
    url = "http://github.com/brifuture/",
    
    install_requires = requires,

    include_package_data = True,
    zip_safe=False,
    # package_data = {
    #     '':['./bffacilities/locale/*', './bffacilities/flask/*.py', './bffacilities/imgs/*'],
    # },
    exclude_package_data = {'': ['__pycache__']},

    # download_url = "",
    keywords = [ "facilities" ],
    classifiers = [ 
        "Programming Language :: Python", 
        "Programming Language :: Python :: 3" ,
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent"
    ],

    long_description = long_description,
    long_description_content_type="text/markdown",
)