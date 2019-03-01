# -*- coding: utf-8 -*-

# from distutils.core import setup
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt', "r", encoding="utf-8") as f:
    requires = f.read().splitlines()

setup( 
    name = "brifuture-facilities", 
    packages = find_packages(where='.'), 
    version = "0.0.1alpha4",

    entry_points = {
        "console_scripts": [
            'bf_broadcast = myfacilities.broadcast:main',
            'bf_gitrepo   = myfacilities.gitrepo:main',
        ]
    },

    description = "BriFuture's scripts set, all scripts will be written with Python3",
    author = "BriFuture",
    author_email = "jw.brifuture@gmail.com",
    license = "GPLv3",
    url = "http://github.com/brifuture/",
    
    install_requires = requires,

    include_package_data = True,
    zip_safe=True,
    exclude_package_data = {'': ['__pycache__']},

    # download_url = "",
    keywords = [ "webserver", "shadowsocks-manager" ],
    classifiers = [ 
        "Programming Language :: Python", 
        "Programming Language :: Python :: 3" ,
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent"
    ],

    long_description = long_description,
    long_description_content_type="text/markdown",
)