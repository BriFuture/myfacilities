# -*- coding: utf-8 -*-

__version__ = "0.0.18"
__author__ = "BriFuture"

"""
Update Since 0.0.17: scripts in `myscripts` folder could add more scripts
without change main.py file.
"""

from .utils import createLogger, initGetText
from ._constants import BFF_ROOT_PATH, BFF_OTHER_PATH


#Dictionary with console color codes to print text
terminal_colors = {
    'HEADER' : "\033[95m",
    'OKBLUE' : "\033[94m",
    'RED' : "\033[91m",
    'OKYELLOW' : "\033[93m",
    'GREEN' : "\033[92m",
    'LIGHTBLUE' : "\033[96m",
    'WARNING' : "\033[93m",
    'FAIL' : "\033[91m",
    'ENDC' : "\033[0m",
    'BOLD' : "\033[1m",
    'UNDERLINE' : "\033[4m" 
}