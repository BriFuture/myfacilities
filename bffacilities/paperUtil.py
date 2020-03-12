#!/usr/env/bin python3

__version__ = '0.0.1'
from ._util import createLogger, initGetText

tr = initGetText("gitrepo")
logger = createLogger("gitrepo.log", stream=True)

from ._frame import Frame
from pathlib import Path
from argparse import ArgumentParser
import os
class PaperUtil(Frame):
    def __init__(self):
        self.config = {}

    def readConfig(self, file):
        pass

    def initArgs(self):
        """ @return ArgumentParser """
        parser = ArgumentParser(description=tr("Used for check downloaded papers and whether a paper is downloaded."))

        action = parser.add_mutually_exclusive_group()
        action.add_argument("-c", "--check", help=tr("Check whether the papers is downloaded ."))
        action.add_argument("-l", "--list", action="store_true",
                            help=tr("List existing papers under <{}>.").format(os.getcwd()))

        parser.add_argument("-V", "--version", 
            help=tr("Print this script version"), action="version", version=f'paperUtil {__version__}')
        self.parser = parser

    def parseArgs(self, argv):
        args = self.parser.parse_args(argv)
        if args.check:
            check = args.check
            print(check)
        else:
            # default action
            _list_papers(os.getcwd())

def _list_papers(dir):
    path = Path(dir)
    dirs = [x for x in path.iterdir() if x.is_dir() and x.name.isdigit()]
    for p in dirs:
        pdirs = [('- ' + x.name) for x in p.iterdir() if x.is_file()]
        papers = "\n".join(pdirs)
        print(f"{p.name} ({len(pdirs)})")
        print(papers)
        print()
    print('current', path)


def main(argv = None):
    pu = PaperUtil()
    pu.initArgs()
    if argv is None: argv = sys.argv[1:]
    pu.parseArgs(argv)