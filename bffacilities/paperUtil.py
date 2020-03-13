#!/usr/env/bin python3

__version__ = '0.0.1'
from ._util import createLogger, initGetText

tr = initGetText("gitrepo")
logger = createLogger("gitrepo.log", stream=True)

from ._frame import Frame
from pathlib import Path
from argparse import ArgumentParser
import os
from colorama import Fore, Style
class PaperUtil(Frame):
    def __init__(self):
        self.config = {}
        self.papers = []

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
            self._check_paper(check)
        else:
            # default action
            self._list_papers(os.getcwd())

    def _list_papers(self, dir, printable=True):
        path = Path(dir)
        dirs = [x for x in path.iterdir() if x.is_dir() and x.name.isdigit()]
        if printable: print('current', path)
        paperCount = 0
        for p in dirs:
            pdirs = []
            for x in p.iterdir():
                if x.is_file():
                    pdirs.append(x.name)
                    self.papers.append(x.name.upper())
                    paperCount += 1
            if not printable:
                continue
            print(f"folder: {p.name} ({len(pdirs)})")
            for d in pdirs:
                print('- ', d)
            print('--------------------\n')
        print(f'Summary: {paperCount} papers')

    def _check_paper(self, name):
        self._list_papers(os.getcwd(), False)
        if name.upper() in self.papers:
            print(f"{Fore.GREEN}[Found] < {name} > {Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}[Not Found] < {name} > {Style.RESET_ALL}")
        # print(name)
        pass

def main(argv = None):
    pu = PaperUtil()
    pu.initArgs()
    if argv is None: argv = sys.argv[1:]
    pu.parseArgs(argv)