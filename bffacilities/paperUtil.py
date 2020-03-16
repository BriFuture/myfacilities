#!/usr/env/bin python3

__version__ = '0.0.2'
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
        # self.papers = []

    def readConfig(self, file):
        pass

    def initArgs(self):
        """ @return ArgumentParser """
        parser = ArgumentParser(description=tr("Used for check downloaded papers and whether a paper is downloaded."))

        action = parser.add_mutually_exclusive_group()
        action.add_argument("-c", "--check", help=tr("Check whether the papers is downloaded ."))
        action.add_argument("-l", "--list", action="store_true",
                            help=tr("List existing papers under <{}>.").format(os.getcwd()))
        action.add_argument("-o", "--open", help=tr("Opening papers"))

        parser.add_argument("-V", "--version", 
            help=tr("Print this script version"), action="version", version=f'paperUtil {__version__}')
        self.parser = parser

    def parseArgs(self, argv):
        args = self.parser.parse_args(argv)
        if args.check:
            check = args.check
            self._check_paper(check, False)
        elif args.open:
            self._open_paper(args.open)
        else:
            # default action
            self._list_papers(os.getcwd())

    def _list_papers(self, dir, printable=True):
        path = Path(dir)
        dirs = [x for x in path.iterdir() if x.is_dir() and x.name.isdigit()]
        if printable: print('current', path)
        paperCount = 0
        self.papers = {} 
        for p in dirs:
            pdirs = [x.name.strip() for x in p.iterdir() if x.is_file()]
            self.papers[p.name] = pdirs
            paperCount += len(pdirs)
            if not printable:
                continue
            print(f"folder: {p.name} ({len(pdirs)})")
            for d in pdirs:
                print('- ', d)
            print('--------------------\n')
        if printable:
            print(f'Summary: {paperCount} papers')

    def _check_paper(self, name: str, strict=True):
        self._list_papers(os.getcwd(), False)
        name = name.strip()
        if strict:
            for pdir, files in self.papers.items():
                if name in files:
                    print(f"{Fore.GREEN}[Found] {name}{Style.RESET_ALL}")
                    return pdir
        else:
            found = False
            for pdir, files in self.papers.items():
                for f in files:
                    if f.find(name) >= 0:
                        print(f"{Fore.GREEN}[Possible] {f}{Style.RESET_ALL}")
                        found = (name == f) if found is False else True
            if found:
                print(f"\n{Fore.GREEN}[Found] {name}{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.RED}[Not Found] {name}{Style.RESET_ALL}")
        return None

    def _open_paper(self, name):
        import subprocess, platform
        curDir = os.getcwd()
        fdir = self._check_paper(name)
        if fdir is None:
            return
        try:
            if platform.system() == 'Darwin':       # macOS
                file = f"{curDir}/{fdir}/'{name}'"
                subprocess.call(('open', file))
            elif platform.system() == 'Windows':    # Windows
                file = f"{curDir}\\{fdir}\\{name}"
                os.startfile(file)
                # subprocess.run(('open', file), check=True)
            else:                                   # linux variants
                file = f"{curDir}/{fdir}/'{name}'"
                subprocess.call(('xdg-open', file))
        except Exception as e:
            logger.warning(e)
def main(argv = None):
    pu = PaperUtil()
    pu.initArgs()
    if argv is None: argv = sys.argv[1:]
    pu.parseArgs(argv)