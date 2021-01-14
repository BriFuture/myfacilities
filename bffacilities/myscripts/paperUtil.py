#!/usr/env/bin python3
# -*- coding: utf-8 -*-

__version__ = '0.0.2'
from bffacilities.utils import createLogger, initGetText
from bffacilities._frame import Frame, common_entry
from pathlib import Path
from argparse import ArgumentParser
import os
from colorama import Fore, Style

tr = initGetText("gitrepo")
logger = createLogger("gitrepo.log", stream=True, savefile=False)

class PaperUtil(Frame):
    def __init__(self):
        self.config = {}
        # self.papers = []

    def readConfig(self, file):
        p = Path(".papers")
        p.mkdir(parents=True, exist_ok=True)

    def initArgs(self):
        """ @return ArgumentParser """
        parser = ArgumentParser(description=tr("Used for check downloaded papers and whether a paper is downloaded."))

        action = parser.add_mutually_exclusive_group()
        action.add_argument("-c", "--check", help=tr("Check whether the papers is downloaded ."))
        action.add_argument("-l", "--list", action="store_true",
                            help=tr("List existing papers under <{}>.").format(os.getcwd()))
        action.add_argument("-o", "--open", help=tr("Opening papers"))
        action.add_argument("-a", "--add", help=tr("Add a paper that needs to be download"))
        action.add_argument("-L", "--list-tmp", action="store_true", help=tr("elete all tmp record for papers that need to be download"))
        action.add_argument("-D", "--delete-tmp", action="store_true", help=tr("Delete all tmp record for paper already downloaded"))

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
        elif args.add:
            self._add(args.add)
        elif args.list_tmp:
            try:
                with open(".papers/record.txt", "r", encoding="utf-8") as f:
                    for l in f:
                        print(l.strip())
            except:
                pass
        elif args.delete_tmp:
            with open(".papers/record.txt", "w", encoding="utf-8") as f:
                f.write("")
        else:
            # default action
            self._list_papers(os.getcwd())

    def _add(self, a):
        possible = self._find_possible_exist(a)
        confirm = True
        if possible:
            uinput = input(f"Still Add {a}: \n")
            confirm = (uinput.strip().upper() == 'Y')
            print(uinput, confirm)
        if confirm:
            with open(".papers/record.txt", "a+", encoding="utf-8") as f:
                f.write(a + "\n")

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
                # byte = d.encode()
                try:
                    print('- ', d)
                except:
                    print(f'{Fore.YELLOW}- {d.encode()}{Style.RESET_ALL}')
            print('--------------------\n')
        if printable:
            print(f'Summary: {paperCount} papers')

    def _find_possible_exist(self, name: str):
        self._list_papers(os.getcwd(), False)
        name = name.strip()
        found = False
        for pdir, files in self.papers.items():
            for f in files:
                upperF = f.upper()
                if upperF.find(name.upper()) >= 0:
                    print(f"{Fore.YELLOW}[Possible] {f}{Style.RESET_ALL}")
                    found = True
        return found

    def _check_paper(self, name: str, strict=True):
        self._list_papers(os.getcwd(), False)
        name = name.strip()
        if strict:
            for pdir, files in self.papers.items():
                if name in files:
                    print(f"{Fore.GREEN}[Found] {name}{Style.RESET_ALL}")
                    return pdir
        else:
            found = self._find_possible_exist(name)
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
main = common_entry(PaperUtil)