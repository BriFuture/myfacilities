#!/usr/env/bin python3
# -*- coding: utf-8 -*-

"""Use as script entry
"""

import sys, os
import os.path as osp
from ._constants import BFF_ROOT_PATH as RootPath

from pathlib import Path
from .utils import initGetText
from . import __version__
from ._plugin import load_scripts
import json

#  add custom sub-commands in this file
with open(osp.join(RootPath, "myscripts/meta.json")) as f:
    availableCmds = json.load(f)

availableCmds["tray"] = "tray"

tr = initGetText("bffacility")
def onError(parser):
    print(tr('Available sub commands: '))
    cmds = ""
    for i, c in enumerate(availableCmds):
        cmds += f"{i+1}: {c}  \t"
    print(cmds, "\n")
    parser.print_usage()

def loadPriScript(cmd, arg):
    """arg contain subscripts:
    for example: `bff pri test -h`
    arg should be `test -h`
    """
    pripath = Path(RootPath) / "../_pri"
    pripath = pripath.resolve()
    if len(arg) < 1:
        print("Not Supported!")
        return
    try:
        subscript = arg[0]
        p, ss = osp.split(subscript)
        pripath = Path(osp.join(pripath, p))
        if ss.endswith(".py"):
            ss = ss[:-3]
        print(ss, pripath)
        arg = arg[1:]
        load_scripts(ss, arg, pripath)
    except Exception as e:
        print("Running: ", e) 

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser(prog='bffacility', 
        description="Usage: bffacility <subcommand> [args]")
    parser.add_argument('subcmd', type=str, nargs="?", help=tr('type sub-command to exec certain function'))
    parser.add_argument('-V', action="version", help=tr(f'show version: {__version__}'), version=f'%(prog)s {__version__}')

    sys.argv.pop(0)  # remove script name
    args = vars(parser.parse_args(sys.argv[:1]))
    cmd = args["subcmd"]
    
    arg = sys.argv[1:]

    if cmd == 'tray':
        from .win_tray import main as tray
        tray(arg)
        return
    elif cmd == 'pri':
        loadPriScript(cmd, arg)
        return
    elif cmd in availableCmds:
        load_scripts(availableCmds[cmd], arg)
        return

    onError(parser)
