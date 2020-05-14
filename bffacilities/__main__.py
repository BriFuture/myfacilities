#!/usr/env/bin python3
# -*- coding: utf-8 -*-

"""Use as script entry
"""

import sys
from .utils import initGetText
from . import __version__

availableCmds = ['gitrepo', 'monitor', 'broadcast', 'paperutil', 'tray']

def main():
    tr = initGetText("bffacility")
    from argparse import ArgumentParser
    parser = ArgumentParser(prog='bffacility', 
        description="Usage: bffacility <subcommand> [args]")
    parser.add_argument('subcmd', type=str, help=tr('type sub-command to exec certain function'))
    parser.add_argument('-V', action="version", help=tr(f'show version: {__version__}'), version=f'%(prog)s {__version__}')

    args = parser.parse_args(sys.argv[1:2])
    cmd = args.subcmd

    sys.argv.pop(0)
    
    if cmd in availableCmds:
        arg = sys.argv[1:]
    #  add custom sub-commands here
    if cmd == 'gitrepo':
        from .myscripts.gitrepo import main as gitrepo
        gitrepo(arg)
    elif cmd == 'monitor':
        from .myscripts.monitor import main as monitor
        monitor(arg)
    elif cmd == 'broadcast':
        from .myscripts.broadcast import main as broadcast
        broadcast(arg)
    elif cmd == 'paperutil':
        from .myscripts.paperUtil import main as paperUtil
        paperUtil(arg)
    elif cmd == 'tray':
        print('test')
        from .win_tray import main as tray
        tray(arg)
    else:
        print('Available sub commands: ')
        cmds = ""
        for i, c in enumerate(availableCmds):
            cmds += f"{i+1}: {c}  \t"
        print(cmds, "\n")
        parser.print_usage()
        sys.exit(0)
