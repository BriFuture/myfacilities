import sys
from ._util import initGetText
from . import __version__

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
    availableCmds = ['gitrepo', 'monitor', 'broadcast', 'paperutil']
    if cmd in availableCmds:
        arg = sys.argv[1:]
    #  add custom sub-commands here
    if cmd == 'gitrepo':
        from .gitrepo import main as gitrepo
        gitrepo(arg)
    elif cmd == 'monitor':
        from .monitor import main as monitor
        monitor(arg)
    elif cmd == 'broadcast':
        from .broadcast import main as broadcast
        broadcast()
    elif cmd == 'paperutil':
        from .paperUtil import main as paperUtil
        paperUtil(arg)
    else:
        print('Available sub commands: ')
        cmds = ""
        for i, c in enumerate(availableCmds):
            cmds += f"{i+1}: {c}  \t"
        print(cmds, "\n")
        parser.print_usage()
        sys.exit(0)