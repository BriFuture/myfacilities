#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Description: Monitor folder's change and restart predefined Program, Original File is copied from Liaoxuefei's tutorial.
Monitor file changes, and execute prepared commands.
I made some modifications and optimizations. Now the changed filename will be send to the program and you can use {name} within
cmd arguments to get the name.

Author: BriFuture
"""

__author__ = 'BriFuture'
__version__ = '0.2.1'

import os
import sys

from ._util import createLogger, initGetText

tr = initGetText("monitor")
logger = createLogger("monitor.log", stream=True)


from pathlib import Path
import json
from argparse import ArgumentParser
from ._frame import Frame
class Configuration(Frame):
    _DEFAULT_LOC = Path(os.getcwd()) / ".bfc" 
    _DEFAULT_FILE = Path(os.getcwd()) / ".bfc" / "monitor.json"

    def __init__(self):
        self.config = {
            # command name, str
            "cmd": None,
            # command arguments,  []
            "cmd_args": None,
            # monitor directory, "."
            "mon_dir": ".",

            "recursive": True,
            # monitor file extensions, for example: [".py", ".js"]
            "mon_ext": None,
            # exclusion file names, file extension needed, for example "test.py"
            "exclude": None
        }
        self._addArgs()

    def readConfig(self, file: Path):
        if not file or not file.exists():
            return
        with file.open("r", encoding="utf-8") as f:
            self.config = json.loads(f.read())
            # wont update cmdline args
            # for k, v in config.items():
            #     if k not in self.config:
            #         self.config[k] = v

    def _addArgs(self):
        parser = ArgumentParser(description=tr("Monitor file changes, and execute prepared commands."))

        self.parser = parser
        argument = parser.add_mutually_exclusive_group(required=False)
        argument.add_argument("-c", "--cmd", help=tr("""Specify cmd to execute after file events triggered. If the command have its options,
            for example 'python -V', use quote \" to wrap the whole command. It's recommend on parsing arguments in this way, so you can use {{name}}
            to identify which file has been changed"""))
        argument.add_argument("-C", "--config", help=tr("Read from config file, only file name is needed, for example, if you type '--config test', then this script will find a file named test.json which locates under {}. Note: The contents in the file will override other options given as command arguments.")
            .format(Configuration._DEFAULT_FILE))

        parser.add_argument("-a", "--argument", nargs="*", help=tr("Append command arguments for specified CMD. These arguments should not start with '-'"))
        parser.add_argument("-d", "--directory", help=tr("The directory to monitor, . by default."))
        parser.add_argument("-r", "--recursive", help=tr("Recusively monitor the diretory and sub directories"), action="store_true")
        parser.add_argument("-e", "--extension", nargs="*", type=str, help=tr("Specify the file suffix that needs to monitor, .py extension by default"))
        parser.add_argument("-E", "--exclude", nargs="*", type=str, help=tr("File that should be excluded when events triggerd"))
        parser.add_argument("-s", "--save-config", help=tr("Save the cofiguration into a file, the default file is {}").format(self._DEFAULT_FILE), nargs="?")
        parser.add_argument("-S", "--start-at-first", help=tr("Start the process or command when this scripts running."), action="store_true")
        parser.add_argument("--version", help=tr("Print version and exit."), action="version", version="Monitor's version is {}".format(__version__))


    def parseArgs(self, argv):
        args = self.parser.parse_args(argv)

        if args.config is None and args.cmd is None:
            logger.critical("Neither a config file is specified, nor cmd is given.")
            sys.exit(1)

        if args.config:
            self.readConfig(args.config)
            return

        if args.cmd is not None and " " in args.cmd:
            self.config["cmd"] = args.cmd.split(" ")
        else:
            self.config["cmd"] = [args.cmd]
        self.config["cmd_args"] = args.argument or []
        # , default is [".py"]
        self.config["mon_ext"] = args.extension or [".py"]
        mon_dir = args.directory or os.getcwd()
        self.config["mon_dir"] = str(Path(mon_dir).resolve())
        self.config["exclude"] = args.exclude or []
        self.config["recursive"] = args.recursive or True
        self.config["start"] = args.start_at_first

        if args.save_config:
            sc = self._DEFAULT_FILE / "monitor_{}.json".format(args.save_config)
        else:
            sc = self._DEFAULT_FILE 

        with sc.open("w", encoding="utf-8") as f:
            # mon_dir = self.config["mon_dir"]
            # self.config["mon_dir"] = str(mon_dir)
            json.dump(self.config, f)
            # self.config["mon_dir"] = mon_dir
Configuration._DEFAULT_LOC.mkdir(parents=True, exist_ok=True)

import time
from watchdog.events import FileSystemEventHandler

class MyFileSystemEventHander(FileSystemEventHandler):

    def __init__(self, fn, config: Configuration):
        super(MyFileSystemEventHander, self).__init__()
        self.restart = fn
        self.config = config
        self.last = time.time()

    def on_any_event(self, event):
        # for debounce 
        cur = time.time()
        if cur - self.last < 0.25:
            return
        self.last = cur
        ext_able = False
        src = Path(event.src_path)
        if src.name  not in self.config["exclude"]:
            for ext in self.config["mon_ext"]:
                if src.suffix == ext:
                    ext_able = True
                    break

        if ext_able:
            logger.info('File changed: {}'.format(src))
            self.restart(name=src.name)


from watchdog.observers import Observer
import subprocess

class NewProcess(object):

    def __init__(self, config: dict):
        self.process = None
        self.config = config
        # self.command = self.config["cmd_args"][:]
        # self.command.insert(0, self.config["cmd"])
        # self.command[0:0] = self.config["cmd"]
        command = " ".join(self.config["cmd"])
        if len(self.config["cmd_args"]) > 0:
            self.args = "{} {}".format(command, ' '.join(self.config["cmd_args"]))
        else:
            self.args = command
        # self.args = ' '.join(self.command)

    def start(self, name=""):
        args = self.args.replace("{{name}}", name)
        logger.info('[Start process] {} ...'.format(args))
        args = args.split(" ")
        self.process = subprocess.Popen(
            args, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)

    def stop(self):
        if self.process:
            logger.info('[Kill process] [{}]...'.format(self.process.pid))

            self.process.kill()
            self.process.wait()
            logger.info('[Process ended] code {}.'.format(
                self.process.returncode))
            self.process = None

    def restart(self, name=None):
        self.stop()
        self.start(name=name)

    def start_watch(self):
        observer = Observer()
        observer.schedule(MyFileSystemEventHander(self.restart, self.config), 
            path=self.config["mon_dir"], 
            recursive=self.config["recursive"]
        )
        observer.start()
        logger.info('Watching directory {}...'.format(self.config["mon_dir"]))
        if(self.config["start"]):
            self.start()
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()


np = None
def main(argv = None):
    global np
    conf = Configuration()
    if argv is None:
        argv = sys.argv[1:]
    conf.parseArgs(argv)
    
    np = NewProcess(conf.config)
    try:
        np.start_watch()
    except KeyboardInterrupt:
        print("Keyboard Interrupt, process killed")
        np.stop()
    finally:
        np.stop()


import sys
# 退出该程序时，关闭子进程
def beforeExit(signum, frame):
    print("Exit")
    global np
    np.stop()
    sys.exit()

if __name__ == '__main__':
    # import signal
    # signal.signal(signal.SIGINT, beforeExit)
    main()
    