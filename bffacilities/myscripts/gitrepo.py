#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""A command which may simplify the creation or deletion a shared git repository on personal mini git server.

"""

__version__ = "0.0.18"

import shutil
import os
import sys
from pathlib import Path
from argparse import ArgumentParser
import json
from bffacilities._frame import Frame
from bffacilities.utils import createLogger, initGetText
from bffacilities._constants import _CONFIG_DIR

tr = initGetText("gitrepo")
logger = createLogger("gitrepo.log", stream=True)

DEFAULT_REPO_DIR = "/src"


class GitRepoFrame(Frame):
    ConfigFile = _CONFIG_DIR / "gitrepo.json"

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = {
            "repo_dir": DEFAULT_REPO_DIR,
            "recommended": False,
            "user": "git",
            "group": "git",
        }
    
    def _initConfigFile(self, file):
        if not file.exists():
            with file.open("w") as f:
                json.dump(self.config, f)
            return
    def readConfig(self, file):
        self._initConfigFile(file)
        try:
            with file.open() as f:
                self.config = json.loads(f.read())
        except:
            self._initConfigFile(file)

        if "repo_dir" not in self.config:
            self.config["repo_dir"] = DEFAULT_REPO_DIR

        if "recommended" not in self.config:
            self.config["recommended"] = False

    def initArgs(self):
        self.readConfig(GitRepoFrame.ConfigFile)
        parser = ArgumentParser(prog='bffacility gitrepo', description=tr("""Run this script under SUPER USER privilege. Create or remove git repository
    under direcotry: {}. Currently only works for Linux. Most operations are only tested under Linux.
    Script version : {}.
        """).format(self.config["repo_dir"], __version__))

        parser.add_argument('repo', nargs="?", type=str, help=tr("Specify the repository name, for example: \
            Test.git, if .git suffix is not typed, this script will add it automatically"))

        action = parser.add_mutually_exclusive_group()
        action.add_argument("-i", "--init",
                            help=tr("Default action, Init a new git repository under {}.").format(
                                self.config["repo_dir"]),
                            action="store_true")

        action.add_argument("-d", "--delete",
                            help=tr("Remove a existing repository under {}.").format(
                                self.config["repo_dir"]),
                            action="store_true")

        action.add_argument("-l", "--list",
                            help=tr("List existing repositories under {}.").format(
                                self.config["repo_dir"]),
                            action="store_true")

        parser.add_argument("-s", "--src-dir", help=tr("Specify the src folder for git repository, for example, the default path is `/src`.\
            Do NOT put single-slash `/` as the src folder, it will make unestimated error."))
        parser.add_argument("-u", "--user", help=tr(
            "Specify the git repository owner, for example, the default owner is `git`."))
        parser.add_argument("-g", "--group", help=tr(
            "Specify the git repository owner group, for example, the default owner group is `git`."))
        parser.add_argument("-w", "--windows", help=tr(
            "Force this script running on Windows platform."), action="store_true")
        parser.add_argument("-f", "--force", help=tr(
            "Force this script running even without SUPER USER previlige."), action="store_true")
        parser.add_argument( "-V", "--version", 
            help=tr("Print this script version"), action="version", version=f'gitrepo {__version__}')
        self.parser = parser


    def parseArgs(self, argv):
        args = self.parser.parse_args(argv)
        config = self.config
        config["action"] = "init"
        if args.delete:
            config["action"] = "delete"
        elif args.list:
            config["action"] = "list"

        config["windows"] = args.windows
        config["normal_user"] = args.force

        if config["action"] != "list":
            if not args.repo:
                logger.warning(
                    "Repository name should be specified. Or -l (--list) options should be specified. \n------------------------")
                self.parser.print_help()
                sys.exit(1)
        else:
            config["windows"] = True
            config["normal_user"] = True

        if args.src_dir is not None and args.src_dir != "/":
            config["src"] = args.src_dir

        repo = args.repo or "Test"
        if repo[-4:] != ".git":
            repo = "{}.git".format(repo)

        config["repo"] = repo

    def run(self):
        config = self.config
        repo = Path(config["repo_dir"]) / config["repo"]
        try:
            if config["action"] == "delete":
                deleteRepo(repo)
            elif config["action"] == "init":
                createRepo(repo)
            else:
                listRepo(repo)
        except Exception as e:
            logger.warning(e)

def createRepo(repo: Path):
    if not repo.exists():
        return
    logger.warning("Repository: {} exists, if you want to recreate a new repository, please delete it first.\
        Now change the owner of that repository".format(repo))
    try:
        shutil.chown(repo, config["user"], config["group"])
        logger.info("Repository owner changed.")
    except Exception as e:
        logger.critical(
            "You may need to run this script under super user privilege.")
        logger.critical(e)
        

    logger.info("Creating an repository: {}".format(repo))
    repo.mkdir(parents=True)
    import subprocess
    try:
        subprocess.run(
            ["git", "init", "--bare", "--shared", str(repo.absolute())])
        shutil.chown(repo, config["user"], config["group"])
    except Exception as e:
        logger.critical("Git may not be installed, please install git first, \
            or you are not running this command under super user privilege. You may need to run this script under super user privilege.")
        logger.critical(e)

    logger.info("Repository created.")


def deleteRepo(repo: Path):
    if not repo.exists():
        logger.warning("Repository: {} not exists".format(repo))
        return
    logger.info("Deleteing an existing repository: {}".format(repo))
    try:
        # repo.rmdir()
        shutil.rmtree(str(repo))
        logger.info("Repository deleted.")
    except Exception as e:
        logger.warning(e)


def listRepo(repo: Path):
    repo_dir = repo.parent
    repos = []
    for r in repo_dir.iterdir():
        if r.suffix == ".git" and r.is_dir():
            head = r / "HEAD"
            if head.exists():
                repos.append(r)
    logger.info("Repositories found count: {}".format(len(repos)))
    for re in repos:
        logger.info("Found a repository: {}".format(re.stem))


def is_root():
    if hasattr(os, "getuid"):
        return os.getuid() == 0
    return False


def main(argv = None):
    gp = GitRepoFrame()
    gp.initArgs()
    if argv is None:
        argv = sys.argv[1:]
    gp.parseArgs(argv)
    if not sys.platform.startswith('linux') and not config["windows"]:
        logger.warning("Sorry, this script can only run on linux systems.")
        return

    if not is_root() and not config["normal_user"]:
        logger.warning(
            "Try running this script as normal user.\n---------------")
        parser.print_help()
        return
    gp.run()


def first_taste():
    # for fisrt using this script
    if config["recommended"] is False:
        parser.print_help()
        with GitRepoFrame.ConfigFile.open("w") as f:
            config["recommended"] = True
            json.dump(config, f)


if __name__ == "__main__":
    main()
