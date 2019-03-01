#!/usr/bin/python3
# -*- coding: utf-8 -*-

from . import createLogger, _CONFIG_DIR
from pathlib import Path

DEFAULT_REPO_DIR = "/src"

logger = createLogger("gitrepo.log", stream=True)
config = {
    "repo_dir": DEFAULT_REPO_DIR,
    "recommended": False,
    "user": "git",
    "group": "git",
}
config_file = _CONFIG_DIR / "gitrepo.yaml"

import yaml
def readConfig():
    global config
    if not config_file.exists():
        with config_file.open("w") as f:
            yaml.dump(config, f)
        return
    
    with config_file.open() as f:
        config = yaml.load(f.read())
    
    if "repo_dir" not in config:
        config["repo_dir"] = DEFAULT_REPO_DIR
    
    if "recommended" not in config:
        config["recommended"] = False



import argparse 
def parseArgs():
    readConfig()
    parser = argparse.ArgumentParser(description="Create or remove git repository \
        under direcotry: {}. Currently only works for Linux. \
        Most operations are only tested under Linux".format(config["repo_dir"]))

    parser.add_argument('repo', type=str, help="Specify the repository name, for example: \
        Test.git, if .git suffix is not typed, this script will add it automatically")

    action = parser.add_mutually_exclusive_group()
    action.add_argument("-i", "--init", 
        help="Default action, Init a new git repository under {}.".format(config["repo_dir"]),
        action="store_true")

    action.add_argument("-d", "--delete", 
        help="Remove a existing repository under {}.".format(config["repo_dir"]),
        action="store_true")

    parser.add_argument("-s", "--src-dir", help="Specify the src folder for git repository, for example, the default path is `/src`.\
        Do NOT put single-slash `/` as the src folder, it will make unestimated error.")
    parser.add_argument("-u", "--user", help="Specify the git repository owner, for example, the default owner is `git`")
    parser.add_argument("-g", "--group", help="Specify the git repository owner group, for example, the default owner group is `git`")

    args = parser.parse_args()
    if args.src_dir is not None and args.src_dir != "/":
        config["src"] = args.src_dir

    config["action"] = "init"
    if args.delete:
        config["action"] = "delete"

    repo = args.repo
    if repo[-4:] != ".git":
        repo = "{}.git".format(repo)

    config["repo"] = repo

    return parser

import os, shutil
def createRepo(repo: Path):
    if repo.exists():
        logger.warning("Repository: {} exists, if you want to recreate a new repository, please delete it first.\
            Now change the owner of that repository".format(repo))
        try:
            shutil.chown(repo, config["user"], config["group"])
        except Exception as e:
            logger.critical("You may need to run this script under super user privilege.")
            logger.critical(e)
        logger.info("Repository owner changed.")
        return

    logger.info("Creating an repository: {}".format(repo))
    repo.mkdir(parents=True)
    import subprocess
    try:
        subprocess.run(["sudo", "-s", "git", "init", "--bare", "--shared", str(repo.absolute())])
        shutil.chown(repo, config["user"], config["group"])
    except Exception as e:
        logger.critical("Git may not be installed, please install git first, \
            or you are not running this command under super user privilege. You may need to run this script under super user privilege.\
            \n{}".format(fnf))
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
    except Exception as e:
        logger.warning(e)
    logger.info("Repository deleted.")


def main():
    # if os.pl
    parser = parseArgs()
    
    repo = Path(config["repo_dir"]) / config["repo"]
    try:
        if config["action"] == "delete":
            deleteRepo(repo)
        else:
            createRepo(repo)
    except:
            # for fisrt using this script
        if config["recommended"] is False:
            parser.print_help()
            with config_file.open("w") as f:
                config["recommended"] = True
                yaml.dump(config, f)
        
    

if __name__ == "__main__":
    main()