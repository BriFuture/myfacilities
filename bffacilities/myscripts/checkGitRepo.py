import json
from bffacilities._constants import _CONFIG_DIR
from bffacilities import createLogger
import os
import os.path as osp

logger = createLogger("checkGitRepo", savefile=False, stream=True)

os.makedirs(_CONFIG_DIR, exist_ok=True)
configFile = _CONFIG_DIR / "checkGitRepo.json"

def addDirIntoMonitor(dirs):
    """dirs: array(abspath)
    """
    content = None
    try :
        with open(configFile, "r") as f:
            fileContent = f.read()
            if len(fileContent) > 0:
                content = json.loads(fileContent)
    except Exception as e:
        logger.error(f"error reading configFile {configFile}: {e}")

    if content is None:
        content = []
    added = 0
    for d in dirs:
        if d in content:
            logger.warning(f"Directory {d} already added.")
            continue
        added += 1
        logger.info(f"Directory {d} will be added.")
        content.append(d)

    if added:
        try:
            with open(configFile, "w") as f:
                json.dump(content, f)
        except Exception as e:
            logger.error(f"error writing configFile {configFile}: {e}")
def delDirFromMonitor(dirs):
    content = None
    try :
        with open(configFile, "r") as f:
            content = json.load(f)
    except Exception as e:
        logger.error(f"error reading configFile {configFile}: {e}")

    if content is None:
        logger.warning(f"Directory does not exist in empty monitor list.")
        return
    deled = 0
    for d in dirs:
        if d not in content:
            logger.warning(f"Directory does not exist in empty monitor list.")
            continue
        logger.info(f"directory {d} will be removed.")
        deled += 1
        content.remove(d)
    with open(configFile, "w") as f:
        json.dump(content, f)

import subprocess as sp
import io
def executeGit(cmd):
    content = None
    try :
        with open(configFile, "r") as f:
            content = json.load(f)
    except Exception as e:
        logger.error(f"Error reading configFile {configFile}: {e}")
    if content is None:
        logger.error(f"No directory added in monitor list, please add them first.")
        return
    assert type(content) is list, "Error content"
    git = ["git"]
    git.extend(cmd)

    # d = content[0]
    processes = []
    for d in content:
        if not osp.exists(d):
            logger.warning(f"Directory {d} already deleted, consider to remove it from monitor list")
            continue
        logger.info(f"================ Entering directory {d} ===============\n")
        p = sp.Popen(git, cwd=d)
        p.wait()
        print("\n")
        # processes.append((d, p))
        # print(p.stdout.read())
    # strIOs = []
    # for d, p in processes:
    #     p.wait()
    #     strIO = io.TextIOWrapper(io.BytesIO(p.stdout.read()))
    #     strIOs.append((d, strIO))
    # for d, strIO in strIOs:
    #     logger.info(f"================ Entering directory {d} ===============\n")
    #     print(strIO.read())
def validateDirs(directories):
    dirs = []
    for d in directories:
        d = osp.abspath(d)
        if not osp.exists(osp.join(d, ".git")):
            logger.warning(f"Directory {d} may not be git repository.")
            continue
        if osp.exists(d):
            dirs.append(d)
    return dirs
def main(args = None):
    from argparse import ArgumentParser
    parser = ArgumentParser(prog="checkGitRepo", description="Check git repo and do some actions")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-a", "--add-dir", type=str, nargs="+", help="The directory will be added into program monitor list")
    group.add_argument("-d", "--del-dir", type=str, nargs="+", help="Delete directory from monitor list")
    group.add_argument("-e", "--execute", type=str, nargs="+", help="execute git command, for example: checkGitRepo -e 'reset -- .' will call 'git reset -- .'")
    group.add_argument("-s", "--status", action="store_true", help="equvalent of 'git status'")
    group.add_argument("-o", "--open", action="store_true", help="Open config file")

    args = vars(parser.parse_args(args))

    if args["add_dir"]:
        directories = args["add_dir"]
        dirs = validateDirs(directories)
        addDirIntoMonitor(dirs)
            # print(d)
    elif args["del_dir"]:
        directories = args["del_dir"]
        dirs = []
        for d in directories:
            dirs.append(osp.abspath(d))
        delDirFromMonitor(dirs)
    elif args["status"]:
        executeGit(["status"])
    elif args["execute"]:
        executeGit(args["execute"])
    elif args["open"]:
        import webbrowser
        webbrowser.open(configFile)
    else:
        parser.print_help()

