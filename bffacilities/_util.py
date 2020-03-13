
from pathlib import Path
from ._constants import _LOG_DIR
import logging

## For Translating

LANGUAGE_DIR = (Path(__file__).parent / "locale").resolve()
import gettext
def initGetText(domain="myfacilities") -> gettext.gettext:
    gettext.bindtextdomain(domain, LANGUAGE_DIR)
    gettext.textdomain(domain)
    gettext.find(domain, "locale", languages=["zh_CN", "en_US"])
    return gettext.gettext

def createLogger(name: str, savefile = True, stream = False, 
    level = logging.INFO, baseDir = _LOG_DIR):
    """create logger, specify name, for example: test.log
    suffix is not necessary but helpful
    """

    log_file = baseDir / name

    logger = logging.getLogger()
    logger.setLevel(level)
    if savefile:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s:  %(message)s"))
        fh.setLevel(level)
        logger.addHandler(fh)

    if stream:
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:  %(message)s"))
        logger.addHandler(sh)
        # print(f"Stream Logger for {name}")

    return logger