
import platform
import os
import atexit
from ._constants import _LOG_DIR
import logging

def createLogger(name: str, savefile = True, stream = False, 
    level = logging.INFO, baseDir = _LOG_DIR):
    """create logger, specify name, for example: test.log
    suffix is not necessary but helpful
    """
    
    log_file = baseDir / name

    logger = logging.getLogger(name)
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
    return logger


logger = logging.getLogger('bffacilities')

def lockfile(fileName, start = None, stop = None):
    """
    original : Fix Multiple instances of scheduler problem  https://github.com/viniciuschiele/flask-apscheduler/issues/51
    :param str filename: specified lock filename, such as app.lock
    :param function start: callback for app loop start
    :param function stop: callback for app loop stop
    """
    if platform.system() != 'Windows':
        fcntl = __import__("fcntl")
        f = open(fileName, "wb")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if start is not None:
                start()
            # logger.debug("Scheduler Started...")
        except Exception as e:
            logger.error('Exit the scheduler, error - {}'.format(e))
            if stop is not None:
                stop()
        def unlock():
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()
        atexit.register(unlock)
    else:
        msvcrt = __import__("msvcrt")
        f = open(fileName, "wb")
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            if start is not None:
                start()
            # logger.debug("Scheduler Started...")
        except Exception as e:
            logger.error('Exit the scheduler, error - {}'.format(e))
            if stop is not None:
                stop()
        def _unlock_file():
            try:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                f.close()
                os.remove(f.name)
            except IOError:
                raise
        atexit.register(_unlock_file)
