"""
Used as common Frame for myscripts
"""
class Frame(object):
    def __init__(self):
        self.config = {}

    def readConfig(self, file):
        pass

    def initArgs(self):
        """ @return ArgumentParser """
        pass

    def parseArgs(self, argv):
        # args = self._initArgs().parse_args()
        pass
import sys
def common_entry(frame):
    def entry_main(argv = None):
        f = frame()
        f.initArgs()
        if argv is None: argv = sys.argv[1:]
        f.parseArgs(argv)
    return entry_main