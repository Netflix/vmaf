__copyright__ = "Copyright 2016, Netflix, Inc."
__license__ = "Apache, Version 2.0"

import os
import sys

def print_usage():
    print "usage: " + os.path.basename(sys.argv[0]) + \
        " d"

if __name__ == '__main__':

    if len(sys.argv):
        pass