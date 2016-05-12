#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


import fcntl


class FileLockingPosix(object):


    LOCK_EX = fcntl.LOCK_EX
    LOCK_SH = fcntl.LOCK_SH
    LOCK_NB = fcntl.LOCK_NB
    LOCK_UN = fcntl.LOCK_UN


    def lock(self, stream, flags=LOCK_SH):
        return fcntl.flock(stream.fileno(), flags)


    def unlock(self, stream):
        return fcntl.flock(stream.fileno(), self.LOCK_UN)
    

# version
__id__ = "$Id: FileLockingPosix.py,v 1.1.1.1 2006-11-27 00:10:05 aivazis Exp $"

# End of file 
