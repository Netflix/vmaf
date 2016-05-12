#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


def spawn(onParent, onChild):
    import os

    pid = os.fork()
    if pid > 0:
        return onParent(pid)

    import os
    pid = os.getpid()

    return onChild(pid)


def spawn_pty(onParent, onChild):
    import pty

    pid, fd = pty.fork()
    if pid > 0:
        return onParent(pid, fd)

    import os
    pid = os.getpid()

    return onChild(pid)


# version
__id__ = "$Id: subprocesses.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

# End of file 
