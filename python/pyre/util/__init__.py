#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


def tmp():
    from tmpdir import tmp
    return tmp()


def spawn(onParent, onChild):
    from subprocesses import spawn
    return spawn(onParent, onChild)


def spawn_pty(onParent, onChild):
    from subprocesses import spawn_pty
    return spawn_pty(onParent, onChild)


def expandMacros(raw, substitutions):
    from expand import expandMacros
    return expandMacros(raw, substitutions)


# version
__id__ = "$Id: __init__.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#  End of file 
