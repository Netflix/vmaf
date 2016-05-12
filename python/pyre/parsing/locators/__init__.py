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


def simple(tag):
    from SimpleLocator import SimpleLocator
    return SimpleLocator(tag)


def script(source, line, function):
    from ScriptLocator import ScriptLocator
    return ScriptLocator(source, line, function)


def file(source, line=-1, column=-1):
    if line == -1 and column == -1:
        from SimpleFileLocator import SimpleFileLocator
        return SimpleFileLocator(source)
    
    from FileLocator import FileLocator
    return FileLocator(source, line, column)


def chain(this, next):
    from ChainLocator import ChainLocator
    return ChainLocator(this, next)


# version
__id__ = "$Id: __init__.py,v 1.1.1.1 2006-11-27 00:10:05 aivazis Exp $"

# End of file 
