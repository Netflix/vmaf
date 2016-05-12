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


def locate(filename, pathlist=None, extensions=None):

    if pathlist is None:
        pathlist = ["", "."]

    if extensions is None:
        extensions = []

    import os
    path, base = os.path.split(filename) 
    name, ext = os.path.splitext(base)

    guesses = [filename]

    if not ext:
        for suffix in extensions:
            guesses.append(filename + suffix)

    import os
    for guess in guesses:
        for path in pathlist:
            candidate = os.path.join(path, guess)
            if os.path.isfile(candidate):
                return candidate

    raise IOError("Could not locate %r in %r" % (filename, pathlist))


# version
__id__ = "$Id: locate.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

#  End of file 
