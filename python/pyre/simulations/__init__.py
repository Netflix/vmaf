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


def controller(name=None):
    from SimulationController import SimulationController
    return SimulationController(name)



def simpleSolver(name=None):
    from SimpleSolver import SimpleSolver
    return SimpleSolver(name)


# version
__id__ = "$Id: __init__.py,v 1.1.1.1 2006-11-27 00:10:06 aivazis Exp $"

# End of file 
