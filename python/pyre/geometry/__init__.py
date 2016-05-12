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

# modellers
def loader():
    from Loader import Loader
    return Loader()


def modeller():
    from GeometricalModeller import GeometricalModeller
    return GeometricalModeller()

# mesh
def mesh(dim, order):
    from Mesh import Mesh
    return Mesh(dim, order)

# persistence
def renderer(format=None):
    if format is None:
        format = "pml"
        
    if format == "pml":
        from pml.Renderer import Renderer
        return Renderer()

    import journal
    journal.error.log("'%s': unknown geometry rendering format" % format)
    return None
    

def parser(format=None):
    if format is None:
        format = "pml"
        
    if format == "pml":
        from pml.Parser import Parser
        return Parser()

    import journal
    journal.error.log("'%s': unknown geometry parsing format" % format)
    return None
    

# version
__id__ = "$Id: __init__.py,v 1.1.1.1 2006-11-27 00:09:56 aivazis Exp $"

# End of file
