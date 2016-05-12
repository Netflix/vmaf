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


class Mesh(object):


    def handle(self):
        return self._mesh


    def statistics(self):
        import pyre._pyre
        return pyre._pyre.statistics(self._mesh)


    def vertex(self, vertexid):
        import pyre._pyre
        return pyre._pyre.vertex(self._mesh, vertexid)


    def simplex(self, simplexid):
        import pyre._pyre
        return pyre._pyre.simplex(self._mesh, simplexid)


    def vertices(self):
        import pyre._pyre
        dim, order, vertices, simplices = pyre._pyre.statistics(self._mesh)

        for i in range(vertices):
            yield pyre._pyre.vertex(self._mesh, i)

        return


    def simplices(self):
        import pyre._pyre
        dim, order, vertices, simplices = pyre._pyre.statistics(self._mesh)

        for i in range(simplices):
            yield pyre._pyre.simplex(self._mesh, i)

        return


    def __init__(self, dim, order):
        self.dim = dim
        self.order = order
        
        try:
            import pyre._pyre
        except ImportError:
            import journal
            error = journal.error('pyre')
            error.line("unable to import the C++ pyre extensions")
            error.log("mesh objects are not supported")
            self._mesh = None
            return
                                      
        self._mesh = pyre._pyre.createMesh(dim, order)

        return

# version
__id__ = "$Id: Mesh.py,v 1.1.1.1 2006-11-27 00:09:56 aivazis Exp $"

# End of file 
