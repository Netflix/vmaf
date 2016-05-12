#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                                  Jiao Lin
#                      California Institute of Technology
#                        (C) 2008  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


class reference:

    separator = '###'

    def __init__(self, id, table):
        self.table = table
        self.id = id
        return


    def dereference(self, db):
        id = self.id
        if id is None: return None
        table = self.table
        all = db.fetchall( table, where = "id='%s'" % id )
        n = len(all)
        if n == 0:
            raise RuntimeError, "Cannot get record of id %r of table %r" % (
                id, table )
        elif n > 1:
            raise RuntimeError, "There should be only 1 record of id %r of table %r. Got %d" % (
                id, table, n )
        return all[0]


    def __str__(self):
        return '%s%s%s' % (self.table.name, self.separator, self.id)
    


def fromString( s, tableRegistry ):
    tablename, id = s.split( reference.separator )
    return reference(id, tableRegistry.get( tablename ) )


# version
__id__ = "$Id$"

# End of file 
