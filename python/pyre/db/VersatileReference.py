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


'''
It is assumed that all tables that can be referred to have
a column "id" and it is the primary key for that table.
Also assumed is that the id column is "varchar" type.

Compared to "Reference", VersatileReference can refer
to records in different tables. The purpose is to
have a "abstract" reference. For example, say we
have an abstract class "shape", and several subclasses
exist: block, cylinder, and sphere. Blocks, cylinders,
and spheres are in three different tables. And in another
table, we should be able to create a versatile reference
called "shape".

'''


def tableRegistry( ):
    return TableRegistry()



class TableRegistry:

    'registry to map table name to table'

    def __init__(self):
        self._store = {}
        return


    def register(self, table):
        self._store[ table.name ] = table
        return


    def get(self, name):
        ret = self._store.get( name )
        if ret is None:
            raise KeyError, "Table %s is not yet registered. Registered tables are %s" %(
                name, self._store.keys() )
        return ret


    def tables(self):
        return self._store.values()


    def itertables(self):
        return self._store.itervalues()



from Column import Column


class VersatileReference(Column):

    def type(self):
        return "character varying (%d)" % self.length


    def __init__(self, name, tableRegistry, length = 1024, **kwds):
        '''a versatile reference column
        
        - name: name of this reference
        '''
        Column.__init__(self, name, **kwds)
        self.length = length
        self.tableRegistry = tableRegistry
        return


    def __get__(self, instance, cls = None):
        ret = Column.__get__(self, instance, cls = cls)
        
        # class variable request
        if ret is self: return self
        
        if not isinstance(ret, reference):
            ret = self._cast( ret )
        return ret


    def _checkReferredTable(self, table):
        try: table.id
        except AttributeError:
            msg = "Table %s does not have a 'id' column. Cannot create reference." % (
                table, )
            raise RuntimeError, msg
        from VarChar import VarChar
        assert isinstance(table.id, VarChar), "'id' column of table %r is not a varchar" %(
            table)
        return


    def _cast(self, value):
        if value is None: return None
        if isinstance( value, reference ):
            return value
        if isinstance( value, basestring ):
            return self._reference_from_str( value )
        if isinstance( value, Table ):
            return self._cast( (value.__class__, value.id) )
        if isinstance( value, tuple):
            if len(value) != 2:
                raise ValueError, "don't know how to cast %s to a reference" % ( value, )
            table, id = value
            if isinstance(table, basestring):
                table = self.tableRegistry.get( table )
            return reference( id, table )
        if not isinstance(value, reference): 
            raise ValueError, "don't know how to cast %s to a reference" % (value,)
        return reference
    

    def _format(self, reference):
        reference = self._cast( reference )
        if reference is None: return ''
        if isinstance( reference, basestring ):
            if len( reference ):
                raise RuntimeError, "If a reference is not an empty string, it should have been casted to a reference"
            return reference
            raise ValueError, "reference is an empty string"
        return str( reference )


    def _reference_from_str(self, value):
        #value is string of a tuple of table name and id in that table
        if not isinstance(value, basestring):
            raise TypeError, "%r is not a string" % value
        if len(value) == 0: return None
        return referenceFromStr( value, self.tableRegistry )
    


from Table import Table
from _reference import reference, fromString as referenceFromStr


# version
__id__ = "$Id$"

# End of file 
