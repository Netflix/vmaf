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
'''

from Column import Column


class Reference(Column):


    def type(self):
        return "character varying (%d)" % self.length 


    def __init__(self, name, table, default="", **kwds):
        '''a reference column
        
        - name: name of this reference
        - table: the table this reference refers to
        '''
        self._checkReferredTable(table)
        self.length = table.id.length
        
        length = kwds.get('length')
        if length: raise ValueError, "'length' is not a valid keyword for 'Reference'"
        
        Column.__init__(self, name, default, **kwds)

        self.referred_table = table
        return


    def __get__(self, instance, cls = None):
        ret = Column.__get__(self, instance, cls = cls)
        
        # class variable request
        if ret is self: return self

        return self._cast( ret )


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
        if value is None or value == '': return None
        if isinstance( value, reference ): return value
        if isinstance( value, self.referred_table ):
            return reference( value.id, self.referred_table )
        #value is an id that refers to an item in a table
        ret = reference( value, self.referred_table )
        return ret


    def _format(self, value):
        reference = self._cast( value )
        if reference is None: return ''
        return reference.id



from _reference import reference


# version
__id__ = "$Id$"

# End of file 
