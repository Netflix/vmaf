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


import journal
debug = journal.debug('db' )


class DBManager(object):


    def autocommit(self, flag=True):
        return self.db.autocommit(flag)


    def commit(self):
        return self.db.commit()


    def connect(self, **kwds):
        raise NotImplementedError("class %r must override 'connect'" % self.__class__.__name__)


    def cursor(self):
        return self.db.cursor()


    def insertRow(self, row, tableName=None):
        # build the arguments
        if tableName is None:
            name = row.name
        else:
            name = tableName
            
        values = row.getFormattedWriteableValues()
        columns = row.getWriteableColumnNames()

        # build the sql query
        sql = "INSERT INTO %s (\n    %s\n    ) VALUES (\n    %s\n    )" % (
            name, ", ".join(columns), ", ".join(["%s"] * len(columns)))

        # execute the sql statement
        c = self.db.cursor()
        try:
            c.execute(sql, values)
        except:
            debug.log( 'sql: %s, values: %s' % (sql, values) )
            raise
        self.db.commit()

        return


    def updateRow(self, table, assignments, where=None):

        columns = []
        values = []
        
        row = table()
        for column, value in assignments:
            row._setColumnValue( column, value )
            formatted = row._getFormattedColumnValue( column )
            values.append( formatted )
            columns.append( column )
            continue
        
        expr = ", ".join(["%s=%%s" % column for column in columns])
        sql = "UPDATE %s\n    SET %s" % (table.name, expr)
        if where:
            sql += "\n    WHERE %s" % where

        # execute the sql statement
        c = self.db.cursor()

        try:
            c.execute(sql, values)
        except:
            debug.log( 'sql: %s, values: %s' % (sql, values) )
            raise
        self.db.commit()

        return


    def deleteRow(self, table, where=None):
        sql = "DELETE FROM %s" % table.name
        if where:
            sql += "\n    WHERE %s" % where

            # execute the sql statement
            c = self.db.cursor()
            c.execute(sql)
            self.db.commit()

            return


    def createTable(self, table):
        # build the list of table columns
        fields = []
        for name, column in table._columnRegistry.iteritems():
            text = "    %s %s" % (name, column.declaration())
            fields.append(text)

        # build the query
        sql = "CREATE TABLE %s (\n%s\n    )" % (table.name, ",\n".join(fields))

        # execute the sql statement
        c = self.db.cursor()

        try:
            c.execute(sql)
        except:
            debug.log( 'sql: %s' % sql )
            raise
        self.db.commit()

        return


    def dropTable(self, table, cascade=False):
        sql = "DROP TABLE %s" % table.name
        if cascade:
            sql += " CASCADE"

        # execute the sql statement
        c = self.db.cursor()
        c.execute(sql)

        return


    def fetchall(self, table, where=None, sort=None):
        columns = table._columnRegistry.keys()
        
        # build the sql statement
        sql = "SELECT %s FROM %s" % (", ".join(columns), table.name)
        if where:
            sql += " WHERE %s" % where
        if sort:
            sql += " ORDER BY %s" % sort
        
        # execute the sql statement
        c = self.db.cursor()
        debug.log( 'fetchall: sql=%r' % sql )
        c.execute(sql)

        #print c.fetchall(),'<br>'
        #print c.fetchall(),'<br>'
        # walk through the result of the query
        rows = c.fetchall()
        debug.log( 'query result: %r' % (rows,) )
        items = []
        for row in rows:
            # create the object
            item = table()
            item.locator = self.locator
            
            # build the dictionary with the column information
            values = {}
            for key, value in zip(columns, row):
	    	if value is not None:
                    values[key] = value
            # attach it to the object
            item._priv_columns = values

            # add this object tothepile
            items.append(item)

        debug.log( 'items: %r' % (items,) )
        return items
    
    
    def fetchAttributeFromAll(self, table, attribute, where=None, sort=None):
        columns = table._columnRegistry.keys()
        
        # build the sql statement
        sql = "SELECT %s FROM %s" % (attribute, table.name)
        if where:
            sql += " WHERE %s" % where
        if sort:
            sql += " ORDER BY %s" % sort
        
        # execute the sql statement
        c = self.db.cursor()
        debug.log( 'fetchAttributeFromAll: sql=%r' % sql )
        c.execute(sql)

        #print c.fetchall(),'<br>'
        #print c.fetchall(),'<br>'
        # walk through the result of the query
        rows = c.fetchall()
        debug.log( 'query result: %r' % (rows,) )
#        items = []
#        for row in rows:
#            # create the object
#            item = table()
#            item.locator = self.locator
#            
#            # build the dictionary with the column information
#            values = {}
#            for key, value in zip(columns, row):
#            if value is not None:
#                    values[key] = value
#            # attach it to the object
#            item._priv_columns = values
#
#            # add this object tothepile
#            items.append(item)
#
#        debug.log( 'items: %r' % (items,) )
        return rows   


    def __init__(self, name, **kwds):
        self.db = self.connect(database=name, **kwds)

        import pyre.parsing.locators
        self.locator = pyre.parsing.locators.simple("%s database" % name)

        return


# version
__id__ = "$Id: DBManager.py,v 1.4 2008-04-21 03:07:30 aivazis Exp $"

# End of file 
