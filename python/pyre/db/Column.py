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


class Column(object):


    def type(self):
        raise NotImplementedError("class %r must override 'type'" % self.__class__.__name__)


    def getFormattedValue(self, instance, cls = None):
        'obtain value that is formatted for db access'
        value = self.__get__(instance, cls = cls)
        return self._format( value )
    

    def declaration(self):
        text = [ self.type() ]
        if self.default is not None:
            text.append("DEFAULT %r" % self.default)
        if self.constraints:
            text.append(self.constraints)

        return " ".join(text)


    def __init__(self, name, default=None, auto=False, constraints=None, meta=None):
        self.name = name
        self.default = default
        self.auto = auto
        self.constraints = constraints

        if meta is None:
            meta = {}
        self.meta = meta

        return


    def __get__(self, instance, cls=None):

        # attempt to get hold of the instance's attribute record
        try:
            return instance._getColumnValue(self.name)

        # instance is None when accessed as a class variable
        except AttributeError:
            # catch bad descriptors or changes in the python conventions
            if instance is not None:
                import journal
                firewall = journal.firewall("pyre.inventory")
                firewall.log("AttributeError on non-None instance. Bad descriptor?")

            # interpret this usage as a request for the trait object itself
            return self

        except KeyError:
            # look up the registered default value
            default = self.default

            # if we don't have a default, mark this column as uninitialized
            if default is None:
                return default

            # otherwise, store the default as the actual field value
            return instance._setColumnValue(self.name, default)


        # not reachable
        import journal
        journal.firewall('pyre.db').log("UNREACHABLE")
        return None


    def __set__(self, instance, value):
        value = self._cast(value)
        return instance._setColumnValue(self.name, value)


    def _cast(self, value):
        return value


    def _format(self, value):
        'format the given value so that it can be used in db cmd'
        #by default, just return the value
        return value


# version
__id__ = "$Id: Column.py,v 1.3 2008-04-13 07:50:19 aivazis Exp $"

# End of file 
