#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2008  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


import psycopg2
from DBManager import DBManager


class Psycopg2(DBManager):


    # exceptions
    ProgrammingError = psycopg2.ProgrammingError
    IntegrityError = psycopg2.IntegrityError


    # interface
    def connect(self, **kwds):
        ret = psycopg2.connect(**kwds)
        if not hasattr(ret, 'autocommit'):
            return wrapper( ret )
        return ret


class wrapper(object):

    def __init__(self, core):
        self._core = core
        return


    def __getattribute__(self, name):
        try: return getattr( self._core, name )
        except: return object.__getattribute__(self, name)


    def autocommit(self, on_off=1):
        """autocommit(on_off=1) -> switch autocommit on (1) or off (0)"""
        if on_off > 0:
            self.set_isolation_level(0)
        else:
            self.set_isolation_level(2)


# version
__id__ = "$Id: Psycopg2.py,v 1.1 2008-04-04 08:36:46 aivazis Exp $"

# End of file 
