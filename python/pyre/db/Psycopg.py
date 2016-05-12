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


import psycopg2 as psycopg
from DBManager import DBManager


class Psycopg(DBManager):


    # exceptions
    ProgrammingError = psycopg.ProgrammingError
    IntegrityError = psycopg.IntegrityError

    # interface
    def connect(self, **kwds):
        return psycopg.connect(**kwds)


# version
__id__ = "$Id: Psycopg.py,v 1.1.1.1 2006-11-27 00:09:55 aivazis Exp $"

# End of file 
