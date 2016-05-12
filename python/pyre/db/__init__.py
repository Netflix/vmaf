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


def connect(database, wrapper=None, **kwds):
    if wrapper is None or wrapper == "psycopg":
        from Psycopg import Psycopg
        return Psycopg(database, **kwds)

    if wrapper == "psycopg2":
        from Psycopg2 import Psycopg2
        return Psycopg2(database, **kwds)

    import journal
    journal.error("pyre.db").log("%r: unknown db wrapper type" % wrapper)
    return None


def bigint(**kwds):
    from BigInt import BigInt
    return BigInt(**kwds)


def boolean(**kwds):
    from Boolean import Boolean
    return Boolean(**kwds)


def char(**kwds):
    from Char import Char
    return Char(**kwds)


def date(**kwds):
    from Date import Date
    return Date(**kwds)


def double(**kwds):
    from Double import Double
    return Double(**kwds)


def doubleArray(**kwds):
    from DoubleArray import DoubleArray
    return DoubleArray(**kwds)


def integer(**kwds):
    from Integer import Integer
    return Integer(**kwds)


def integerArray(**kwds):
    from IntegerArray import IntegerArray
    return IntegerArray(**kwds)


def interval(**kwds):
    from Interval import Interval
    return Interval(**kwds)


def real(**kwds):
    from Real import Real
    return Real(**kwds)


def reference(**kwds):
    from Reference import Reference
    return Reference(**kwds)


def smallint(**kwds):
    from SmallInt import SmallInt
    return SmallInt(**kwds)


def tableRegistry():
    from VersatileReference  import tableRegistry
    return tableRegistry()


def time(**kwds):
    from Time import Time
    return Time(**kwds)


def timestamp(**kwds):
    from Timestamp import Timestamp
    return Timestamp(**kwds)


def varchar(**kwds):
    from VarChar import VarChar
    return VarChar(**kwds)


def varcharArray(**kwds):
    from VarCharArray import VarCharArray
    return VarCharArray(**kwds)


def versatileReference(**kwds):
    from VersatileReference import VersatileReference
    return VersatileReference(**kwds)

# version
__id__ = "$Id: __init__.py,v 1.2 2008-04-04 08:37:14 aivazis Exp $"

# End of file 
