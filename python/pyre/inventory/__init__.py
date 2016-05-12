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

# factories
def facility(name, **kwds):
    from Facility import Facility
    return Facility(name, **kwds)


def curator(name):
    from odb.Curator import Curator
    return Curator(name)


def registry(name):
    from odb.Inventory import Inventory
    return Inventory(name)


# persistence
def codecPML():
    from pml.CodecPML import CodecPML
    return CodecPML()


def renderer(mode="pml"):
    if mode == "pml":
        from pml.Renderer import Renderer
        return Renderer()

    import journal
    journal.error.log("'%s': unknown registry rendering mode" % mode)
    return None
    

def parser(mode="pml"):
    if mode == "pml":
        from pml.Parser import Parser
        return Parser()

    import journal
    journal.error.log("'%s': unknown registry parsing mode" % mode)
    return None
    

# builtin property types
def array(name, **kwds):
    from properties.Array import Array
    return Array(name, **kwds)


def bool(name, **kwds):
    from properties.Bool import Bool
    return Bool(name, **kwds)


def dimensional(name, **kwds):
    from properties.Dimensional import Dimensional
    return Dimensional(name, **kwds)


def float(name, **kwds):
    from properties.Float import Float
    return Float(name, **kwds)


def inputFile(name, **kwds):
    from properties.InputFile import InputFile
    return InputFile(name, **kwds)


def int(name, **kwds):
    from properties.Integer import Integer
    return Integer(name, **kwds)


def list(name, **kwds):
    from properties.List import List
    return List(name, **kwds)


def outputFile(name, **kwds):
    from properties.OutputFile import OutputFile
    return OutputFile(name, **kwds)


def preformatted(name, **kwds):
    from properties.Preformatted import Preformatted
    return Preformatted(name, **kwds)


def slice(name, **kwds):
    from properties.Slice import Slice
    return Slice(name, **kwds)


def str(name, **kwds):
    from properties.String import String
    return String(name, **kwds)


# built-in validators
def less(value):
    from validators.Less import Less
    return Less(value)


def lessEqual(value):
    from validators.LessEqual import LessEqual
    return LessEqual(value)


def greater(value):
    from validators.Greater import Greater
    return Greater(value)


def greaterEqual(value):
    from validators.GreaterEqual import GreaterEqual
    return GreaterEqual(value)


def range(low, high):
    from validators.Range import Range
    return Range(low, high)


def choice(set):
    from validators.Choice import Choice
    return Choice(set)


# logical operators on validators
def isBoth(v1, v2):
    from validators.And import And
    return And(v1, v2)


def isEither(v1, v2):
    from validators.Or import Or
    return Or(v1, v2)


def isNot(v):
    from validators.Not import Not
    return Not(v)


# version
__id__ = "$Id: __init__.py,v 1.1.1.1 2006-11-27 00:10:01 aivazis Exp $"

# End of file 
