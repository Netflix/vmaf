#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

#factory method

def parser():
    return Parser()


# implementation of the Parser singleton

from pyre.util.Singleton import Singleton


class Parser(Singleton):


    def extend(self, *modules):
        for module in modules:
            self.context.update(module.__dict__)
        return


    def parse(self, string):
        return eval(string, self.context)
    

    def init(self, *args, **kwds):
        self.context = self._initializeContext()
        return


    def _initializeContext(self):
        context = {}

        modules = self._loadModules()
        for module in  modules:
            context.update(module.__dict__)

        return context


    def _loadModules(self):

        import SI
        import angle
        import area
        import density
        import energy
        import force
        import length
        import mass
        import power
        import pressure
        import speed
        import substance
        import temperature
        import time
        import volume

        modules = [
            SI,
            angle, area, density, energy, force, length, mass, power, pressure, speed, substance,
            temperature, time, volume
            ]

        return modules
        
    
# version
__id__ = "$Id: unitparser.py,v 1.1.1.1 2006-11-27 00:10:08 aivazis Exp $"

# End of file 
