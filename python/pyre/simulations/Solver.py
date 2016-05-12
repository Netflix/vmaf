#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

from pyre.components.Component import Component


class Solver(Component):


    def initialize(self, app):
        self._loopInfo.log("initializing solver '%s'" % self.name)
        return


    def launch(self, app):
        self._loopInfo.log("launching solver '%s'" % self.name)
        return


    def newStep(self, t, step):
        self.t = t
        self.step = step
        self._loopInfo.log(
            "%s: step %d: starting with t = %s" % (self.name, self.step, self.t))
        return


    def applyBoundaryConditions(self):
        self._loopInfo.log(
            "%s: step %d: applying boundary conditions" % (self.name, self.step))
        return


    def stableTimestep(self, dt):
        self._loopInfo.log(
            "%s: step %d: stable timestep dt = %s" % (self.name, self.step, dt))
        return dt


    def advance(self, dt):
        self._loopInfo.log(
            "%s: step %d: advancing the solution by dt = %s" % (self.name, self.step, dt))
        return


    def publishState(self, directory):
        self._monitorInfo.log(
            "%s: step %d: publishing monitoring information at t = %s in '%s'" % 
            (self.name, self.step, self.t, directory))
        return


    def plotFile(self, directory):
        self._loopInfo.log(
            "%s: step %d: visualization information at t = %s in '%s'" % 
            (self.name, self.step, self.t, directory))
        return


    def checkpoint(self, filename):
        self._loopInfo.log(
            "%s: step %d: checkpoint at t = %s in '%s'" % (self.name, self.step, self.t, filename))
        return


    def endTimestep(self, t):
        self._loopInfo.log(
            "%s: step %d: end of timestep at t = %s" % (self.name, self.step, t))
        return


    def endSimulation(self, steps, t):
        self._loopInfo.log(
            "%s: end of timeloop: %d timesteps, t = %s" % (self.name, steps, t))
        return


    def __init__(self, name, facility=None):
        if facility is None:
            facility = "solver"
            
        Component.__init__(self, name, facility)
        
        self._elc = None
                
        import journal
        self._loopInfo = journal.debug("%s.timeloop" % name)
        self._monitorInfo = journal.debug("%s.monitoring" % name)

        from pyre.units.time import second
        self.t = 0.0 * second

        self.step = 0

        return


# version
__id__ = "$Id: Solver.py,v 1.1.1.1 2006-11-27 00:10:06 aivazis Exp $"

#
# End of file
