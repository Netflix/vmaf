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


from Column import Column


class Time(Column):


    def type(self):
        if not self.tz:
            return "time without time zone"
        return "time"


    def __init__(self, name, tz=True, **kwds):
        Column.__init__(self, name, **kwds)
        self.tz = tz
        return


    def __get__(self, instance, cls=None):
        ret = Column.__get__(self, instance, cls = cls)
        if ret is None:
            import time
            return time.ctime()
        return self._cast(ret)


    def _cast(self, value):
        format = '%a %b %d %H:%M:%S %Y'
        if isinstance(value, basestring):
            return calendar.timegm(time.strptime(value, format))
        if isinstance(value, time.struct_time):
            return calendar.timegm(value)
        if isinstance(value, float) or isinstance(value, int):
            return value
        raise NotImplementedError


    def _format(self, value):
        return time.asctime(time.gmtime(value))
    

import time, calendar

# version
__id__ = "$Id: Time.py,v 1.1.1.1 2006-11-27 00:09:55 aivazis Exp $"

# End of file 
