#!/usr/bin/env python
#
## utility functions for distributed computing
# adapted from J. Kim & M. McKerns utility functions
# by mmckerns@caltech.edu

"""
utilities for distributed computing
"""

class portnumber(object):
    '''port selector

Usage:
    >>> pick = portnumber(min=1024,max=65535)
    >>> print pick()
    '''

    def __init__(self, min=0, max=64*1024):
        '''select a port number from a given range.

The first call will return a random number from the available range,
and each subsequent call will return the next number in the range.

Inputs:
    min -- minimum port number  [default = 0]
    max -- maximum port number  [default = 65536]
        '''
        self.min = min
        self.max = max
        self.first = -1
        self.current = -1
        return

    def __call__(self):
        import random
        
        if self.current < 0: #first call
            self.current = random.randint(self.min, self.max)
            self.first = self.current
            return self.current
        else:
            self.current += 1
            
            if self.current > self.max:
                self.current = self.min
            if self.current == self.first: 
                raise RuntimeError, 'Range exhausted'
            return self.current


def print_exc_info():
    """thread-safe return of string from print_exception call"""

    import StringIO, traceback
    
    sio = StringIO.StringIO()
    traceback.print_exc(file=sio) #thread-safe print_exception to string
    sio.seek(0, 0)
    
    return sio.read()


def spawn(onParent, onChild):
    """a fork wrapper

Calls onParent(pid, fromchild) in parent process,
      onChild(pid, toparent) in child process.
    """
    
    import os
    
    c2pread, c2pwrite = os.pipe()
        
    pid = os.fork()
    if pid > 0:
        os.close(c2pwrite)            
        fromchild = os.fdopen(c2pread)
        return onParent(pid, fromchild)

    os.close(c2pread)
    toparent = os.fdopen(c2pwrite, 'w', 0)
    pid = os.getpid()

    return onChild(pid, toparent)


def spawn2(onParent, onChild):
    """an alternate fork wrapper

Calls onParent(pid, fromchild, tochild) in parent process,
      onChild(pid, fromparent, toparent) in child process.
    """
    
    import os

    p2cread, p2cwrite = os.pipe()
    c2pread, c2pwrite = os.pipe()
        
    pid = os.fork()
    if pid > 0:
        os.close(p2cread)
        os.close(c2pwrite)            
        fromchild = os.fdopen(c2pread, 'r')
        tochild = os.fdopen(p2cwrite, 'w', 0)
        return onParent(pid, fromchild, tochild)

    os.close(p2cwrite)
    os.close(c2pread)
    fromparent = os.fdopen(p2cread, 'r')
    toparent = os.fdopen(c2pwrite, 'w', 0)
    pid = os.getpid()

    return onChild(pid, fromparent, toparent)


if __name__ == '__main__':

    pick = portnumber(min=1024,max=65535)
    print pick()


# End of file
