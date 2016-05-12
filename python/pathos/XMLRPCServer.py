#!/usr/bin/env python
#
## XMLRPC Server class
# adapted from J. Kim's XMLRPC server class
# by mmckerns@caltech.edu

"""
This module contains the base class for pathos XML-RPC servers,
and derives from python's SimpleXMLRPCServer.


Usage
=====

A typical setup for an XML-RPC server will roughly follow this example:

    >>> # establish a XML-RPC server on a host at a given port
    >>> host = 'localhost'
    >>> port = 1234
    >>> server = XMLRPCServer(host, port)
    >>> print 'port=%d' % server.port
    >>>
    >>> # register a method the server can handle requests for
    >>> def add(x, y):
    ...     return x + y
    >>> server.register_function(add)
    >>>
    >>> # activate the callback methods and begin serving requests
    >>> server.activate()
    >>> server.serve()


The following is an example of how to make requests to the above server:

    >>> # establish a proxy connection to the server at (host,port)
    >>> host = 'localhost'
    >>> port = 1234
    >>> proxy = xmlrpclib.ServerProxy('http://%s:%d' % (host, port))
    >>> print '1 + 2 =', proxy.add(1, 2)
    >>> print '3 + 4 =', proxy.add(3, 4)

"""
__all__ = ['XMLRPCServer']

import os
import socket
from SimpleXMLRPCServer import SimpleXMLRPCDispatcher
import journal
from Server import Server #XXX: in pythia-0.6, was pyre.ipc.Server
from XMLRPCRequestHandler import XMLRPCRequestHandler
import util


class XMLRPCServer(Server, SimpleXMLRPCDispatcher):
    '''extends base pathos server to an XML-RPC dispatcher'''

    def activate(self):
        """install callbacks"""
        
        Server.activate(self)
        self._selector.notifyOnReadReady(self._socket, self._onConnection)
        self._selector.notifyWhenIdle(self._onSelectorIdle)

        
    def serve(self):
        """enter the select loop... and wait for service requests"""
        
        timeout = 5
        Server.serve(self, 5)


    def _marshaled_dispatch(self, data, dispatch_method = None):
        """override SimpleXMLRPCDispatcher._marshaled_dispatch() fault string"""

        import xmlrpclib
        from xmlrpclib import Fault

        params, method = xmlrpclib.loads(data)

        # generate response
        try:
            if dispatch_method is not None:
                response = dispatch_method(method, params)
            else:
                response = self._dispatch(method, params)
            # wrap response in a singleton tuple
            response = (response,)
            response = xmlrpclib.dumps(response, methodresponse=1)
        except Fault, fault:
            fault.faultString = util.print_exc_info()
            response = xmlrpclib.dumps(fault)
        except:
            # report exception back to server
            response = xmlrpclib.dumps(
                xmlrpclib.Fault(1, "\n%s" % util.print_exc_info())
                )

        return response


    def _registerChild(self, pid, fromchild):
        """register a child process so it can be retrieved on select events"""
        
        self._activeProcesses[fromchild] = pid
        self._selector.notifyOnReadReady(fromchild,
                                         self._handleMessageFromChild)


    def _unRegisterChild(self, fd):
        """remove a child process from active process register"""
        
        del self._activeProcesses[fd]


    def _handleMessageFromChild(self, selector, fd):
        """handler for message from a child process"""
        
        line = fd.readline()
        if line[:4] == 'done':
            pid = self._activeProcesses[fd]
            os.waitpid(pid, 0)
        self._unRegisterChild(fd)


    def _onSelectorIdle(self, selector):
        '''something to do when there's no requests'''
        return True


    def _installSocket(self, host, port):
        """prepare a listening socket"""
        
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if port == 0: #Get a random port
            pick = util.portnumber(min=port, max=64*1024)
            while True:
                try:
                    port = pick()
                    s.bind((host, port))
                    break
                except socket.error:
                    continue
        else: #Designated port
            s.bind((host, port))
            
        s.listen(10)
        self._socket = s
        self.host = host
        self.port = port
        return
        
    def _onConnection(self, selector, fd):
        '''upon socket connection, establish a request handler'''
        if isinstance(fd, socket.SocketType):
            return self._onSocketConnection(fd)
        return None


    def _onSocketConnection(self, socket):
        '''upon socket connections, establish a request handler'''
        conn, addr = socket.accept()
        handler = XMLRPCRequestHandler(server=self, socket=conn)
        handler.handle()
        return True


    def __init__(self, host, port):
        '''create a XML-RPC server

Takes two initial inputs:
    host  -- hostname of XML-RPC server host
    port  -- port number for server requests
        '''
        Server.__init__(self)
        SimpleXMLRPCDispatcher.__init__(self,allow_none=False,encoding=None)

        self._installSocket(host, port)
        self._activeProcesses = {} #{ fd : pid }


if __name__ == '__main__':
    pass


# End of file
