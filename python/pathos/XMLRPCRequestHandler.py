#!/usr/bin/env python
#
## XMLRPC request handler class
# adapted from J. Kim's XMLRPC request handler class
# by mmckerns@caltech.edu

"""
This module contains the base class for XML-RPC request handlers,
and derives from python's base HTTP request handler. Intended to
handle requests for a `pathos.XMLRPCServer`.

"""
__all__ = ['XMLRPCRequestHandler']

import os
import xmlrpclib
from BaseHTTPServer import BaseHTTPRequestHandler
import journal
import util

class XMLRPCRequestHandler(BaseHTTPRequestHandler):
    ''' create a XML-RPC request handler '''

    _debug = journal.debug('pathos')
    _error = journal.debug('pathos')
        
    def do_POST(self):
        """ Access point from HTTP handler """
        
        def onParent(pid, fromchild, tochild):
            self._server._registerChild(pid, fromchild)
            tochild.write('done\n')
            tochild.flush()

        def onChild(pid, fromparent, toparent):
            try:
                response = self._server._marshaled_dispatch(data)
                self._sendResponse(response)
                line = fromparent.readline()
                toparent.write('done\n')
                toparent.flush()
            except:
                journal.debug('pathos').log(util.print_exc_info())
            os._exit(0)

        try:
            data = self.rfile.read(int(self.headers['content-length']))
            params, method = xmlrpclib.loads(data)
            if method == 'run':
                return util.spawn2(onParent, onChild)
            else:
                response = self._server._marshaled_dispatch(data)
                self._sendResponse(response)
                return
        except:
            self._error.log(util.print_exc_info())
            self.send_response(500)
            self.end_headers()
            return


    def log_message(self, format, *args):
        """ Overriding BaseHTTPRequestHandler.log_message() """

        self._debug.log("%s - - [%s] %s\n" %
                        (self.address_string(),
                         self.log_date_time_string(),
                         format%args))


    def _sendResponse(self, response):
        """ Write the XML-RPC response """

        self.send_response(200)
        self.send_header("Content-type", "text/xml")
        self.send_header("Content-length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)
        self.wfile.flush()
        self.connection.shutdown(1)


    def __init__(self, server, socket):
        """
Override BaseHTTPRequestHandler.__init__(): we need to be able
to have (potentially) multiple handler objects at a given time.

Inputs:
    server  -- server object to handle requests for 
    socket  -- socket connection 
        """

        ## Settings required by BaseHTTPRequestHandler
        self.rfile = socket.makefile('rb', -1)
        self.wfile = socket.makefile('wb', 0)
        self.connection = socket
        self.client_address = (server.host, server.port)
        
        self._server = server


# End of file
