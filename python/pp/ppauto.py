# Parallel Python Software: http://www.parallelpython.com
# Copyright (c) 2005-2009, Vitalii Vanovschi
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the author nor the names of its contributors
#      may be used to endorse or promote products derived from this software
#      without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
"""Parallel Python Software, Auto-Discovery Service

http://www.parallelpython.com - updates, documentation, examples and support
forums
"""

import socket
import sys
import time
import logging
import thread

copyright = "Copyright (c) 2005-2009 Vitalii Vanovschi. All rights reserved"
version = "1.5.7"

# broadcast every 10 sec
BROADCAST_INTERVAL = 10


class Discover(object):
    """Auto-discovery service class"""

    def __init__(self, base, isclient=False):
        self.base = base
        self.hosts = []
        self.isclient = isclient

    def run(self, interface_addr, broadcast_addr):
        """Starts auto-discovery"""
        self.interface_addr = interface_addr
        self.broadcast_addr = broadcast_addr
        self.bsocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.bsocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.bsocket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        try:
            self.listen()
        except:
            sys.excepthook(*sys.exc_info())

    def broadcast(self):
        """Sends a broadcast"""
        if self.isclient:
            logging.debug("Client sends initial broadcast to (%s, %i)"
                    % self.broadcast_addr)
            self.bsocket.sendto("C", self.broadcast_addr)
        else:
            while True:
                logging.debug("Server sends broadcast to (%s, %i)"
                        % self.broadcast_addr)
                self.bsocket.sendto("S", self.broadcast_addr)
                time.sleep(BROADCAST_INTERVAL)

    def listen(self):
        """Listens for broadcasts from other clients/servers"""
        logging.debug("Listening (%s, %i)" % self.interface_addr)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        s.bind(self.interface_addr)

        thread.start_new_thread(self.broadcast, ())

        while True:
            try:
                message, (host, port) = s.recvfrom(1024)
                remote_address = (host, self.broadcast_addr[1])
                hostid = host + ":" + str(self.broadcast_addr[1])
                logging.debug("Discovered host (%s, %i) message=%c"
                        % (remote_address + (message[0], )))
                if not self.base.autopp_list.get(hostid, 0) and self.isclient \
                        and message[0] == 'S':
                    logging.debug("Connecting to host %s" % (hostid, ))
                    thread.start_new_thread(self.base.connect1,
                            remote_address+(False, ))
                if not self.isclient and message[0] == 'C':
                    logging.debug("Replying to host %s" % (hostid, ))
                    self.bsocket.sendto("S", self.broadcast_addr)
            except:
                logging.error("An error has occured during execution of "
                        "Discover.listen")
                sys.excepthook(*sys.exc_info())
