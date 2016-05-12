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
"""
Parallel Python Software, PP Transport

http://www.parallelpython.com - updates, documentation, examples and support
forums
"""
import os
import struct
import socket
import logging

copyright = "Copyright (c) 2005-2009 Vitalii Vanovschi. All rights reserved"
version = "1.5.7"

# compartibility with Python 2.6
try:
    import hashlib
    sha_new = hashlib.sha1
    md5_new = hashlib.md5
except ImportError:
    import sha
    import md5
    sha_new = sha.new
    md5_new = md5.new


class Transport(object):

    def send(self, msg):
        raise NotImplemented("abstact function 'send' must be implemented "\
                "in a subclass")

    def receive(self, preprocess=None):
        raise NotImplemented("abstact function 'receive' must be implemented "\
                "in a subclass")

    def authenticate(self, secret):
        remote_version = self.receive()
        if version != remote_version:
            logging.error("PP version mismatch (local: pp-%s, remote: pp-%s)"
                % (version, remote_version))
            logging.error("Please install the same version of PP on all nodes")
            return False
        srandom = self.receive()
        answer = sha_new(srandom+secret).hexdigest()
        self.send(answer)
        response = self.receive()
        if response == "OK":
            return True
        else:
            return False

    def close(self):
        pass

    def _connect(self, host, port):
        pass


class CTransport(Transport):
    """Cached transport
    """
    rcache = {}

    def hash(self, msg):
        return md5_new(msg).hexdigest()

    def csend(self, msg):
        hash1 = self.hash(msg)
        if hash1 in self.scache:
            self.send("H" + hash1)
        else:
            self.send("N" + msg)
            self.scache[hash1] = True

    def creceive(self, preprocess=None):
        msg = self.receive()
        if msg[0] == 'H':
            hash1 = msg[1:]
        else:
            msg = msg[1:]
            hash1 = self.hash(msg)
            self.rcache[hash1] = map(preprocess, (msg, ))[0]
        return self.rcache[hash1]


class PipeTransport(Transport):

    def __init__(self, r, w):
        self.scache = {}
        self.exiting = False
        if isinstance(r, file) and isinstance(w, file):
            self.r = r
            self.w = w
        else:
            raise TypeError("Both arguments of PipeTransport constructor " \
                    "must be file objects")

    def send(self, msg):
        self.w.write(struct.pack("!Q", len(msg)))
        self.w.flush()
        self.w.write(msg)
        self.w.flush()

    def receive(self, preprocess=None):
        size_packed = self.r.read(struct.calcsize("!Q"))
        msg_len = struct.unpack("!Q", size_packed)[0]
        msg = self.r.read(msg_len)
        return map(preprocess, (msg, ))[0]

    def close(self):
        self.w.close()
        self.r.close()


class SocketTransport(Transport):

    def __init__(self, socket1=None):
        if socket1:
            self.socket = socket1
        else:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.scache = {}

    def send(self, data):
        size = struct.pack("!Q", len(data))
        t_size = struct.calcsize("!Q")
        s_size = 0L
        while s_size < t_size:
            p_size = self.socket.send(size[s_size:])
            if p_size == 0:
                raise RuntimeError("Socket connection is broken")
            s_size += p_size

        t_size = len(data)
        s_size = 0L
        while s_size < t_size:
            p_size = self.socket.send(data[s_size:])
            if p_size == 0:
                raise RuntimeError("Socket connection is broken")
            s_size += p_size

    def receive(self, preprocess=None):
        e_size = struct.calcsize("!Q")
        r_size = 0
        data = ""
        while r_size < e_size:
            msg = self.socket.recv(e_size-r_size)
            if msg == "":
                raise RuntimeError("Socket connection is broken")
            r_size += len(msg)
            data += msg
        e_size = struct.unpack("!Q", data)[0]

        r_size = 0
        data = ""
        while r_size < e_size:
            msg = self.socket.recv(e_size-r_size)
            if msg == "":
                raise RuntimeError("Socket connection is broken")
            r_size += len(msg)
            data += msg
        return data

    def close(self):
        self.socket.close()

    def _connect(self, host, port):
        self.socket.connect((host, port))


class CPipeTransport(PipeTransport, CTransport):
    pass


class CSocketTransport(SocketTransport, CTransport):
    pass

# Parallel Python Software: http://www.parallelpython.com
