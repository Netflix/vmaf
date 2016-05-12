#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                         June Kim & Mike McKerns, Caltech
#                        (C) 1997-2010  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
"""
pathos: a framework for heterogeneous computing

Pathos is a framework for heterogenous computing. It primarily provides
the communication mechanisms for configuring and launching parallel
computations across heterogenous resources. Pathos provides stagers and
launchers for parallel and distributed computing, where each launcher
contains the syntactic logic to configure and launch jobs in an execution
environment.  Some examples of included launchers are: a queue-less
MPI-based launcher, a ssh-based launcher, and a multiprocessing launcher.
Pathos also provides a map-reduce algorithm for each of the available
launchers, thus greatly lowering the barrier for users to extend their
code to parallel and distributed resources.  Pathos provides the ability
to interact with batch schedulers and queuing systems, thus allowing large
computations to be easily launched on high-performance computing resources.
One of the most powerful features of pathos is  "tunnel", which enables a
user to automatically wrap any distributed service calls within a ssh-tunnel.

Pathos is divided into four subpackages::
    - dill: a utility for serialization of python objects
    - pox: utilities for filesystem exploration and automated builds
    - pyina: a MPI-based parallel mapper and launcher
    - pathos: distributed parallel map-reduce and ssh communication


Pathos Subpackage 
=================

The pathos subpackage provides a few basic tools to make distributed
computing more accessable to the end user. The goal of pathos is to
allow the user to extend their own code to distributed computing with
minimal refactoring.

Pathos provides methods for configuring, launching, monitoring, and
controlling a service on a remote host. One of the most basic features
of pathos is the ability to configure and launch a RPC-based service
on a remote host. Pathos seeds the remote host with a small `portpicker`
script, which allows the remote host to inform the localhost of a port
that is available for communication.

Beyond the ability to establish a RPC service, and then post requests,
is the ability to launch code in parallel. Unlike parallel computing
performed at the node level (typically with MPI), pathos enables the
user to launch jobs in parallel across heterogeneous distributed resources.
Pathos provides a distributed map-reduce algorithm, where a mix of
local processors and distributed RPC services can be selected.  Pathos
also provides a very basic automated load balancing service, as well as
the ability for the user to directly select the resources.

The high-level "pp_map" interface, yields a map-reduce implementation that
hides the RPC internals from the user. With pp_map, the user can launch
their code in parallel, and as a distributed service, using standard python
and without writing a line of server or parallel batch code.

RPC servers and communication in general is known to be insecure.  However,
instead of attempting to make the RPC communication itself secure, pathos
provides the ability to automatically wrap any distributes service or
communication in a ssh-tunnel. Ssh is a universally trusted method.
Using ssh-tunnels, pathos has launched several distributed calculations
on national lab clusters, and to date has performed test calculations
that utilize node-to-node communication between two national lab clusters
and a user's laptop.  Pathos allows the user to configure and launch
at a very atomistic level, through raw access to ssh and scp. 

Pathos is in the early development stages, and any user feedback is
highly appreciated. Contact Mike McKerns [mmckerns at caltech dot edu]
with comments, suggestions, and any bugs you may find. A list of known
issues is maintained at http://dev.danse.us/trac/pathos/query.


Major Features
==============

Pathos provides a configurable distributed parallel-map reduce interface
to launching RPC service calls, with::
    - a map-reduce interface that extends the python 'map' standard
    - the ability to submit service requests to a selection of servers
    - the ability to tunnel server communications with ssh
    - automated load-balancing between multiprocessing and RPC servers

The pathos core is built on low-level communication to remote hosts using
ssh. The interface to ssh, scp, and ssh-tunneled connections can::
    - configure and launch remote processes with ssh
    - configure and copy file objects with scp
    - establish an tear-down a ssh-tunnel

To get up and running quickly, pathos also provides infrastructure to::
    - easily establish a ssh-tunneled connection to a RPC server


Current Release
===============

This release version is pathos-0.1a1. You can download it here.
The latest version of pathos is available from::
    http://dev.danse.us/trac/pathos

Pathos is distributed under a modified BSD license.


Installation
============

Pathos is packaged to install from source, so you must
download the tarball, unzip, and run the installer::
    [download]
    $ tar -xvzf pathos-0.1a1.tgz
    $ cd pathos-0.1a1
    $ python setup py build
    $ python setup py install

You will be warned of any missing dependencies and/or settings after
you run the "build" step above. Pathos depends on dill, pox, and pyina,
each of which are essentially subpackages of pathos that are also
released independently. Pathos also depends on slightly modified
versions of `pyre` and `parallel python`. All the aforementioned
packages are available on this site, and you must install all of
the dependencies for pathos to have full functionality for heterogeneous
computing. Currently, pyina is optional.

Alternately, pathos can be installed with easy_install::
    [download]
    $ easy_install -f . pathos


Requirements
============

Pathos requires::
    - python, version >= 2.5, version < 3.0
    - dill, version >= 0.1a1
    - pox, version >= 0.1a1
    - pyre, version == 0.8-pathos (*)
    - pp, version == 1.5.7-pathos (*)

Optional requirements::
    - setuptools, version >= 0.6
    - pyina, version >= 0.1a1
    - rpyc, version >= 3.0.6


Usage Notes
===========

Probably the best way to get started is to look at a few of the
examples provided within pathos. See `pathos.examples` for a
set of scripts that demonstrate the configuration and launching of
communications with ssh and scp.

Important classes and functions are found here::
    - pathos.pathos.pp_map          [the map-reduce API definition]
    - pathos.pathos.core            [the high-level command interface] 
    - pathos.pathos.hosts           [the hostname registry interface] 
    - pathos.pathos.Launcher        [the launcher base class]
    - pathos.pathos.Tunnel          [the tunnel base class]

Pathos also provides three convience scripts that are used to establish
secure distributed connections. These scripts are installed to a directory
on the user's $PATH, and thus can be run from anywhere::
    - pathos_tunnel.py              [establish a ssh-tunnel connection]
    - pathos_server.py              [launch a remote RPC server]
    - tunneled_pathos_server.py     [launch a tunneled remote RPC server]

Typing `--help` as an argument to any of the above three scripts will print
out an instructive help message.


More Information
================

Please see http://dev.danse.us/trac/pathos/pyina for further information.
"""
__version__ = '0.1a1'
__author__ = 'Mike McKerns'

__license__ = """
This software is part of the open-source DANSE project at the California
Institute of Technology, and is available subject to the conditions and
terms laid out below. By downloading and using this software you are
agreeing to the following conditions.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met::

    - Redistribution of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistribution in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentations and/or other materials provided with the distribution.

    - Neither the name of the California Institute of Technology nor
      the names of its contributors may be used to endorse or promote
      products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Copyright (c) 2010 California Institute of Technology. All rights reserved.


If you use this software to do productive scientific research that leads to
publication, we ask that you acknowledge use of the software by citing the
following paper in your publication::

    "pathos: a framework for heterogeneous computing",
     Michael McKerns and Michael Aivazis, unpublished;
     http://dev.danse.us/trac/pathos

"""
# high-level interface
import core
import hosts

# launchers
from LauncherSSH import LauncherSSH as SSH_Launcher
from LauncherSCP import LauncherSCP as SCP_Launcher

# tunnels
from Tunnel import Tunnel as SSH_Tunnel

# mappers
import pp_map

# strategies

# tools, utilities, etc
import util

def copyright():
    """print copyright and reference"""
    print __license__[-417:]
    return

# end of file
