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
Parallel Python Software, Execution Server

http://www.parallelpython.com - updates, documentation, examples and support
forums
"""

import os
import thread
import logging
import inspect
import sys
import types
import time
import atexit
import user
try:
    import dill as pickle
except ImportError:
    import cPickle as pickle
import pptransport
import ppauto

copyright = "Copyright (c) 2005-2009 Vitalii Vanovschi. All rights reserved"
version = "1.5.7"
__version__ = version + "-pathos"

# reconnect persistent rworkers in 5 sec
_RECONNECT_WAIT_TIME = 5

# we need to have set even in Python 2.3
try:
    set
except NameError:
    from sets import Set as set 

_USE_SUBPROCESS = False
try:
    import subprocess
    _USE_SUBPROCESS = True
except ImportError:
    import popen2


class _Task(object):
    """Class describing single task (job)
    """

    def __init__(self, server, tid, callback=None,
            callbackargs=(), group='default'):
        """Initializes the task"""
        self.lock = thread.allocate_lock()
        self.lock.acquire()
        self.tid = tid
        self.server = server
        self.callback = callback
        self.callbackargs = callbackargs
        self.group = group
        self.finished = False
        self.unpickled = False

    def finalize(self, sresult):
        """Finalizes the task.

           For internal use only"""
        self.sresult = sresult
        if self.callback:
            self.__unpickle()
        self.lock.release()
        self.finished = True

    def __call__(self, raw_result=False):
        """Retrieves result of the task"""
        self.wait()

        if not self.unpickled and not raw_result:
            self.__unpickle()

        if raw_result:
            return self.sresult
        else:
            return self.result

    def wait(self):
        """Waits for the task"""
        if not self.finished:
            self.lock.acquire()
            self.lock.release()

    def __unpickle(self):
        """Unpickles the result of the task"""
        self.result, sout = pickle.loads(self.sresult)
        self.unpickled = True
        if len(sout) > 0:
            print sout,
        if self.callback:
            args = self.callbackargs + (self.result, )
            self.callback(*args)


class _Worker(object):
    """Local worker class
    """
    command = "\"" + sys.executable + "\" -u \"" \
            + os.path.dirname(os.path.abspath(__file__))\
            + os.sep + "ppworker.py\""

    if sys.platform.startswith("win"):
        # workargound for windows
        command = "\"" + command + "\""
    else:
        # do not show "Borken pipe message" at exit on unix/linux
        command += " 2>/dev/null"

    def __init__(self, restart_on_free, pickle_proto):
        """Initializes local worker"""
        self.restart_on_free = restart_on_free
        self.pickle_proto = pickle_proto
        self.start()

    def start(self):
        """Starts local worker"""
        if _USE_SUBPROCESS:
            proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, \
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, \
                    shell=True)
            self.t = pptransport.CPipeTransport(proc.stdout, proc.stdin)
        else:
            self.t = pptransport.CPipeTransport(\
                    *popen2.popen3(self.command)[:2])

        self.pid = int(self.t.receive())
        self.t.send(str(self.pickle_proto))
        self.is_free = True

    def stop(self):
        """Stops local worker"""
        self.is_free = False
        self.t.close()

    def restart(self):
        """Restarts local worker"""
        self.stop()
        self.start()

    def free(self):
        """Frees local worker"""
        if self.restart_on_free:
            self.restart()
        else:
            self.is_free = True


class _RWorker(pptransport.CSocketTransport):
    """Remote worker class
    """

    def __init__(self, host, port, secret, message=None, persistent=True):
        """Initializes remote worker"""
        self.persistent = persistent
        self.host = host
        self.port = port
        self.secret = secret
        self.address = (host, port)
        self.id = host + ":" + str(port)
        logging.debug("Creating Rworker id=%s persistent=%s"
                % (self.id, persistent))
        self.connect(message)
        self.is_free = True

    def __del__(self):
        """Closes connection with remote server"""
        self.close()

    def connect(self, message=None):
        """Connects to a remote server"""
        while True:
            try:
                pptransport.SocketTransport.__init__(self)
                self._connect(self.host, self.port)
                if not self.authenticate(self.secret):
                    logging.error("Authentication failed for host=%s, port=%s"
                            % (self.host, self.port))
                    return False
                if message:
                    self.send(message)
                self.is_free = True
                return True
            except:
                if not self.persistent:
                    logging.debug("Deleting from queue Rworker %s"
                            % (self.id, ))
                    return False
#                print sys.excepthook(*sys.exc_info())
                logging.debug("Failed to reconnect with " \
                        "(host=%s, port=%i), will try again in %i s"
                        % (self.host, self.port, _RECONNECT_WAIT_TIME))
                time.sleep(_RECONNECT_WAIT_TIME)


class _Statistics(object):
    """Class to hold execution statisitcs for a single node
    """

    def __init__(self, ncpus, rworker=None):
        """Initializes statistics for a node"""
        self.ncpus = ncpus
        self.time = 0.0
        self.njobs = 0
        self.rworker = rworker


class Template(object):
    """Template class
    """

    def __init__(self, job_server, func, depfuncs=(), modules=(),
            callback=None, callbackargs=(), group='default', globals=None):
        """Creates Template instance

           jobs_server - pp server for submitting jobs
           func - function to be executed
           depfuncs - tuple with functions which might be called from 'func'
           modules - tuple with module names to import
           callback - callback function which will be called with argument
                   list equal to callbackargs+(result,)
                   as soon as calculation is done
           callbackargs - additional arguments for callback function
           group - job group, is used when wait(group) is called to wait for
           jobs in a given group to finish
           globals - dictionary from which all modules, functions and classes
           will be imported, for instance: globals=globals()"""
        self.job_server = job_server
        self.func = func
        self.depfuncs = depfuncs
        self.modules = modules
        self.callback = callback
        self.callbackargs = callbackargs
        self.group = group
        self.globals = globals

    def submit(self, *args):
        """Submits function with *arg arguments to the execution queue
        """
        return self.job_server.submit(self.func, args, self.depfuncs,
                self.modules, self.callback, self.callbackargs,
                self.group, self.globals)


class Server(object):
    """Parallel Python SMP execution server class
    """

    default_port = 60000
    default_secret = "epo20pdosl;dksldkmm"

    def __init__(self, ncpus="autodetect", ppservers=(), secret=None,
            loglevel=logging.WARNING, logstream=sys.stderr,
            restart=False, proto=0):
        """Creates Server instance

           ncpus - the number of worker processes to start on the local
                   computer, if parameter is omitted it will be set to
                   the number of processors in the system
           ppservers - list of active parallel python execution servers
                   to connect with
           secret - passphrase for network connections, if omitted a default
                   passphrase will be used. It's highly recommended to use a
                   custom passphrase for all network connections.
           loglevel - logging level
           logstream - log stream destination
           restart - wheather to restart worker process after each task completion
           proto - protocol number for pickle module

           With ncpus = 1 all tasks are executed consequently
           For the best performance either use the default "autodetect" value
           or set ncpus to the total number of processors in the system
        """

        if not isinstance(ppservers, tuple):
            raise TypeError("ppservers argument must be a tuple")

        self.__initlog(loglevel, logstream)
        logging.debug("Creating server instance (pp-" + version+")")
        self.__tid = 0
        self.__active_tasks = 0
        self.__active_tasks_lock = thread.allocate_lock()
        self.__queue = []
        self.__queue_lock = thread.allocate_lock()
        self.__workers = []
        self.__rworkers = []
        self.__rworkers_reserved = []
        self.__rworkers_reserved4 = []
        self.__sourcesHM = {}
        self.__sfuncHM = {}
        self.__waittasks = []
        self.__waittasks_lock = thread.allocate_lock()
        self.__exiting = False
        self.__accurate_stats = True
        self.autopp_list = {}
        self.__active_rworkers_list_lock = thread.allocate_lock()
        self.__restart_on_free = restart
        self.__pickle_proto = proto

        # add local directory and sys.path to PYTHONPATH
        pythondirs = [os.getcwd()] + sys.path

        if "PYTHONPATH" in os.environ and os.environ["PYTHONPATH"]:
            pythondirs += os.environ["PYTHONPATH"].split(os.pathsep)
        os.environ["PYTHONPATH"] = os.pathsep.join(set(pythondirs))

        atexit.register(self.destroy)
        self.__stats = {"local": _Statistics(0)}
        self.set_ncpus(ncpus)

        self.ppservers = []
        self.auto_ppservers = []

        for ppserver in ppservers:
            ppserver = ppserver.split(":")
            host = ppserver[0]
            if len(ppserver)>1:
                port = int(ppserver[1])
            else:
                port = Server.default_port
            if host.find("*") == -1:
                self.ppservers.append((host, port))
            else:
                if host == "*":
                    host = "*.*.*.*"
                interface = host.replace("*", "0")
                broadcast = host.replace("*", "255")
                self.auto_ppservers.append(((interface, port),
                        (broadcast, port)))
        self.__stats_lock = thread.allocate_lock()
        if secret is not None:
            if not isinstance(secret, types.StringType):
                raise TypeError("secret must be of a string type")
            self.secret = str(secret)
        elif hasattr(user, "pp_secret"):
            secret = user["pp_secret"]
            if not isinstance(secret, types.StringType):
                raise TypeError("secret must be of a string type")
            self.secret = str(secret)
        else:
            self.secret = Server.default_secret
        self.__connect()
        self.__creation_time = time.time()
        logging.info("pp local server started with %d workers"
                % (self.__ncpus, ))

    def submit(self, func, args=(), depfuncs=(), modules=(),
            callback=None, callbackargs=(), group='default', globals=None):
        """Submits function to the execution queue

            func - function to be executed
            args - tuple with arguments of the 'func'
            depfuncs - tuple with functions which might be called from 'func'
            modules - tuple with module names to import
            callback - callback function which will be called with argument
                    list equal to callbackargs+(result,)
                    as soon as calculation is done
            callbackargs - additional arguments for callback function
            group - job group, is used when wait(group) is called to wait for
            jobs in a given group to finish
            globals - dictionary from which all modules, functions and classes
            will be imported, for instance: globals=globals()
        """

        # perform some checks for frequent mistakes
        if self.__exiting:
            raise RuntimeError("Cannot submit jobs: server"\
                    " instance has been destroyed")

        if not isinstance(args, tuple):
            raise TypeError("args argument must be a tuple")

        if not isinstance(depfuncs, tuple):
            raise TypeError("depfuncs argument must be a tuple")

        if not isinstance(modules, tuple):
            raise TypeError("modules argument must be a tuple")

        if not isinstance(callbackargs, tuple):
            raise TypeError("callbackargs argument must be a tuple")

        for module in modules:
            if not isinstance(module, types.StringType):
                raise TypeError("modules argument must be a list of strings")

        tid = self.__gentid()

        if globals:
            modules += tuple(self.__find_modules("", globals))
            modules = tuple(set(modules))
            self.__logger.debug("Task %i will autoimport next modules: %s" %
                    (tid, str(modules)))
            for object1 in globals.values():
                if isinstance(object1, types.FunctionType) \
                        or isinstance(object1, types.ClassType):
                    depfuncs += (object1, )

        task = _Task(self, tid, callback, callbackargs, group)

        self.__waittasks_lock.acquire()
        self.__waittasks.append(task)
        self.__waittasks_lock.release()

        # if the function is a method of a class add self to the arguments list
        if isinstance(func, types.MethodType) and func.im_self is not None:
            args = (func.im_self, ) + args

        # if there is an instance of a user deined class in the arguments add
        # whole class to dependancies
        for arg in args:
            # Checks for both classic or new class instances
            if isinstance(arg, types.InstanceType) \
                    or str(type(arg))[:6] == "<class":
                depfuncs += (arg.__class__, )

        # if there is a function in the arguments add this
        # function to dependancies
        for arg in args:
            if isinstance(arg, types.FunctionType):
                depfuncs += (arg, )

        sfunc = self.__dumpsfunc((func, ) + depfuncs, modules)
        sargs = pickle.dumps(args, self.__pickle_proto)

        self.__queue_lock.acquire()
        self.__queue.append((task, sfunc, sargs))
        self.__queue_lock.release()

        self.__logger.debug("Task %i submited, function='%s'" %
                (tid, func.func_name))
        self.__scheduler()
        return task

    def wait(self, group=None):
        """Waits for all jobs in a given group to finish.
           If group is omitted waits for all jobs to finish
        """
        while True:
            self.__waittasks_lock.acquire()
            for task in self.__waittasks:
                if not group or task.group == group:
                    self.__waittasks_lock.release()
                    task.wait()
                    break
            else:
                self.__waittasks_lock.release()
                break

    def get_ncpus(self):
        """Returns the number of local worker processes (ppworkers)"""
        return self.__ncpus

    def set_ncpus(self, ncpus="autodetect"):
        """Sets the number of local worker processes (ppworkers)

        ncpus - the number of worker processes, if parammeter is omitted
                it will be set to the number of processors in the system"""
        if ncpus == "autodetect":
            ncpus = self.__detect_ncpus()
        if not isinstance(ncpus, int):
            raise TypeError("ncpus must have 'int' type")
        if ncpus < 0:
            raise ValueError("ncpus must be an integer > 0")
        if ncpus > len(self.__workers):
            self.__workers.extend([_Worker(self.__restart_on_free, 
                    self.__pickle_proto) for x in\
                    range(ncpus - len(self.__workers))])
        self.__stats["local"].ncpus = ncpus
        self.__ncpus = ncpus

    def get_active_nodes(self):
        """Returns active nodes as a dictionary
        [keys - nodes, values - ncpus]"""
        active_nodes = {}
        for node, stat in self.__stats.items():
            if node == "local" or node in self.autopp_list \
                    and self.autopp_list[node]:
                active_nodes[node] = stat.ncpus
        return active_nodes

    def get_stats(self):
        """Returns job execution statistics as a dictionary"""
        for node, stat in self.__stats.items():
            if stat.rworker:
                try:
                    stat.rworker.send("TIME")
                    stat.time = float(stat.rworker.receive())
                except:
                    self.__accurate_stats = False
                    stat.time = 0.0
        return self.__stats

    def print_stats(self):
        """Prints job execution statistics. Useful for benchmarking on
           clusters"""

        print "Job execution statistics:"
        walltime = time.time()-self.__creation_time
        statistics = self.get_stats().items()
        totaljobs = 0.0
        for ppserver, stat in statistics:
            totaljobs += stat.njobs
        print " job count | % of all jobs | job time sum | " \
                "time per job | job server"
        for ppserver, stat in statistics:
            if stat.njobs:
                print "    %6i |        %6.2f |     %8.4f |  %11.6f | %s" \
                        % (stat.njobs, 100.0*stat.njobs/totaljobs, stat.time,
                        stat.time/stat.njobs, ppserver, )
        print "Time elapsed since server creation", walltime

        if not self.__accurate_stats:
            print "WARNING: statistics provided above is not accurate" \
                  " due to job rescheduling"
        print

    # all methods below are for internal use only

    def insert(self, sfunc, sargs, task=None):
        """Inserts function into the execution queue. It's intended for
           internal use only (ppserver.py).
        """
        if not task:
            tid = self.__gentid()
            task = _Task(self, tid)
        self.__queue_lock.acquire()
        self.__queue.append((task, sfunc, sargs))
        self.__queue_lock.release()

        self.__logger.debug("Task %i inserted" % (task.tid, ))
        self.__scheduler()
        return task

    def connect1(self, host, port, persistent=True):
        """Conects to a remote ppserver specified by host and port"""
        try:
            rworker = _RWorker(host, port, self.secret, "STAT", persistent)
            ncpus = int(rworker.receive())
            hostid = host+":"+str(port)
            self.__stats[hostid] = _Statistics(ncpus, rworker)

            for x in range(ncpus):
                rworker = _RWorker(host, port, self.secret, "EXEC", persistent)
                self.__update_active_rworkers(rworker.id, 1)
                # append is atomic - no need to lock self.__rworkers
                self.__rworkers.append(rworker)
            #creating reserved rworkers
            for x in range(ncpus):
                rworker = _RWorker(host, port, self.secret, "EXEC", persistent)
                self.__update_active_rworkers(rworker.id, 1)
                self.__rworkers_reserved.append(rworker)
            #creating reserved4 rworkers
            for x in range(ncpus*0):
                rworker = _RWorker(host, port, self.secret, "EXEC", persistent)
#                    self.__update_active_rworkers(rworker.id, 1)
                self.__rworkers_reserved4.append(rworker)
            logging.debug("Connected to ppserver (host=%s, port=%i) \
                    with %i workers" % (host, port, ncpus))
            self.__scheduler()
        except:
            pass
#            sys.excepthook(*sys.exc_info())

    def __connect(self):
        """Connects to all remote ppservers"""
        for ppserver in self.ppservers:
            thread.start_new_thread(self.connect1, ppserver)

        discover = ppauto.Discover(self, True)
        for ppserver in self.auto_ppservers:
            thread.start_new_thread(discover.run, ppserver)

    def __detect_ncpus(self):
        """Detects the number of effective CPUs in the system"""
        #for Linux, Unix and MacOS
        if hasattr(os, "sysconf"):
            if "SC_NPROCESSORS_ONLN" in os.sysconf_names:
                #Linux and Unix
                ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
                if isinstance(ncpus, int) and ncpus > 0:
                    return ncpus
            else:
                #MacOS X
                return int(os.popen2("sysctl -n hw.ncpu")[1].read())
        #for Windows
        if "NUMBER_OF_PROCESSORS" in os.environ:
            ncpus = int(os.environ["NUMBER_OF_PROCESSORS"])
            if ncpus > 0:
                return ncpus
        #return the default value
        return 1

    def __initlog(self, loglevel, logstream):
        """Initializes logging facility"""
        log_handler = logging.StreamHandler(logstream)
        log_handler.setLevel(loglevel)
        LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'
        log_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        self.__logger = logging.getLogger('')
        self.__logger.addHandler(log_handler)
        self.__logger.setLevel(loglevel)

    def __dumpsfunc(self, funcs, modules):
        """Serializes functions and modules"""
        hashs = hash(funcs + modules)
        if hashs not in self.__sfuncHM:
            sources = [self.__get_source(func) for func in funcs]
            self.__sfuncHM[hashs] = pickle.dumps(
                    (funcs[0].func_name, sources, modules),
                    self.__pickle_proto)
        return self.__sfuncHM[hashs]

    def __find_modules(self, prefix, dict):
        """recursively finds all the modules in dict"""
        modules = []
        for name, object in dict.items():
            if isinstance(object, types.ModuleType) \
                    and name not in ("__builtins__", "pp"):
                if object.__name__ == prefix+name or prefix == "":
                    modules.append(object.__name__)
                    modules.extend(self.__find_modules(
                            object.__name__+".", object.__dict__))
        return modules

    def __scheduler(self):
        """Schedules jobs for execution"""
        self.__queue_lock.acquire()
        while self.__queue:
            if self.__active_tasks < self.__ncpus:
                #TODO: select a job number on the basis of heuristic
                task = self.__queue.pop(0)
                for worker in self.__workers:
                    if worker.is_free:
                        worker.is_free = False
                        break
                else:
                    self.__logger.error("There are no free workers left")
                    raise RuntimeError("Error: No free workers")
                self.__add_to_active_tasks(1)
                try:
                    self.__stats["local"].njobs += 1
                    thread.start_new_thread(self.__run, task+(worker, ))
                except:
                    pass
            else:
                for rworker in self.__rworkers:
                    if rworker.is_free:
                        rworker.is_free = False
                        task = self.__queue.pop(0)
                        self.__stats[rworker.id].njobs += 1
                        thread.start_new_thread(self.__rrun, task+(rworker, ))
                        break
                else:
                    if len(self.__queue) > self.__ncpus:
                        for rworker in self.__rworkers_reserved:
                            if rworker.is_free:
                                rworker.is_free = False
                                task = self.__queue.pop(0)
                                self.__stats[rworker.id].njobs += 1
                                thread.start_new_thread(self.__rrun,
                                        task+(rworker, ))
                                break
                        else:
                            break
                            # this code will not be executed
                            # and is left for further releases
                            if len(self.__queue) > self.__ncpus*0:
                                for rworker in self.__rworkers_reserved4:
                                    if rworker.is_free:
                                        rworker.is_free = False
                                        task = self.__queue.pop(0)
                                        self.__stats[rworker.id].njobs += 1
                                        thread.start_new_thread(self.__rrun,
                                                task+(rworker, ))
                                        break
                    else:
                        break

        self.__queue_lock.release()

    def __get_source(self, func):
        """Fetches source of the function"""
        hashf = hash(func)
        if hashf not in self.__sourcesHM:
            #get lines of the source and adjust indent
            sourcelines = inspect.getsourcelines(func)[0]
            #remove indentation from the first line
            sourcelines[0] = sourcelines[0].lstrip()
            self.__sourcesHM[hashf] = "".join(sourcelines)
        return self.__sourcesHM[hashf]

    def __rrun(self, job, sfunc, sargs, rworker):
        """Runs a job remotelly"""
        self.__logger.debug("Task (remote) %i started" % (job.tid, ))

        try:
            rworker.csend(sfunc)
            rworker.send(sargs)
            sresult = rworker.receive()
            rworker.is_free = True
        except:
            self.__logger.debug("Task %i failed due to broken network " \
                    "connection - rescheduling" % (job.tid, ))
            self.insert(sfunc, sargs, job)
            self.__scheduler()
            self.__update_active_rworkers(rworker.id, -1)
            if rworker.connect("EXEC"):
                self.__update_active_rworkers(rworker.id, 1)
                self.__scheduler()
            return

        job.finalize(sresult)

        # remove the job from the waiting list
        if self.__waittasks:
            self.__waittasks_lock.acquire()
            self.__waittasks.remove(job)
            self.__waittasks_lock.release()

        self.__logger.debug("Task (remote) %i ended" % (job.tid, ))
        self.__scheduler()

    def __run(self, job, sfunc, sargs, worker):
        """Runs a job locally"""

        if self.__exiting:
            return
        self.__logger.debug("Task %i started" % (job.tid, ))

        start_time = time.time()

        try:
            worker.t.csend(sfunc)
            worker.t.send(sargs)
            sresult = worker.t.receive()
        except:
            if self.__exiting:
                return
            else:
                sys.excepthook(*sys.exc_info())

        worker.free()

        job.finalize(sresult)

        # remove the job from the waiting list
        if self.__waittasks:
            self.__waittasks_lock.acquire()
            self.__waittasks.remove(job)
            self.__waittasks_lock.release()

        self.__add_to_active_tasks(-1)
        if not self.__exiting:
            self.__stat_add_time("local", time.time()-start_time)
        self.__logger.debug("Task %i ended" % (job.tid, ))
        self.__scheduler()

    def __add_to_active_tasks(self, num):
        """Updates the number of active tasks"""
        self.__active_tasks_lock.acquire()
        self.__active_tasks += num
        self.__active_tasks_lock.release()

    def __stat_add_time(self, node, time_add):
        """Updates total runtime on the node"""
        self.__stats_lock.acquire()
        self.__stats[node].time += time_add
        self.__stats_lock.release()

    def __stat_add_job(self, node):
        """Increments job count on the node"""
        self.__stats_lock.acquire()
        self.__stats[node].njobs += 1
        self.__stats_lock.release()

    def __update_active_rworkers(self, id, count):
        """Updates list of active rworkers"""
        self.__active_rworkers_list_lock.acquire()

        if id not in self.autopp_list:
            self.autopp_list[id] = 0
        self.autopp_list[id] += count

        self.__active_rworkers_list_lock.release()

    def __gentid(self):
        """Generates a unique job ID number"""
        self.__tid += 1
        return self.__tid - 1

    def destroy(self):
        """Kills ppworkers and closes open files"""
        self.__exiting = True
        self.__queue_lock.acquire()
        self.__queue = []
        self.__queue_lock.release()

        for worker in self.__workers:
            worker.t.exiting = True
            if sys.platform.startswith("win"):
                os.popen('TASKKILL /PID '+str(worker.pid)+' /F')
            else:
                try:
                    os.kill(worker.pid, 9)
                    os.waitpid(worker.pid, 0)
                except:
                    pass

# Parallel Python Software: http://www.parallelpython.com
