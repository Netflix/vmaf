#!/usr/bin/env python

"""
VMAF - Video Multimethod Assessment Fusion

VMAF is a perceptual video quality assessment algorithm developed by Netflix.
VMAF Development Kit (VDK) is a software package that contains the VMAF algorithm implementation,
as well as a set of tools that allows a user to train and test a custom VMAF model.
"""

from __future__ import print_function

import os
import sys
import pkg_resources
import platform

from setuptools import setup, find_packages, Command
from setuptools.command.install_egg_info import install_egg_info as _install_egg_info
from setuptools.dist import Distribution


DOCLINES = __doc__.strip().split("\n")
os.chdir(os.path.dirname(os.path.abspath(__file__)))        # setup.py should be ran from its folder, just in case one tries to execute it from a different folder

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: Apache License
Programming Language :: C
Programming Language :: Python
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Programming Language :: Python :: 3.6
Topic :: Software Development
Topic :: Scientific/Engineering
Operating System :: Microsoft :: Windows
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
"""


class EntryPoints(Command):
    description = 'get entrypoints for a distribution'
    user_options = [
        ('dist=', None, 'get entrypoints for specified distribution'),
    ]

    def initialize_options(self):
        self.dist = self.distribution.get_name()

    def finalize_options(self):
        """ This command doesn't need to finalize its options """

    def run(self):
        req_entry_points = pkg_resources.get_entry_map(self.dist)
        if req_entry_points and 'console_scripts' in req_entry_points:
            for entry in list(req_entry_points['console_scripts'].values()):
                print(entry, file=sys.stdout)


class install_egg_info(_install_egg_info):  # noqa
    """Override the setuptools namespace package templates.
    Customizes the "nspkg.pth" files so that they're compatible with "--editable" packages.
    See this pip issue for details: https://github.com/pypa/pip/issues/3 (modifications to the original implementation are marked with CHANGED)
    """
    _nspkg_tmpl = (
        # CHANGED: Add the import of pkgutil needed on the last line.
        "import sys, types, os, pkgutil",
        "p = os.path.join(sys._getframe(1).f_locals['sitedir'], *%(pth)r)",
        "ie = os.path.exists(os.path.join(p, '__init__.py'))",
        "m = not ie and "
        "sys.modules.setdefault(%(pkg)r, types.ModuleType(%(pkg)r))",
        "mp = (m or []) and m.__dict__.setdefault('__path__', [])",
        "(p not in mp) and mp.append(p)",
        # CHANGED: Fix the resulting __path__ on the namespace packages to
        # properly traverse "--editable" packages too.
        "mp[:] = m and pkgutil.extend_path(mp, %(pkg)r) or mp",
    )
    "lines for the namespace installer"

    _nspkg_tmpl_multi = (
        # CHANGED: Use "__import__" to ensure the parent package has been
        # loaded before attempting to read it from sys.modules.
        # This avoids a possible issue with nested namespace packages where the
        # parent could be skipped due to an existing __init__.py file.
        'm and __import__(%(parent)r) and setattr(sys.modules[%(parent)r], %(child)r, m)',
    )
    "additional line(s) when a parent package is indicated"


class VmafDistribution(Distribution, object):

    PINNED_TXT = 'pinned.txt'

    excluded_platform_packages = {}

    def __init__(self, attrs):
        attrs['version'] = self.get_version()
        attrs['install_requires'] = list(self.load_pinned_deps())
        super(VmafDistribution, self).__init__(attrs)

    def get_version(self):
        """ Return version from project's VERSION file """
        try:
            with open('../VERSION') as fh:
                for line in fh:
                    if line.startswith('VMAF Development'):
                        return line.strip().rpartition(' ')[2]
        except Exception:
            pass

        return '0.0-dev'

    def get_command_class(self, command):
        """Return a customized command class or the base one."""
        if command == 'install_egg_info':
            return install_egg_info
        elif command == 'entrypoints':
            return EntryPoints

        return super(VmafDistribution, self).get_command_class(command)

    @property
    def excluded_packages(self):
        platform_name = platform.system().lower()
        if platform_name in self.excluded_platform_packages:
            return set(pkg.lower() for pkg in self.excluded_platform_packages[platform_name])
        return set()

    def load_pinned_deps(self):
        """Load a pinned.txt file.

        The pinned.txt file contains a list of dependencies that this Python
        project depends on. Although the PyGradle build system will ignore this
        file and never install dependencies declared via this method, it is
        important to declare the dependencies using this method to maintain
        backwards compatibility with non-PyGradle build systems.

        """
        # calculate this only once
        blacklisted = self.excluded_packages
        try:
            reqs = []
            with open(self.PINNED_TXT) as fh:
                reqs = fh.readlines()
            # Don't include the version information so that we don't mistakenly
            # introduce a version conflict issue.
            for req in reqs:
                if req:
                    name, version = req.split('==')
                    if name and name.lower() not in blacklisted:
                        yield name
        except IOError:
            raise StopIteration

setup(
    distclass=VmafDistribution,

    name='vmaf',
    description=DOCLINES[0],
    long_description='\n'.join(DOCLINES[2:]),
    author='Zhi Li',
    author_email='zli@netflix.com',
    license='Apache License',
    url="http://github.com/netflix/vmaf",
    download_url='https://github.com/scipy/scipy/releases',

    classifiers=[line for line in CLASSIFIERS.split('\n') if line],
    platforms=["Windows", "Linux", "Mac OS-X", "Unix"],

    package_dir={'': 'src'},
    packages=find_packages('src'),
    include_package_data=True,
)
