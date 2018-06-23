#!/usr/bin/env python

"""
VMAF - Video Multimethod Assessment Fusion

VMAF is a perceptual video quality assessment algorithm developed by Netflix.
VMAF Development Kit (VDK) is a software package that contains the VMAF algorithm implementation,
as well as a set of tools that allows a user to train and test a custom VMAF model.
"""

from setuptools import setup, find_packages
from setuptools.dist import Distribution


class VmafDistribution(Distribution, object):

    def __init__(self, attrs):
        attrs['version'] = self.get_version()
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


setup(
    distclass=VmafDistribution,
    name='vmaf',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    include_package_data=True,
    install_requires=[
        "numpy>=1.12.0",
        "scipy>=0.17.1",
        "matplotlib>=2.0.0",
        "pandas>=0.19.2",
        "scikit-learn>=0.18.1",
        "scikit-image>=0.13.1",
        "h5py>=2.6.0",
    ]
)
