#!/usr/bin/env python

"""
VMAF - Video Multimethod Assessment Fusion

VMAF is a perceptual video quality assessment algorithm developed by Netflix.
VMAF Development Kit (VDK) is a software package that contains the VMAF algorithm implementation,
as well as a set of tools that allows a user to train and test a custom VMAF model.
"""

import os
from setuptools import setup, find_packages


PYTHON_PROJECT = os.path.dirname(os.path.abspath(__file__))
VMAF_PROJECT = os.path.dirname(PYTHON_PROJECT)


def get_version():
    """Version from project's VERSION file"""
    try:
        with open(os.path.join(VMAF_PROJECT, "VERSION")) as fh:
            for line in fh:
                if line.startswith("VMAF Development"):
                    return line.strip().rpartition(" ")[2]
    except Exception:
        pass
    return "0.0-dev"


setup(
    name="vmaf",
    version=get_version(),

    author="Zhi Li",
    author_email="zli@netflix.com",
    description="Video Multimethod Assessment Fusion",
    long_description=open(os.path.join(PYTHON_PROJECT, "README.rst")).read(),
    long_description_content_type="text/x-rst",
    url="https://github.com/Netflix/vmaf",

    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    install_requires=[
        "numpy>=1.12.0",
        "scipy>=0.17.1,<1.3.0",  # TODO python3: scipy 1.3.0 removed some previously deprecated functions, still used by vmaf
        "matplotlib>=2.0.0",
        "pandas>=0.19.2",
        "scikit-learn>=0.18.1",
        "scikit-image>=0.13.1",
        "h5py>=2.6.0",
        "sureal>=0.1.0",
    ]
)
