#!/usr/bin/env python3

"""
VMAF - Video Multimethod Assessment Fusion

VMAF is a perceptual video quality assessment algorithm developed by Netflix.
VMAF Development Kit (VDK) is a software package that contains the VMAF algorithm implementation,
as well as a set of tools that allows a user to train and test a custom VMAF model.
"""

import os
from distutils.core import setup
from Cython.Build import cythonize
import numpy


PYTHON_PROJECT = os.path.dirname(os.path.abspath(__file__))


def get_version():
    """Version from vmaf __init__"""
    try:
        with open(os.path.join(PYTHON_PROJECT, "vmaf", "__init__.py")) as fh:
            for line in fh:
                if line.startswith("__version__"):
                    return line.strip().rpartition(" ")[2].replace('"', "")
    except Exception:
        pass
    return "0.0-dev"

ext_module = cythonize(['vmaf/core/adm_dwt2_cy.pyx'])
ext_module[0].include_dirs = [numpy.get_include(), '../libvmaf/src']

setup(
    name="vmaf",
    version=get_version(),
    author="Zhi Li",
    author_email="zli@netflix.com",
    description="Video Multimethod Assessment Fusion",
    long_description=open(os.path.join(PYTHON_PROJECT, "README.rst")).read(),
    long_description_content_type="text/x-rst",
    url="https://github.com/Netflix/vmaf",
    packages=["vmaf", "vmaf.tools", "vmaf.core", "vmaf.script"],
    include_package_data=True,
    install_requires=[
        "numpy>=1.18.2",
        "scipy>=1.4.1",
        "matplotlib>=3.2.1",
        "pandas>=1.0.3",
        "scikit-learn>=0.22.2",
        "scikit-image>=0.16.2",
        "h5py>=2.6.0",
        "sureal>=0.4.2",
        "dill>=0.3.1",
    ],
    entry_points = {
        'console_scripts': [
            'run_cleaning_cache=vmaf.script.run_cleaning_cache:main',
            'run_psnr=vmaf.script.run_psnr:main',
            'run_result_assembly=vmaf.script.run_result_assembly:main',
            'run_testing=vmaf.script.run_testing:main',
            'run_toddnoiseclassifier=vmaf.script.run_toddnoiseclassifier:main',
            'run_vmaf=vmaf.script.run_vmaf:main',
            'run_vmaf_cross_validation=vmaf.script.run_vmaf_cross_validation:main',
            'run_vmaf_in_batch=vmaf.script.run_vmaf_in_batch:main',
            'run_vmaf_training=vmaf.script.run_vmaf_training:main',
        ],
    },
    ext_modules=ext_module,
)
