Add project root and python directory to PYTHONPATH:
export PYTHONPATH=[project root dir]/python:$PYTHONPATH

Copy config_template.py to config.py

Optional python libraries
dependency: pathos for parallel running VMAF calculation in batch mode

System setup has been tested on Ubuntu 14.04 VM with VMWare Fusion, running on
host Mac OS X version e.g. 10.10.3

TODO:

1) Support for 10-bit YUV
1) training/testing framework