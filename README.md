Add project root and python directory to PYTHONPATH:
export PYTHONPATH=[project root dir]/python:$PYTHONPATH

Copy config_template.py to config.py

dependency: pathos
if running in parallel, need pathos

need python2, numpy

System setup has been tested on Ubuntu 14.04 VM with VMWare Fusion, running on
host Mac OS X version e.g. 10.10.3