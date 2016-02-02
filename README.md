Add project root and python directory to PYTHONPATH:
export PYTHONPATH=[project root dir]:$PYTHONPATH

Copy config_template.py to config.py

make lib

-----------------------------------------

No need:

In config.py, set ROOT to your project root directory
ROOT = "/home/zli/Projects/stash/zli/vmaf_oss"

Add VMAF executable to PATH:
export PATH=[project root dir]/feature:$PATH

