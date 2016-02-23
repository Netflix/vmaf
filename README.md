VMAF - Video Multi-Method Assessment Fusion
===================

VMAF is a software package developed by Netflix Inc. containing tools for perceptual video quality measurement.

##Prerequisite

The VMAF package has its core feature extraction library written in C, and the rest glue code written in Python. It also incorporates an external C++ library named libsvm. To build the C/C++ code, it requires cc and g++. To run commands, it requires Python 2 installed.

(Optional) To perform VMAF computation in parallel on multiple reference/distorted video pairs, you will need a Python package named [pathos](https://pypi.python.org/pypi/pathos) (>= 0.1a1). For example, run_vmaf_in_batch will execute VMAF in parallel if it detects that pathos is installed; otherwise, it will fall back to sequential execution. To install pathos, on Ubuntu, run (assuming pip pre-installed):

`pip install pathos`

(Optional) The VMAF training/testing framework requires a Python data processing package named [pandas](http://pandas.pydata.org/) (>= 0.13.1). To install pandas, on Ubuntu, run:

`pip install pandas`

##Installation
After cloning VMAF repo to local, cd to the repo directory and run:

`make`

There is a subdirectory named python. Add the python subdirectory to PYTHONPATH:

`export PYTHONPATH=[path_to_repo_dir]/python:$PYTHONPATH`

##Testing

The package has thus far been tested on Ubuntu 14.04 LTS and Mac OS X 10.10.5.

After installation, run:

`./test`

##Execution

There are two execution modes to run VMAF -- single mode and batch mode.

To run VMAF on a single reference/distorted video pair, run:

`./run_vmaf [format] [width] [height] [ref_video] [dis_video]`

where format is among yuv420p, yuv422p, yuv444p (YUV 8-bit) and yuv420p10le, yuv422p10le, yuv444p10le (YUV 10-bit little endian).

For example:

`./run_vmaf yuv420p 576 324 resource/yuv/src01_hrc00_576x324.yuv resource/yuv/src01_hrc01_576x324.yuv`

To run VMAF in batch mode, create an input text file with each line of format (check examples in example_batch_input):

`[fmt] [width] [height] [ref_file] [dis_file]`

After that, run:

`./run_vmaf_in_batch input_file`

For example:

`./run_vmaf_in_batch example_batch_input`

##To-do List

- Training/testing framework for customization based on user's own video dataset