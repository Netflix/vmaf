VMAF - Video Multi-Method Assessment Fusion
===================

VMAF is a software package developed by Netflix Inc. containing tools for perceptual video quality measurement. VMAF is licensed under Apache License, Version 2.0.

The package has thus far been tested in Ubuntu 14.04 LTS.



##Dependency

The VMAF package has its core feature extraction library written in C, and the rest glue code written in Python. It also incorporates an external C++ library named libsvm. To build the C/C++ code, you system must be pre-installed with cc and g++. To run commands, it assumes that your system has Python2 installed.

To run VMAF calculation on multiple reference/distorted video pairs in parallel, you will need a Python package named pathos (>= 0.1a1). For example, run_vmaf_in_batch will first detect if pathos is installed. If not, it will fall back to sequential execution. To install pathos, in Ubuntu, run (assuming pip pre-installed):

`pip install pathos`



##Installation
After cloning repo to local, cd to repo directory and run:

`./make`

There is a subdirectory named python. Add the path to the python subdirectory to PYTHONPATH:

`export PYTHONPATH=[path_to_repo_dir]/python:$PYTHONPATH`



##Testing

After installation, run:

`./test`



##Execution

There are two execution modes to run VMAF: single mode and batch mode.

To run VMAF on a single reference/distorted video pair, run:

`./run_vmaf [format] [width] [height] [ref_video] [dis_video]`

where format is among yuv420, yuv422 and yuv444.

For example:

`./run_vmaf yuv420 576 324 resource/yuv/src01_hrc00_576x324.yuv resource/yuv/src01_hrc00_576x324.yuv`

To run VMAF in batch mode, create an input file with lines containing (check examples in example_batch_input):

`[fmt] [width] [height] [ref_file] [dis_file]\n`

Then run:

`./run_vmaf_in_batch input_file`



##To-do List

- Support for 10-bit YUV input
- Training/testing framework for model customization based on user's own video dataset