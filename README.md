VMAF - Video Multi-Method Assessment Fusion
===================

VMAF is a software package developed by Netflix Inc. containing tools for perceptual video quality assessment.

##Prerequisite

The VMAF package has its core feature extraction library written in C, and the rest glue code written in Python. It also incorporates an external C++ library [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/). To build the C/C++ code, it requires cc and g++. To run commands, it requires Python 2 (>= 2.7) installed.

It also requires a number of Python packages:

  - [numpy](http://www.numpy.org/) (>=1.8.2)
  - [scipy](http://www.scipy.org/) (>=0.13.3)
  - [scikit-learn](http://scikit-learn.org/stable/) (==0.14.1)
  - [pandas](http://pandas.pydata.org/) (>=0.13.1)
  
To install, run (need Python package manager pip pre-installed):

`pip install numpy scipy scikit-learn==0.14.1 pandas`

Two additional packages can be installed optionally:

  - [matplotlib](http://matplotlib.org/1.3.1/index.html) (>=1.3.1) -- for result visualization
  - [pathos](https://pypi.python.org/pypi/pathos) (>=0.1a1) -- for execution parallelization

To install, run:

`pip install matplotlib pathos`

##Installation

After cloning VMAF repo to local, cd to the repo directory and run:

`make`

to build the binaries.

There is a subdirectory named python. Add the python subdirectory to the environment variable PYTHONPATH:

`export PYTHONPATH=[path_to_repo_dir]/python:$PYTHONPATH`

##Testing

The package has thus far been tested in Ubuntu 14.04 LTS and Mac OS X 10.10.5.

After installation, run:

`./unittest`

##Basic Usage

There are two basic execution modes to run VMAF -- single mode and batch mode.

To run VMAF on a single reference/distorted video pair, run:

`./run_vmaf format width height reference_path distorted_path`

where 'format' is among yuv420p, yuv422p, yuv444p (YUV 8-bit) and yuv420p10le, yuv422p10le, yuv444p10le (YUV 10-bit little endian).

For example:

`./run_vmaf yuv420p 576 324 resource/yuv/src01_hrc00_576x324.yuv resource/yuv/src01_hrc01_576x324.yuv`

To run VMAF in batch mode, create an input text file with each line of format (check examples in example_batch_input):

`format width height reference_path distorted_path`

After that, run:

`./run_vmaf_in_batch parallelize input_file`

where 'parallelize' is either 'yes' or 'no'. 

For example:

`./run_vmaf_in_batch yes example_batch_input`

##Advanced Usage

VMAF follows a machine-learning based approach to first extract a number of quality-relevant features from both a distorted video and its reference, followed by fusing the features into a final quality score using a non-linear regressor (e.g. a SVM regressor), hence the name 'Video Multi-method Assessment Fusion'.

In addition to the basic executors, the VMAF package also provides a framework to allow any user to train his/her own perceptual quality assessment model. For example, directory resource/model contains a number of pre-trained models, which can be loaded by the aforementioned VMAF executors:

`./run_vmaf format width height reference_path distorted_path [optional_VMAF_model_file]`
or
`./run_vmaf_in_batch parallelize input_file [optional_VMAF_model_file]`

For example:

`./run_vmaf yuv420p 576 324 resource/yuv/src01_hrc00_576x324.yuv resource/yuv/src01_hrc01_576x324.yuv resource/model/nflx_vmaff_rf_v1.pkl`
or 
`./run_vmaf_in_batch yes example_batch_input resource/model/nflx_vmaff_rf_v1.pkl`

A user can customize the model based on:

  - The video dataset it is trained against
  - The features selected
  - The regressor used (and its hyperparameters)
  
Once a model is trained, the VMAF package also provides tools to cross validate it on a different dataset and visualize its performance.

To begin with, create a dataset file following the format of example_dataset.py. A dataset is a collection of distorted videos, each has a unique asset ID and a corresponding reference video, identified by a unique content ID. Each distorted video is also associated with a DMOS (differential mean opinion score), which is obtained through subjective experiments.

See directory resource/dataset for more examples. Also refer to the 'Datasets' section regarding publicly available datasets.

Once a dataset is created, first test the dataset using existing VMAF or other (e.g. PSNR) metric. Run:

`./run_testing quality_type cache_result parallelize test_dataset_file_path [optional_VMAF_model_file]`

where 'quality_type' can be 'VMAF' or 'PSNR'. 'cache_result' is either 'yes' or 'no' specifying if to store the extracted feature results in a data store to be re-used later on (since feature extraction is the most expensive operations here). 'parallelize' is either 'yes' or 'no' specifying if to use parallelization in feature extraction to speed things up.

For example:

`./run_testing VMAF yes yes example_dataset.py`

Make sure matplotlib is installed to visualize the DMOS-prediction scatter plot and inspect the statistics: 
  - PCC - Pearson correlation coefficient
  - SRCC - Spearman rank correlation coefficient
  - RMSE - root mean squared error



##Datasets

(To do)