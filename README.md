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

To start, install *pip* - the Python package manager. (It is also recommended to start with a clean Python environment using *virtualenv* with the *--no-site-package* option.) After that, run:

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

(You can also add it to environment permanently. On Ubuntu, append the line above to ~/.bashrc and 'source ~/.bashrc'. On Mac OS X, append it to ~/.profile and source.)

##Testing

The package has thus far been tested in Ubuntu 14.04 LTS and Mac OS X 10.10.5.

After installation, run:

`./unittest`

##Basic Usage

There are two basic execution modes to run VMAF -- a single mode and a batch mode.

To run VMAF on a single reference/distorted video pair, run:

`./run_vmaf format width height reference_path distorted_path`

where 'format' is among yuv420p, yuv422p, yuv444p (YUV 8-bit) and yuv420p10le, yuv422p10le, yuv444p10le (YUV 10-bit little endian).

For example:

`./run_vmaf yuv420p 576 324 resource/yuv/src01_hrc00_576x324.yuv resource/yuv/src01_hrc01_576x324.yuv`

To run VMAF in batch mode, create an input text file with each line of format (check examples in example_batch_input):

`format width height reference_path distorted_path`

For example:

`yuv420p 576 324 resource/yuv/src01_hrc00_576x324.yuv resource/yuv/src01_hrc01_576x324.yuv`

After that, run:

`./run_vmaf_in_batch parallelize input_file`

where 'parallelize' is either 'yes' or 'no'. 

For example:

`./run_vmaf_in_batch yes example_batch_input`

##Advanced Usage

VMAF follows a machine-learning based approach to first extract a number of quality-relevant features from both a distorted video and its reference full-quality video, followed by fusing them into a final quality score using a non-linear regressor (e.g. a SVM regressor), hence the name 'Video Multi-method Assessment Fusion'.

In addition to the basic executors, the VMAF package also provides a framework to allow any user to train his/her own perceptual quality assessment model. For example, directory *resource/model* contains a number of pre-trained models, which can be loaded by the aforementioned VMAF executors:

`./run_vmaf format width height reference_path distorted_path [optional_VMAF_model_file]`

or

`./run_vmaf_in_batch parallelize input_file [optional_VMAF_model_file]`

For example:

`./run_vmaf yuv420p 576 324 resource/yuv/src01_hrc00_576x324.yuv resource/yuv/src01_hrc01_576x324.yuv resource/model/nflx_vmaff_rf_v1.pkl`

or 

`./run_vmaf_in_batch yes example_batch_input resource/model/nflx_vmaff_rf_v1.pkl`

A user can customize the model based on:

  - The video dataset it is trained on
  - The list of features used
  - The regressor used (and its hyper-parameters)
  
Once a model is trained, the VMAF package also provides tools to cross validate it on a different dataset and visualization.

###Create a Dataset

To begin with, create a dataset file following the format of example_dataset.py. A dataset is a collection of distorted videos, each has a unique asset ID and a corresponding reference video, identified by a unique content ID. Each distorted video is also associated with a DMOS (differential mean opinion score), which is obtained through subjective experiments.

See directory *resource/dataset* for more examples. Also refer to the 'Datasets' section regarding publicly available datasets.

###Validate a Dataset

Once a dataset is created, first validate the dataset using existing VMAF or other (e.g. PSNR) metrics. Run:

`./run_testing quality_type cache_result parallelize test_dataset_file [optional_VMAF_model_file]`

where 'quality_type' can be 'VMAF' or 'PSNR'. 

'cache_result' is either 'yes' or 'no' specifying if to store the extracted feature results in a data store to be re-used later on (since feature extraction is the most expensive operations here). 

'parallelize' is either 'yes' or 'no' specifying if to parallelize feature extraction to speed things up. Sometimes it is desirable to disable parallelization for debugging purpose (e.g. some error messages can only be displayed when parallelization is disabled).

For example:

`./run_testing VMAF yes yes example_dataset.py`

Make sure matplotlib is installed to visualize the DMOS-prediction scatter plot and inspect the statistics: 

  - PCC - Pearson correlation coefficient
  - SRCC - Spearman rank correlation coefficient
  - RMSE - root mean squared error

###Train a New Model

Now that we are confident that the dataset is created correctly and we have some benchmark result on existing metrics, we proceed to train a new quality assessment model. Run:

`./run_training cache_result parallelize train_dataset_file feature_param_file model_param_file output_model_file`

Here 'cache_result' is either 'yes' or 'no', 'parallelize' is either 'yes' or 'no', similar as before. 

For example:

`./run_training yes yes example_dataset.py resource/feature_param/vmaf_feature_v1.py resource/model_param/libsvmnusvr_v1.py workspace/model/test_model.pkl`

'feature_param_file' defines the set of features used. For example, both dictionaries

`feature_dict = {'VMAF_feature':'all', }`

and

`feature_dict = {'VMAF_feature':['vif', 'adm'], }`

are valid specifications of selected features. Here 'VMAF_feature' is an 'aggregate' feature type, and 'vif', 'adm' are the 'atomic' feature types within the aggregate type. In the first case, 'all' specifies that all atomic features of 'VMAF_feature' are selected. A feature_dict dictionary can also contain more than one aggregate feature types.

'model_param_file' defines the type and hyper-parameters of the regressor to be used. For details, refer to the self-explanatory examples in directory *resource/model_param*.

The trained model is output to 'output_model_file'. Once it is obtained, it can be used by the *run_vmaf* or *run_vmaf_in_batch* executors, or used by *run_test* to validate another dataset.

![training scatter](/resource/images/scatter_training.png)
![testing scatter](/resource/images/scatter_testing.png)

Above are two example scatter plots obtained from running the *run_traing* and *run_testing* executors on a training and a testing dataset, respectively.


##Datasets

To be completed...