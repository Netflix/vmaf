VMAF - Video Multi-Method Assessment Fusion
===================

VMAF is a perceptual video quality assessment algorithm developed by Netflix Inc. VMAF Development Kit (VDK) is a software package that contains the VMAF algorithm implementation, as well as a set of tools that allows a user to train and test a custom VMAF model. Read [this](http://techblog.netflix.com/2016/06/toward-practical-perceptual-video.html) tech blog post for an overview.

##What's New

- (9/30/2016) Added *vmafossexec* -- a Python-independent C++ implementation under [wrapper/](wrapper). Refer to [Python-independent Implementation](#Python-independent-implementation) for details.
- (9/30/2016) Commands *run_vmaf*, *run_vmaf_in_batch* and *vmafossexec* all support outputing result in XML and JSON format.

##Prerequisite

The VDK package has its core feature extraction library written in C, and the rest glue code written in Python. It also incorporates an external C++ library [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/). To build the C/C++ code, it requires *cc* and *g++*. To run scripts and tests, it requires Python2 (>= 2.7) installed.

It also requires a number of Python packages:

  - [numpy](http://www.numpy.org/) (>=1.10.4)
  - [scipy](http://www.scipy.org/) (>=0.17.0)
  - [matplotlib](http://matplotlib.org/1.3.1/index.html) (>=1.5.1)
  - [pandas](http://pandas.pydata.org/) (>=0.17.1)
  - [scikit-learn](http://scikit-learn.org/stable/) (>=0.17.1)
  - [h5py](http://www.h5py.org/) (>=2.2.1)

Follow [this link](http://www.scipy.org/install.html) to install the *numpy/scipy/matplotlib/pandas* suite on your system. To install *scikit-learn*, first [install](http://python-packaging-user-guide.readthedocs.org/en/latest/installing/) package manager *pip*, then run:

```
sudo pip install --upgrade scikit-learn
```

To install *h5py*, run:

```
sudo pip install --upgrade h5py
```

####Troubleshooting

You can verify if these packages are properly installed and its version/location by:

```
python -c 'import numpy as pkg; print pkg.__version__; print pkg.__file__'
python -c 'import scipy as pkg; print pkg.__version__; print pkg.__file__'
python -c 'import matplotlib as pkg; print pkg.__version__; print pkg.__file__'
python -c 'import pandas as pkg; print pkg.__version__; print pkg.__file__'
python -c 'import sklearn as pkg; print pkg.__version__; print pkg.__file__'
python -c 'import h5py as pkg; print pkg.__version__; print pkg.__file__'
```

If you see that the printed version number is older than the ones aforementioned, it could suggest that a previously installed package with the same name but older version at a different location may have overshadowed the new one. Make sure that the new one's path appears early in the path list, which can be printed by:

```
python -c 'import sys; print sys.path'
```

(Or simply delete the older one).

##Installation

After cloning VMAF repo to local, cd to the repo directory and run:

```
make
```

to build the binaries.

There is a subdirectory named *python*. Add the *python* subdirectory to the environment variable PYTHONPATH:

```
export PYTHONPATH=[path_to_repo_dir]/python:$PYTHONPATH
```

You can also add it to environment permanently. On Ubuntu, append the line above to *~/.bashrc* and run `source ~/.bashrc`. On Mac OS X, append it to *~/.bash_profile* and run `source ~/.bash_profile`.

##Testing

The package has thus far been tested on Ubuntu 14.04 LTS and Mac OS X 10.10.5.

After installation, run:

```
./unittest
```

##Basic Usage

There are two basic execution modes to run VMAF -- a single mode and a batch mode.

To run VMAF on a single reference/distorted video pair, run:

```
./run_vmaf format width height reference_path distorted_path
```

where *format* is among *yuv420p*, *yuv422p*, *yuv444p* (YUV 8-bit) and *yuv420p10le*, *yuv422p10le*, *yuv444p10le* (YUV 10-bit little endian).

For example:

```
./run_vmaf yuv420p 576 324 resource/yuv/src01_hrc00_576x324.yuv resource/yuv/src01_hrc01_576x324.yuv
```

This will generate output like:

```
Aggregate: VMAF_feature_adm2_score:0.925, VMAF_feature_motion_score:3.592, VMAF_feature_vif_scale0_score:0.366, VMAF_feature_vif_scale1_score:0.772, VMAF_feature_vif_scale2_score:0.868, VMAF_feature_vif_scale3_score:0.921, VMAF_score:66.628
```

where *VMAF_score* is the final score and the others are the scores for elementary metrics. *adm2*, *vif_scalex* scores range from 0 (worst) to 1 (best), and *motion* score typically ranges from 0 (static) to 20 (high-motion).

To run VMAF in batch mode, create an input text file with each line of format (check examples in [example_batch_input](example_batch_input)):

```
format width height reference_path distorted_path
```

For example:

```
yuv420p 576 324 resource/yuv/src01_hrc00_576x324.yuv resource/yuv/src01_hrc01_576x324.yuv
yuv420p 576 324 resource/yuv/src01_hrc00_576x324.yuv resource/yuv/src01_hrc00_576x324.yuv
```

After that, run:

```
./run_vmaf_in_batch parallelize input_file
```

where *parallelize* is either *yes* or *no*.

For example:

```
./run_vmaf_in_batch yes example_batch_input
```

##Advanced Usage

VMAF follows a machine-learning based approach to first extract a number of quality-relevant features from a distorted video and its reference full-quality video, followed by fusing them into a final quality score using a non-linear regressor (e.g. a SVM regressor), hence the name 'Video Multi-method Assessment Fusion'.

In addition to the basic executors, the VMAF package also provides a framework to allow any user to train his/her own perceptual quality assessment model. For example, directory [resource/model](resource/model) contains a number of pre-trained models, which can be loaded by the aforementioned VMAF executors:

```
./run_vmaf format width height reference_path distorted_path [optional_VMAF_model_file]
./run_vmaf_in_batch parallelize input_file [optional_VMAF_model_file]
```

For example:

```
./run_vmaf yuv420p 576 324 resource/yuv/src01_hrc00_576x324.yuv resource/yuv/src01_hrc01_576x324.yuv resource/model/nflxtrain_vmafv3.pkl
./run_vmaf_in_batch yes example_batch_input resource/model/nflxtrain_vmafv3.pkl
```

A user can customize the model based on:

  - The video dataset it is trained on
  - The list of features used
  - The regressor used (and its hyper-parameters)
  
Once a model is trained, the VMAF package also provides tools to cross validate it on a different dataset and visualization.

###Create a Dataset

To begin with, create a dataset file following the format in [example_dataset.py](example_dataset.py). A dataset is a collection of distorted videos, each has a unique asset ID and a corresponding reference video, identified by a unique content ID. Each distorted video is also associated with a MOS (mean opinion score), or differential MOS (DMOS), which is obtained through subjective experiments. An example code snippets that defines a dataset is as follows:

```
dataset_name = 'example'
yuv_fmt = 'yuv420p'
width = 1920
height = 1080
ref_videos = [
    {'content_id':0, 'path':'checkerboard.yuv'},
    {'content_id':1, 'path':'flat.yuv'},
]
dis_videos = [
    {'content_id':0, 'asset_id': 0, 'dmos':100, 'path':'checkerboard.yuv'}, # ref
    {'content_id':0, 'asset_id': 1, 'dmos':50,  'path':'checkerboard_dis.yuv'},
    {'content_id':1, 'asset_id': 2, 'dmos':100,  'path':'flat.yuv'}, # ref
    {'content_id':1, 'asset_id': 3, 'dmos':80,  'path':'flat_dis.yuv'},
]
```

See directory [resource/dataset](resource/dataset) for more examples. Also refer to the [Datasets](#datasets) section regarding publicly available datasets.

###Validate a Dataset

Once a dataset is created, first validate the dataset using existing VMAF or other (e.g. PSNR) metrics. Run:

```
./run_testing quality_type cache_result parallelize test_dataset_file [optional_VMAF_model_file]
```

where *quality_type* can be *VMAF* or *PSNR*. 

*cache_result* is either *yes* or *no* specifying if to store the extracted feature results in a data store to be re-used later on (since feature extraction is the most expensive operations here). 

*parallelize* is either *yes* or *no* specifying if to parallelize feature extraction to speed things up. Sometimes it is desirable to disable parallelization for debugging purpose (e.g. some error messages can only be displayed when parallel execution is disabled).

For example:

```
./run_testing VMAF yes yes example_dataset.py
```

Make sure *matplotlib* is installed to visualize the MOS-prediction scatter plot and inspect the statistics: 

  - PCC - Pearson correlation coefficient
  - SRCC - Spearman rank correlation coefficient
  - RMSE - root mean squared error
  
####Troubleshooting

When creating a dataset file, one may make errors (for example, having a typo in a file path) that could go unnoticed but make the execution of `run_testing` fail. For debugging purpose, it is recommended to turn *parallelize* mode to *no*. 

If problem persists, one may need to run the script: 

```
./run_cleaning_cache quality_type test_dataset_file
``` 

to clean up corrupted results in the store before retrying. For example:
 
```
./run_cleaning_cache VMAF example_dataset.py
```

###Train a New Model

Now that we are confident that the dataset is created correctly and we have some benchmark result on existing metrics, we proceed to train a new quality assessment model. Run:

```
./run_training cache_result parallelize train_dataset_file feature_param_file model_param_file output_model_file
```

Here *cache_result* is either *yes* or *no*, *parallelize* is either *yes* or *no*, similar as before. 

For example:

```
./run_training yes yes example_dataset.py resource/feature_param/vmaf_feature_v2.py resource/model_param/libsvmnusvr_v2.py workspace/model/test_model.pkl
```

*feature_param_file* defines the set of features used. For example, both dictionaries below:

```
feature_dict = {'VMAF_feature':'all', }
```

and

```
feature_dict = {'VMAF_feature':['vif', 'adm'], }
```

are valid specifications of selected features. Here *VMAF_feature* is an 'aggregate' feature type, and *vif*, *adm* are the 'atomic' feature types within the aggregate type. In the first case, *all* specifies that all atomic features of *VMAF_feature* are selected. A feature_dict dictionary can also contain more than one aggregate feature types.

*model_param_file* defines the type and hyper-parameters of the regressor to be used. For details, refer to the self-explanatory examples in directory *resource/model_param*. One example is:

```
model_type = "LIBSVMNUSVR"
model_param_dict = {
    # ==== preprocess: normalize each feature ==== #
    'norm_type':'clip_0to1', # rescale to within [0, 1]
    # ==== postprocess: clip final quality score ==== #
    'score_clip':[0.0, 100.0], # clip to within [0, 100]
    # ==== libsvmnusvr parameters ==== #
    'gamma':0.85, # selected
    'C':1.0, # default
    'nu':0.5, # default
    'cache_size':200 # default
}
```

The trained model is output to *output_model_file*. Once it is obtained, it can be used by the *run_vmaf* or *run_vmaf_in_batch*, or used by *run_test* to validate another dataset.

![training scatter](/resource/images/scatter_training.png)
![testing scatter](/resource/images/scatter_testing.png)

Above are two example scatter plots obtained from running the *run_training* and *run_testing* executors on a training and a testing dataset, respectively.

###Cross Validation

[python/script/run_vmaf_cross_validation.py](python/script/run_vmaf_cross_validation.py) provides tools for cross validation of hyper-parameters and models. *run_vmaf_cv* runs training on a training dataset using hyper-parameters specified in a parameter file, output a trained model file, and then test the trained model on another test dataset and report testing correlation scores. 

*run_vmaf_kfold_cv* takes in a dataset file, a parameter file, and a data structure (list of lists) that specifies the folds based on video content's IDs, and run k-fold cross valiation on the video dataset. This can be useful for manually tuning the model parameters.

###Creating New Features And Regressors

You can also customize VMAF by plugging in third-party features or inventing new features, and specify them in a *feature_param_file*. Essentially, the 'aggregate' feature type (e.g. VMAF_feature) specified in the *feature_dict* corresponds to the *TYPE* field of a FeatureExtractor subclass (e.g. VmafFeatureExtractor). All you need to do is to create a new class extending the FeatureExtractor base class.

Similarly, you can plug in a third-party regressor or invent a new regressor and specify them in a *model_param_file*. The *model_type* (e.g. LIBSVMNUSVR) corresponds to the *TYPE* field of a TrainTestModel sublass (e.g. LibsvmnusvrTrainTestModel). All needed is to create a new class extending the TrainTestModel base class.

For instructions on how to extending the FeatureExtractor and TrainTestModel base classes, refer to [CONTRIBUTING.md](CONTRIBUTING.md).

##Datasets

### Netflix Public Dataset

We provide a dataset publicly available to the community for training, testing and verification of results purposes. The dataset file is at [resource/dataset/NFLX_dataset_public.py](resource/dataset/NFLX_dataset_public.py), and the videos (in YUV420P format) can be downloaded [here](https://drive.google.com/folderview?id=0B3YWNICYMBIweGdJbERlUG9zc0k&usp=sharing).

### VQEG HD3 Dataset

We also provide an example dataset file containing video file names from VQEG (Video Quality Expert Group) HD3 videos. The dataset file is at [resource/dataset/VQEGHD3_dataset.py](resource/dataset/VQEGHD3_dataset.py), and the videos is available for downloading from [http://www.cdvl.org/](http://www.cdvl.org/). After login, choose menu 'find videos', and search use keyword 'vqeghd3'. The dataset file includes from src01 to src09 except for src04, which overlaps with the Netflix Public Dataset, and hrc04, hrc07, hrc16, hrc17, hrc18, hrc19, hrc20 and hrc21, which are the mostly relevant distortion types to adaptive streaming. After downloading the videos, convert them to YUV420P format. 

##Python-independent Implementation

We have recently added a C++ implementation that has zero Python-dependency under [wrapper/](wrapper).