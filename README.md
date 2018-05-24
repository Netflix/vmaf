VMAF - Video Multi-Method Assessment Fusion
===================
[![Build Status](https://travis-ci.org/Netflix/vmaf.svg?branch=master)](https://travis-ci.org/Netflix/vmaf)

VMAF is a perceptual video quality assessment algorithm developed by Netflix. VMAF Development Kit (VDK) is a software package that contains the VMAF algorithm implementation, as well as a set of tools that allows a user to train and test a custom VMAF model. For an overview, read [this](http://techblog.netflix.com/2016/06/toward-practical-perceptual-video.html) tech blog post, or [this](resource/doc/VMAF_ICIP17.pdf) slide deck.

## What's New

- (5/23/18) Added multi-threading to `vmafossexec`.
- (1/20/18) Moved custom subjective models into a submodule named [sureal](https://github.com/Netflix/sureal). If you pull the latest changes, you will have to pull the submoddule by `git submodule update --init --recursive` and add `sureal/python/src` to `PYTHONPATH`.
- (8/12/17) VMAF is now included as a filter in [FFmpeg](http://ffmpeg.org/) main branch, and can be configured using: `./configure --enable-libvmaf`.
- (7/16/17) VMAF is now packaged into a library call `libvmaf` and can be called from a C/C++ program directly. See [this](#usage-through-libvmaf) section for details.

## Frequently Asked Questions

Refer to the [FAQ](FAQ.md) page.

## Prerequisites

The VDK package has its core feature extraction library written in C, and the rest scripting code written in Python. It also has a stand-alone C++ implementation that is Python-independent. To build the C/C++ code, it requires `gcc` and `g++` (>=4.8). To run scripts and tests, it requires Python2 (>= 2.7) installed.

It also requires a number of Python packages:

  - [`numpy`](http://www.numpy.org/) (>=1.12.0)
  - [`scipy`](http://www.scipy.org/) (>=0.17.1)
  - [`matplotlib`](http://matplotlib.org/1.3.1/index.html) (>=2.0.0)
  - [`pandas`](http://pandas.pydata.org/) (>=0.19.2)
  - [`scikit-learn`](http://scikit-learn.org/stable/) (>=0.18.1)
  - [`scikit-image`](http://scikit-image.org/) (>=0.13.1)
  - [`h5py`](http://www.h5py.org/) (>=2.6.0)

You will need to install `gfortran` for compiling `scipy`, `freetype` and `pkg-config` required by `matplotlib`, and `hdf5` required by `h5py` (C header files needed). These can't be compiled from source here.

### Prerequisite Installation

#### Linux (Ubuntu)

Install the dependencies:

```
sudo apt-get update -qq && \
sudo apt-get install -y \
  pkg-config gfortran libhdf5-dev libfreetype6-dev liblapack-dev \
  python python-setuptools python-dev python-pip python-tk
```

Upgrade `pip` to the newest version:

```
sudo -H pip install --upgrade pip
```

Then install the required Python packages:

```
pip install --user numpy scipy matplotlib notebook pandas sympy nose scikit-learn scikit-image h5py
```

Make sure your user install executable directory is on your PATH. Add this to the end of `~/.bashrc` and restart your shell:

```
export PATH="$PATH:$HOME/.local/bin"
```

#### macOS

First, install [Homebrew](https://brew.sh), then install the dependencies:

```
brew install gcc freetype pkg-config homebrew/science/hdf5 python
```

This will install an up-to-date version of Python and `pip`, but you must add the following to the end of `~/.bash_profile` and restart your shell:

```
export PATH="/usr/local/opt/python/libexec/bin:$PATH"
```

Now install the required Python packages:

```
brew install numpy scipy
pip install matplotlib notebook pandas sympy nose scikit-learn scikit-image h5py
```

### Troubleshooting

You can verify if these packages are properly installed and its version/location by:

```
python -c 'import numpy as pkg; print(pkg.__version__); print(pkg.__file__)'
python -c 'import scipy as pkg; print(pkg.__version__); print(pkg.__file__)'
python -c 'import matplotlib as pkg; print(pkg.__version__); print(pkg.__file__)'
python -c 'import pandas as pkg; print(pkg.__version__); print(pkg.__file__)'
python -c 'import sklearn as pkg; print(pkg.__version__); print(pkg.__file__)'
python -c 'import skimage as pkg; print(pkg.__version__); print(pkg.__file__)'
python -c 'import h5py as pkg; print(pkg.__version__); print(pkg.__file__)'
```

If you see that the printed version number is older than the ones aforementioned, it could suggest that a previously installed package with the same name but older version at a different location may have overshadowed the new one. Make sure that the new one's path appears early in the path list, which can be printed by:

```
python -c 'import sys; print(sys.path)'
```

(Or simply delete the older one).

## Installation

First, pull submodule `sureal` by running:

```
git submodule update --init --recursive
```

After cloning VMAF repository, `cd` to the repo directory and run:

```
make
```

to build the binaries.

Add the `python/src` and `sureal/python/src` subdirectories to the environment variable `PYTHONPATH`:

```
export PYTHONPATH="$(pwd)/python/src:$(pwd)/sureal/python/src:$PYTHONPATH"
```

You can also add it to the environment permanently, by appending to `~/.bashrc`:

```
echo export PYTHONPATH="$(pwd)/python/src:$(pwd)/sureal/python/src:$PYTHONPATH" >> ~/.bashrc
source ~/.bashrc
```

Under macOS, use `~/.bash_profile` instead.

## Testing

The package has thus far been tested on Ubuntu 16.04 LTS and macOS 10.13.

After installation, run:

```
./unittest
```

## Basic Usage

There are two basic execution modes to run VMAF – a single mode and a batch mode.

### Running in Single Mode

To run VMAF on a single reference/distorted video pair, run:

```
./run_vmaf format width height reference_path distorted_path [--out-fmt output_format]
```

The arguments are the following:

- `format` can be one of:
    - `yuv420p`, `yuv422p`, `yuv444p` (8-Bit YUV)
    - `yuv420p10le`, `yuv422p10le`, `yuv444p10le` (10-Bit little-endian YUV)
- `width` and `height` are the width and height of the videos, in pixels
- `reference_path` and `distorted_path` are the paths to the reference and distorted video files
- `output_format` can be one of:
    - `text`
    - `xml`
    - `json`

For example:

```
./run_vmaf yuv420p 576 324 \
  python/test/resource/yuv/src01_hrc00_576x324.yuv \
  python/test/resource/yuv/src01_hrc01_576x324.yuv \
  --out-fmt json
```

This will generate JSON output like:

```
"aggregate": {
    "VMAF_feature_adm2_score": 0.92542107502749982, 
    "VMAF_feature_motion2_score": 4.0498253541666669, 
    "VMAF_feature_vif_scale0_score": 0.36342048943884936, 
    "VMAF_feature_vif_scale1_score": 0.76664754213485187, 
    "VMAF_feature_vif_scale2_score": 0.86285466690193247, 
    "VMAF_feature_vif_scale3_score": 0.91597177803640772, 
    "VMAF_score": 65.44885887590759, 
    "method": "mean"
}
```

where `VMAF_score` is the final score and the others are the scores for VMAF's elementary metrics.

- `adm2`, `vif_scalex` scores range from 0 (worst) to 1 (best)
- `motion2` score typically ranges from 0 (static) to 20 (high-motion)

### Using `ffmpeg2vmaf`

There is also an `ffmpeg2vmaf` script which can compare any file format supported by `ffmpeg` (note that you need a recent version of `ffmpeg` installed):

```
./ffmpeg2vmaf width height reference_path distorted_path [--out-fmt output_format]
```

### Running in Batch Mode

To run VMAF in batch mode, create an input text file, where each corresponds to the following format (check examples in [example_batch_input](example_batch_input)):

```
format width height reference_path distorted_path
```

For example:

```
yuv420p 576 324 python/test/resource/yuv/src01_hrc00_576x324.yuv python/test/resource/yuv/src01_hrc01_576x324.yuv
yuv420p 576 324 python/test/resource/yuv/src01_hrc00_576x324.yuv python/test/resource/yuv/src01_hrc00_576x324.yuv
```

After that, run:

```
./run_vmaf_in_batch input_file [--out-fmt out_fmt] [--parallelize]
```

where enabling `--parallelize` allows execution on multiple reference-distorted video pairs in parallel.

For example:

```
./run_vmaf_in_batch example_batch_input --parallelize
```

### Predict Quality on a Cellular Phone Screen

VMAF v0.6.1 and later support a custom quality model for cellular phone screen viewing. This model can be invoked by adding `--phone-model` option in the commands `run_vmaf`, `run_vmaf_in_batch` (but also in `run_testing` and `vmafossexec` which are introduced the following sections):

```
./run_vmaf yuv420p 576 324 \
  python/test/resource/yuv/src01_hrc00_576x324.yuv \
  python/test/resource/yuv/src01_hrc01_576x324.yuv \
  --phone-model

./run_vmaf_in_batch example_batch_input --parallelize --phone-model
```

Invoking the phone model will generate VMAF scores higher than in the regular model, which is more suitable for laptop, TV, etc. viewing conditions. An example VMAF–bitrate relationship for the two models is shown below:

![regular vs phone model](/resource/images/phone_model.png)

From the figure it can be interpreted that due to the factors of screen size and viewing distance, the same distorted video would be perceived as having a higher quality when viewed on a phone screen than on a laptop/TV screen, and when the quality score reaches its maximum (100), further increasing the encoding bitrate would not result in any perceptual improvement in quality.

## Advanced Usage

VMAF follows a machine-learning based approach to first extract a number of quality-relevant features (or elementary metrics) from a distorted video and its reference full-quality video, followed by fusing them into a final quality score using a non-linear regressor (e.g. an SVM regressor), hence the name “Video Multi-method Assessment Fusion”.

In addition to the basic commands, the VMAF package also provides a framework to allow any user to train his/her own perceptual quality assessment model. For example, directory [`resource/model`](resource/model) contains a number of pre-trained models, which can be loaded by the aforementioned commands:

```
./run_vmaf format width height reference_path distorted_path [--model model_path]
./run_vmaf_in_batch input_file [--model model_path] --parallelize
```

For example:

```
./run_vmaf yuv420p 576 324 \
  python/test/resource/yuv/src01_hrc00_576x324.yuv \
  python/test/resource/yuv/src01_hrc01_576x324.yuv \
  --model model/nflxtrain_vmafv3.pkl

./run_vmaf_in_batch example_batch_input --model model/nflxtrain_vmafv3.pkl --parallelize
```

A user can customize the model based on:

- The video dataset it is trained on
- The list of features used
- The regressor used (and its hyper-parameters)

Once a model is trained, the VMAF package also provides tools to cross validate it on a different dataset and visualization.

### Create a Dataset

To begin with, create a dataset file following the format in [`example_dataset.py`](example_dataset.py). A dataset is a collection of distorted videos. Each has a unique asset ID and a corresponding reference video, identified by a unique content ID. Each distorted video is also associated with subjective quality score, typically a MOS (mean opinion score), obtained through subjective study. An example code snippet that defines a dataset is as follows:

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

See the directory [`resource/dataset`](resource/dataset) for more examples. Also refer to the [Datasets](#datasets) section regarding publicly available datasets.

### Validate a Dataset

Once a dataset is created, first validate the dataset using existing VMAF or other (PSNR, SSIM or MS-SSIM) metrics. Run:

```
./run_testing quality_type test_dataset_file [--vmaf-model optional_VMAF_model_path] [--cache-result] [--parallelize]
```

where `quality_type` can be `VMAF`, `PSNR`, `SSIM` or `MS_SSIM`.

Enabling `--cache-result` allows storing/retrieving extracted features (or elementary quality metrics) in a data store (since feature extraction is the most expensive operations here).

Enabling `--parallelize` allows execution on multiple reference-distorted video pairs in parallel. Sometimes it is desirable to disable parallelization for debugging purpose (e.g. some error messages can only be displayed when parallel execution is disabled).

For example:

```
./run_testing VMAF example_dataset.py --cache-result --parallelize
```

Make sure `matplotlib` is installed to visualize the MOS-prediction scatter plot and inspect the statistics:

- PCC – Pearson correlation coefficient
- SRCC – Spearman rank order correlation coefficient
- RMSE – root mean squared error

#### Troubleshooting

When creating a dataset file, one may make errors (for example, having a typo in a file path) that could go unnoticed but make the execution of `run_testing` fail. For debugging purposes, it is recommended to disable `--parallelize`.

If the problem persists, one may need to run the script:

```
python python/script/run_cleaning_cache.py quality_type test_dataset_file
```

to clean up corrupted results in the store before retrying. For example:

```
python python/script/run_cleaning_cache.py VMAF example_dataset.py
```

### Train a New Model

Now that we are confident that the dataset is created correctly and we have some benchmark result on existing metrics, we proceed to train a new quality assessment model. Run:

```
./run_vmaf_training train_dataset_filepath feature_param_file model_param_file output_model_file [--cache-result] [--parallelize]
```

For example:

```
./run_vmaf_training example_dataset.py \
  resource/feature_param/vmaf_feature_v2.py \
  resource/model_param/libsvmnusvr_v2.py \
  workspace/model/test_model.pkl \
  --cache-result --parallelize
```

`feature_param_file` defines the set of features used. For example, both dictionaries below:

```
feature_dict = {'VMAF_feature':'all', }
```

and

```
feature_dict = {'VMAF_feature':['vif', 'adm'], }
```

are valid specifications of selected features. Here `VMAF_feature` is an 'aggregate' feature type, and `vif`, `adm` are the 'atomic' feature types within the aggregate type. In the first case, `all` specifies that all atomic features of `VMAF_feature` are selected. A `feature_dict` dictionary can also contain more than one aggregate feature types.

`model_param_file` defines the type and hyper-parameters of the regressor to be used. For details, refer to the self-explanatory examples in directory `resource/model_param`. One example is:

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

The trained model is output to `output_model_file`. Once it is obtained, it can be used by the `run_vmaf` or `run_vmaf_in_batch`, or used by `run_testing` to validate another dataset.

![training scatter](/resource/images/scatter_training.png)
![testing scatter](/resource/images/scatter_testing.png)

Above are two example scatter plots obtained from running the `run_vmaf_training` and `run_testing` commands on a training and a testing dataset, respectively.

### Using Custom Subjective Models

The commands `./run_vmaf_training` and `./run_testing` also support custom subjective models (e.g. DMOS (default), MLE and more), through the submodule repository [sureal](https://github.com/Netflix/sureal). Read [this](resource/doc/dcc17v2.pdf) paper for some background.

The subjective model option can be specified with option `--subj-model subjective_model`, for example:

```
./run_vmaf_training example_raw_dataset.py \
  resource/feature_param/vmaf_feature_v2.py \
  resource/model_param/libsvmnusvr_v2.py \
  workspace/model/test_model.pkl \
  --subj-model MLE --cache-result --parallelize

./run_testing VMAF example_raw_dataset.py --subj-model MLE --cache-result --parallelize
```

Note that for the `--subj-model` option to have effect, the input dataset file must follow a format similar to `example_raw_dataset.py`. Specifically, for each dictionary element in `dis_videos`, instead of having a key named 'dmos' or 'groundtruth' as in `example_dataset.py`, it must have a key named 'os' (stand for opinion score), and the value must be a list of numbers. This is the 'raw opinion score' collected from subjective experiments, which is used as the input to the custom subjective models.

### Cross Validation

[`python/script/run_vmaf_cross_validation.py`](python/script/run_vmaf_cross_validation.py) provides tools for cross-validation of hyper-parameters and models. `run_vmaf_cv` runs training on a training dataset using hyper-parameters specified in a parameter file, output a trained model file, and then test the trained model on another test dataset and report testing correlation scores.

`run_vmaf_kfold_cv` takes in a dataset file, a parameter file, and a data structure (list of lists) that specifies the folds based on video content's IDs, and run k-fold cross valiation on the video dataset. This can be useful for manually tuning the model parameters.

### Creating New Features And Regressors

You can also customize VMAF by plugging in third-party features or inventing new features, and specify them in a `feature_param_file`. Essentially, the 'aggregate' feature type (e.g. `VMAF_feature`) specified in the `feature_dict` corresponds to the `TYPE` field of a `FeatureExtractor` subclass (e.g. `VmafFeatureExtractor`). All you need to do is to create a new class extending the `FeatureExtractor` base class.

Similarly, you can plug in a third-party regressor or invent a new regressor and specify them in a `model_param_file`. The `model_type` (e.g. `LIBSVMNUSVR`) corresponds to the `TYPE` field of a `TrainTestModel` sublass (e.g. `LibsvmnusvrTrainTestModel`). All needed is to create a new class extending the `TrainTestModel` base class.

For instructions on how to extending the `FeatureExtractor` and `TrainTestModel` base classes, refer to [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Datasets

### Netflix Public Dataset

We provide a dataset publicly available to the community for training, testing and verification of results purposes. The dataset file is at [`resource/dataset/NFLX_dataset_public.py`](resource/dataset/NFLX_dataset_public.py), and the videos (in YUV420P format) can be downloaded [here](https://drive.google.com/folderview?id=0B3YWNICYMBIweGdJbERlUG9zc0k&usp=sharing).

### VQEG HD3 Dataset

We also provide an example dataset file containing video file names from VQEG (Video Quality Expert Group) HD3 videos. The dataset file is at [`resource/dataset/VQEGHD3_dataset.py`](resource/dataset/VQEGHD3_dataset.py), and the videos is available for downloading from [http://www.cdvl.org/](http://www.cdvl.org/). After login, choose menu 'find videos', and search use keyword 'vqeghd3'. The dataset file includes from `src01` to `src09` except for `src04`, which overlaps with the Netflix Public Dataset, and `hrc04`, `hrc07`, `hrc16`, `hrc17`, `hrc18`, `hrc19`, `hrc20` and `hrc21`, which are the most relevant distortion types to adaptive streaming. After downloading the videos, convert them to YUV420P format.

## Python-independent Implementation

The VDK package combines feature extraction implementation in C and the rest scripting code in Python. The Python layer allows fast prototyping, but sometimes deploying the Python dependency in production is a pain.
Under [`wrapper`](wrapper), we provide a C++ implementation `vmafossexec` that has no dependency on Python.

Under root, run `vmafossexec` as:

```
wrapper/vmafossexec yuv420p 576 324
  python/test/resource/yuv/src01_hrc00_576x324.yuv \
  python/test/resource/yuv/src01_hrc01_576x324.yuv \
  model/vmaf_v0.6.1.pkl \
  --log vmaf_output.xml
```

For VMAF v0.6.1, the model file is `model/vmaf_v0.6.1.pkl`. The correspondence is documented [here](python/src/vmaf/core/quality_runner.py#L255).

## Usage through `libvmaf`

VMAF is now packaged into a library called `libvmaf`. You can install the library built using object files under wrapper/obj and ptools.

To install the library `libvmaf.a `run:

```
make install
```

This copies the library header `libvmaf.h` under `usr/local/include`, library `libvmaf.a `under `user/local/lib` and all the model files under `usr/local/share`. You can use the header `libvmaf.h` in your program. It contains an API which can be called from any C/C++ program:

```
int compute_vmaf(double* vmaf_score, char* fmt, int width, int height, int (*read_frame)(float *ref_data, float *main_data, float *temp_data,
int stride, void *user_data), void *user_data, char *model_path, char *log_path, char *log_fmt, int disable_clip,
int disable_avx, int enable_transform, int phone_model, int do_psnr, int do_ssim, int do_ms_ssim, char *pool_method);
```

Here, `read_frame` is a callback function which can be used to pass data from a program to VMAF. `user_data` is a program specific data that can be used by the callback function. For sample usage of `compute_vmaf`, refer to [`wrapper/src/main.cpp`](wrapper/src/main.cpp).

To uninstall the library run:

```
make uninstall
```

## Usage through Docker

Install Docker, then, from the VMAF directory, run:

```
docker build -t vmaf .
```

And to use it, just run:

```
docker run --rm vmaf [CLI]
```

Where `[CLI]` is one of `[run_vmaf, run_vmaf_training, run_vmaf_in_batch, run_testing, vmafossexec]`.

For example, if you are under root, to run `run_vmaf` on a sample reference/distorted video pair under `resource/yuv`:

```
docker run --rm -v $(pwd):/files vmaf \
  run_vmaf yuv420p 576 324 \
  /files/python/test/resource/yuv/src01_hrc00_576x324.yuv \
  /files/python/test/resource/yuv/src01_hrc01_576x324.yuv \
  --out-fmt json
```

To run `vmafossexec` with a specified model file:

```
docker run --rm -v $(pwd):/files vmaf \
  vmafossexec yuv420p 576 324 \
  /files/python/test/resource/yuv/src01_hrc00_576x324.yuv \
  /files/python/test/resource/yuv/src01_hrc01_576x324.yuv \
  /files/model/nflxall_vmafv4.pkl
```
