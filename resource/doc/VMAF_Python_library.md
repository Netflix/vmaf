VMAF Python Library
===================

The VMAF Python library offers full functionalities from running basic VMAF command line, running VMAF on a batch of video files, training and testing a VMAF model on video datasets, and visualization tools, etc. It is the playground to experiment with VMAF.

## Build

Make sure you have `python3` (python 3.6 or higher). You can check the version by `python3 --version`.

Follow the steps below:
```
python3 -m pip install virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate
pip install meson cython numpy
sudo [package-manager] install nasm ninja doxygen
```
You need to invoke `[package-manager]` depending on which system you are on: `apt-get` for Ubuntu and Debian, `yum` for CentOS and RHEL, `dnf` for Fedora, `zypper` for openSUSE, `brew` for MacOS (no `sudo`).

Make sure `nasm` is 2.13.02 or higher (check by `nasm --version`) and `ninja` is 1.7.1 or higher (check by `ninja --version`.

Depending on the system, you may also need to install `python-dev`:
```
sudo [package-manager] install [python-dev]
```
where `[python-dev]` is either `python3-dev` or `python3-devel` depending on the system.

Build the binary by:
```
make
```

Install the rest of the required Python packages:
```
pip install -r python/requirements.txt
```

## Testing

Run unittests and make sure they all pass:
```
./unittest
```

## Basic Usage

One can run VMAF either in single mode by `run_vmaf` or in batch mode by `run_vmaf_in_batch`. Besides, `ffmpeg2vmaf` is a command line tool that offers the capability of taking compressed video bitstreams as input.

### `run_vmaf` -- Running VMAF in Single Mode

To run VMAF on a single reference/distorted video pair, run:

```
PYTHONPATH=python ./python/vmaf/script/run_vmaf.py format width height reference_path distorted_path [--out-fmt output_format]
```

The arguments are the following:

- `format` can be one of:
    - `yuv420p`, `yuv422p`, `yuv444p` (8-Bit YUV)
    - `yuv420p10le`, `yuv422p10le`, `yuv444p10le` (10-Bit little-endian YUV)
    - `yuv420p12le`, `yuv422p12le`, `yuv444p12le` (12-Bit little-endian YUV)
    - `yuv420p16le`, `yuv422p16le`, `yuv444p16le` (16-Bit little-endian YUV)
- `width` and `height` are the width and height of the videos, in pixels
- `reference_path` and `distorted_path` are the paths to the reference and distorted video files
- `output_format` can be one of:
    - `text`
    - `xml`
    - `json`

For example:

```
 PYTHONPATH=python ./python/vmaf/script/run_vmaf.py \
  yuv420p 576 324 \
  python/test/resource/yuv/src01_hrc00_576x324.yuv \
  python/test/resource/yuv/src01_hrc01_576x324.yuv \
  --out-fmt json
```

This will generate JSON output like:

```
"aggregate": {
    "VMAF_feature_adm2_score": 0.93458780776205741, 
    "VMAF_feature_motion2_score": 3.8953518541666665, 
    "VMAF_feature_vif_scale0_score": 0.36342081156994926, 
    "VMAF_feature_vif_scale1_score": 0.76664738784617292, 
    "VMAF_feature_vif_scale2_score": 0.86285338927816291, 
    "VMAF_feature_vif_scale3_score": 0.91597186913930484, 
    "VMAF_score": 76.699271371151269, 
    "method": "mean"
}
```

where `VMAF_score` is the final score and the others are the scores for VMAF's elementary metrics.

- `adm2`, `vif_scalex` scores range from 0 (worst) to 1 (best)
- `motion2` score typically ranges from 0 (static) to 20 (high-motion)

### `run_vmaf_in_batch` -- Running VMAF in Batch Mode

To run VMAF in batch mode, create an input text file, where each corresponds to the following format (check examples in [example_batch_input](../../resource/example/example_batch_input)):

```
format width height reference_path distorted_path
```

For example:

```
yuv420p 576 324 python/test/resource/yuv/src01_hrc00_576x324.yuv \
  python/test/resource/yuv/src01_hrc01_576x324.yuv
yuv420p 576 324 python/test/resource/yuv/src01_hrc00_576x324.yuv \
  python/test/resource/yuv/src01_hrc00_576x324.yuv
```

After that, run:

```
run_vmaf_in_batch input_file [--out-fmt out_fmt] [--parallelize]
```

where enabling `--parallelize` allows execution on multiple reference-distorted video pairs in parallel.

For example:

```
PYTHONPATH=python ./python/vmaf/script/run_vmaf_in_batch.py \
  resource/example/example_batch_input --parallelize
```

### Using `ffmpeg2vmaf`

There is also an `ffmpeg2vmaf` command line tool which can compare any file format decodable by `ffmpeg`. `ffmpeg2vmaf` essentially pipes FFmpeg-decoded videos to VMAF. Note that you need a recent version of `ffmpeg` installed (for the first time, run the command line, follow the prompted instruction to specify the path of `ffmpeg`). 

```
PYTHONPATH=python ./python/vmaf/script/ffmpeg2vmaf.py quality_width quality_height reference_path distorted_path \
  [--model model_path] [--out-fmt out_fmt]
```

Here `quality_width` and `quality_height` are the width and height the reference and distorted videos are scaled to before VMAF calculation. This is different from `run_vmaf`'s  `width` and `height`, which specify the raw YUV's width and height instead. The input to `ffmpeg2vmaf` must already have such information specified in the header so that they are FFmpeg-decodable.

Note that with `libvmaf` as a filter in FFmpeg becoming available (see [this](https://ffmpeg.org/ffmpeg-filters.html#libvmaf) section for details), `ffmpeg2vmaf` is no longer the preferred way to pass in compressed video streams to VMAF. 

## Advanced Usage

VMAF follows a machine-learning based approach to first extract a number of quality-relevant features (or elementary metrics) from a distorted video and its reference full-quality video, followed by fusing them into a final quality score using a non-linear regressor (e.g. an SVM regressor), hence the name “Video Multi-method Assessment Fusion”.

In addition to the basic commands, the VMAF package also provides a framework to allow any user to train his/her own perceptual quality assessment model. For example, directory [`model`](../../model) contains a number of pre-trained models, which can be loaded by the aforementioned commands:

```
PYTHONPATH=python ./python/vmaf/script/run_vmaf.py format width height reference_path distorted_path [--model model_path]
PYTHONPATH=python ./python/vmaf/script/run_vmaf_in_batch.py input_file [--model model_path] --parallelize
```

For example:

```
PYTHONPATH=python ./python/vmaf/script/run_vmaf.py \
  yuv420p 576 324 \
  python/test/resource/yuv/src01_hrc00_576x324.yuv \
  python/test/resource/yuv/src01_hrc01_576x324.yuv \
  --model model/other_models/nflxtrain_vmafv3.pkl

PYTHONPATH=python ./python/vmaf/script/run_vmaf_in_batch.py \
  resource/example/example_batch_input \
  --model model/other_models/nflxtrain_vmafv3.pkl --parallelize
```

A user can customize the model based on:

- The video dataset it is trained on
- The list of features used
- The regressor used (and its hyper-parameters)

Once a model is trained, the VMAF package also provides tools to cross validate it on a different dataset and visualization.

### Create a Dataset

To begin with, create a dataset file following the format in [`example_dataset.py`](../../resource/example/example_dataset.py). A dataset is a collection of distorted videos. Each has a unique asset ID and a corresponding reference video, identified by a unique content ID. Each distorted video is also associated with subjective quality score, typically a MOS (mean opinion score), obtained through subjective study. An example code snippet that defines a dataset is as follows:

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
    {'content_id':0, 'asset_id': 0, 'dmos':100, 'path':'checkerboard.yuv'},
    {'content_id':0, 'asset_id': 1, 'dmos':50,  'path':'checkerboard_dis.yuv'},
    {'content_id':1, 'asset_id': 2, 'dmos':100,  'path':'flat.yuv'},
    {'content_id':1, 'asset_id': 3, 'dmos':80,  'path':'flat_dis.yuv'},
]
```

See the directory [`resource/dataset`](../../resource/dataset) for more examples. Also refer to the [Datasets](datasets.md) document regarding publicly available datasets.

### Validate a Dataset

Once a dataset is created, first validate the dataset using existing VMAF or other (PSNR, SSIM or MS-SSIM) metrics. Run:

```
PYTHONPATH=python ./python/vmaf/script/run_testing.py \
 quality_type test_dataset_file \
[--vmaf-model optional_VMAF_model_path] [--cache-result] [--parallelize]
```

where `quality_type` can be `VMAF`, `PSNR`, `SSIM`, `MS_SSIM`, etc.

Enabling `--cache-result` allows storing/retrieving extracted features (or elementary quality metrics) in a data store (since feature extraction is the most expensive operations here).

Enabling `--parallelize` allows execution on multiple reference-distorted video pairs in parallel. Sometimes it is desirable to disable parallelization for debugging purpose (e.g. some error messages can only be displayed when parallel execution is disabled).

For example:

```
PYTHONPATH=python ./python/vmaf/script/run_testing.py \
VMAF resource/example/example_dataset.py \
  --cache-result --parallelize
```

Make sure `matplotlib` is installed to visualize the MOS-prediction scatter plot and inspect the statistics:

- PCC – Pearson correlation coefficient
- SRCC – Spearman rank order correlation coefficient
- RMSE – root mean squared error

#### Troubleshooting

When creating a dataset file, one may make errors (for example, having a typo in a file path) that could go unnoticed but make the execution of `run_testing` fail. For debugging purposes, it is recommended to disable `--parallelize`.

If the problem persists, one may need to run the script:

```
PYTHONPATH=python ./python/vmaf/script/run_cleaning_cache.py quality_type test_dataset_file
```

to clean up corrupted results in the store before retrying. For example:

```
PYTHONPATH=python ./python/vmaf/script/run_cleaning_cache.py VMAF \
  resource/example/example_dataset.py
```

### Train a New Model

Now that we are confident that the dataset is created correctly and we have some benchmark result on existing metrics, we proceed to train a new quality assessment model. Run:

```
PYTHONPATH=python ./python/vmaf/script/run_vmaf_training.py \
  train_dataset_filepath feature_param_file model_param_file \
  output_model_file [--cache-result] [--parallelize]
```

For example:

```
PYTHONPATH=python ./python/vmaf/script/run_vmaf_training.py \
  resource/example/example_dataset.py \
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

The commands `run_vmaf_training` and `run_testing` also support custom subjective models (e.g. DMOS (default), MLE and more), through the package [sureal](https://github.com/Netflix/sureal).

The subjective model option can be specified with option `--subj-model subjective_model`, for example:

```
PYTHONPATH=python ./python/vmaf/script/run_vmaf_training.py \
  resource/example/example_raw_dataset.py \
  resource/feature_param/vmaf_feature_v2.py \
  resource/model_param/libsvmnusvr_v2.py \
  workspace/model/test_model.pkl \
  --subj-model MLE --cache-result --parallelize

PYTHONPATH=python ./python/vmaf/script/run_testing.py \
  VMAF resource/example/example_raw_dataset.py \
  --subj-model MLE --cache-result --parallelize
```

Note that for the `--subj-model` option to have effect, the input dataset file must follow a format similar to [example_raw_dataset.py](../../resource/example/example_raw_dataset.py). Specifically, for each dictionary element in `dis_videos`, instead of having a key named 'dmos' or 'groundtruth' as in [example_dataset.py](../../resource/example/example_dataset.py), it must have a key named `os` (stands for opinion score), and the value must be a list of numbers. This is the "raw opinion score" collected from subjective experiments, which is used as the input to the custom subjective models.

### Cross Validation

[`run_vmaf_cross_validation.py`](../../python/script/run_vmaf_cross_validation.py) provides tools for cross-validation of hyper-parameters and models. `run_vmaf_cv` runs training on a training dataset using hyper-parameters specified in a parameter file, output a trained model file, and then test the trained model on another test dataset and report testing correlation scores. `run_vmaf_kfold_cv` takes in a dataset file, a parameter file, and a data structure (list of lists) that specifies the folds based on video content's IDs, and run k-fold cross valiation on the video dataset. This can be useful for manually tuning the model parameters.

### Creating New Features And Regressors

You can also customize VMAF by plugging in third-party features or inventing new features, and specify them in a `feature_param_file`. Essentially, the "aggregate" feature type (e.g. `VMAF_feature`) specified in the `feature_dict` corresponds to the `TYPE` field of a `FeatureExtractor` subclass (e.g. `VmafFeatureExtractor`). All you need to do is to create a new class extending the `FeatureExtractor` base class.

Similarly, you can plug in a third-party regressor or invent a new regressor and specify them in a `model_param_file`. The `model_type` (e.g. `LIBSVMNUSVR`) corresponds to the `TYPE` field of a `TrainTestModel` sublass (e.g. `LibsvmnusvrTrainTestModel`). All needed is to create a new class extending the `TrainTestModel` base class.

For instructions on how to extending the `FeatureExtractor` and `TrainTestModel` base classes, refer to [`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## Analysis Tools

Overtime, a number of helper tools have been incorporated into the VDK, to facilitate training and validating VMAF models. An overview of the tools available can be found in [this slide deck](VQEG_SAM_2018_111_AnalysisToolsInVMAF.pdf).

### BD-Rate Calculator

A Bjøntegaard-Delta (BD) rate [implementation](../../python/vmaf/tools/bd_rate_calculator.py) is added. Example usage can be found [here](../../python/test/bd_rate_calculator_test.py). The implementation is validated against [MPEG JCTVC-L1100](http://phenix.int-evry.fr/jct/doc_end_user/current_document.php?id=7281).

### LIME (Local-Explainer Model-Agnostic Explanation) Implementation

An implementation of [LIME](https://arxiv.org/pdf/1602.04938.pdf) is also added as part of the repository. For more information, refer to our [analysis tools](VQEG_SAM_2018_111_AnalysisToolsInVMAF.pdf) presentation. The main idea is to perform a local linear approximation to any regressor or classifier and then use the coefficients of the linearized model as indicators of feature importance. LIME can be used as part of the VMAF regression framework, for example:

```
PYTHONPATH=python ./python/vmaf/script/run_vmaf.py yuv420p 576 324 \
    python/test/resource/yuv/src01_hrc00_576x324.yuv \
    python/test/resource/yuv/src01_hrc00_576x324.yuv --local-explain
```

Naturally, LIME can also be applied to any other regression scheme as long as there exists a pre-trained model. For example, applying to BRISQUE:

```
PYTHONPATH=python ./python/vmaf/script/run_vmaf.py yuv420p 576 324 \
    python/test/resource/yuv/src01_hrc00_576x324.yuv \
    python/test/resource/yuv/src01_hrc00_576x324.yuv --local-explain \
    --model model/other_models/nflxall_vmafv1.pkl
```
