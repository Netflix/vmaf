VMAF - Video Multi-Method Assessment Fusion
===================
[![Build Status](https://travis-ci.org/Netflix/vmaf.svg?branch=master)](https://travis-ci.org/Netflix/vmaf)

VMAF is a perceptual video quality assessment algorithm developed by Netflix. VMAF Development Kit (VDK) is a software package that contains the VMAF algorithm implementation, as well as a set of tools that allows a user to train and test a custom VMAF model. For an overview, read [this](http://techblog.netflix.com/2016/06/toward-practical-perceptual-video.html) tech blog post, or [this](resource/doc/VMAF_ICIP17.pdf) slide deck.

## What's New

- (6/19/18) Each VMAF prediction score now comes with a 95% confidence interval (CI), which quantifies the level of confidence that the prediction lies within the interval. Refer to [this](resource/doc/VMAF_Python_library.md#invoking-prediction-confidence-interval) section for details.
- (6/19/18) Added 4K VMAF model under `model/vmaf_4k_v0.6.1.pkl`, which predicts the subjective quality of video displayed on a 4KTV and viewed from the distance of 1.5X the display height. Read [this](resource/doc/VMAF_Python_library.md#predict-quality-on-a-4ktv-screen-at-15h) section for details.
- (6/5/18) Speed optimization to `vmafossexec`: 1) support multi-threading (e.g. use `--thread 0` to use all cores), 2) support frame sampling (e.g. use `--subsample 5` to calculate VMAF on one of every 5 frames). See [this](resource/doc/vmafossexec.md#vmafossexec---python-independent-implementation) section for details.
- (1/20/18) Moved custom subjective models into a submodule named [sureal](https://github.com/Netflix/sureal). If you pull the latest changes, you will have to pull the submoddule by `git submodule update --init --recursive` and add `sureal/python/src` to `PYTHONPATH`.
- (8/12/17) VMAF is now packaged into a library call `libvmaf` and can be called from a C/C++ program directly. See [this](resource/doc/libvmaf.md) document for details. Using this library, VMAF is now included as a filter in [FFmpeg](http://ffmpeg.org/) main branch, and can be configured using: `./configure --enable-libvmaf`.

## Frequently Asked Questions

Refer to the [FAQ](FAQ.md) page.

## Usages

THe VDK package offers multiple ways for a user to interact with VMAF algorithm implementations. The core feature extraction library is written in C. The rest scripting code including the classes for machine learning regression, training and testing VMAF models and etc., is written in Python. Besides, there is C++ "wrapper" code partially replicating the logic in the regression classes, such that the VMAF prediction (excluding training) is fully implemented in C/C++.

There are a number of ways one can use the pakcage: 

  - [VMAF Python library](resource/doc/VMAF_Python_library.md) offers full functionalities from running basic VMAF command line, running VMAF on a batch of video files, training and testing a VMAF model on video datasets, and visualization tools, etc.
  - [`vmafossexec` - a C++ "wrapper" executable](resource/doc/vmafossexec.md) offers running the prediction part of the algorithm in full, such that one can easily deploy VMAF in a production environment without needing to configure the Python dependancies. Additionally, `vmafossexec` offers a number of exclusive features, such as 1) speed optimization using multi-threading and skipping frames, 2) optionally computing PSNR, SSIM and MS-SSIM in the output.
  - [`libvmaf` - a static library](resource/doc/libvmaf.md) offers an interface to allow incorporating VMAF into your C/C++ code. Using this library, VMAF is now included as a filter in [FFmpeg](http://ffmpeg.org/) main branch, and can be configured using: `./configure --enable-libvmaf`.
  - [VMAF dockerfile](Dockerfile) generates a VMAF docker image. Refer to [this](resource/doc/docker.md) document for detailed usages.
  
## Datasets

We also provide [two sample datasets](resource/doc/datasets.md) including the video files and the properly formatted dataset files in Python.

