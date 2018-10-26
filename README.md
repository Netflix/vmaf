VMAF - Video Multi-Method Assessment Fusion
===================
[![Build Status](https://travis-ci.org/Netflix/vmaf.svg?branch=master)](https://travis-ci.org/Netflix/vmaf)

VMAF is a perceptual video quality assessment algorithm developed by Netflix. VMAF Development Kit (VDK) is a software package that contains the VMAF algorithm implementation, as well as a set of tools that allows a user to train and test a custom VMAF model. For an overview, read [this](http://techblog.netflix.com/2016/06/toward-practical-perceptual-video.html) tech blog post, or [this](resource/doc/VMAF_ICIP17.pdf) slide deck.

## News

- (10/25/18) We have published our [second techblog](https://medium.com/netflix-techblog/vmaf-the-journey-continues-44b51ee9ed12) on VMAF with recommendations on best practices.
- (9/13/18) [SUREAL](https://github.com/Netflix/sureal) is no longer a submodule to VMAF. 
- (6/19/18) Each VMAF prediction score now comes with a 95% [confidence interval (CI)](resource/doc/conf_interval.md), which quantifies the level of confidence that the prediction lies within the interval.
- (6/19/18) Added a [4K VMAF model](resource/doc/models.md/#predict-quality-on-a-4ktv-screen-at-15h) under `model/vmaf_4k_v0.6.1.pkl`, which predicts the subjective quality of video displayed on a 4KTV and viewed from the distance of 1.5X the display height.
- (6/5/18) Speed optimization to [`vmafossexec`](resource/doc/vmafossexec.md): 1) support multi-threading (e.g. use `--thread 0` to use all cores), 2) support frame sampling (e.g. use `--subsample 5` to calculate VMAF on one of every 5 frames).

## Frequently Asked Questions

Refer to the [FAQ](FAQ.md) page.

## Usages

The VDK package offers a number of ways for a user to interact with the VMAF algorithm implementations. The core feature extraction library is written in C. The rest scripting code including the classes for machine learning regression, training and testing VMAF models and etc., is written in Python. Besides, there is C++ "wrapper" code partially replicating the logic in the regression classes, such that the VMAF prediction (excluding training) is fully implemented in C/C++.

There are a number of ways one can use the pakcage: 

  - [VMAF Python library](resource/doc/VMAF_Python_library.md) offers full functionalities including running basic VMAF command line, running VMAF on a batch of video files, training and testing a VMAF model on video datasets, and visualization tools, etc.
  - [`vmafossexec` - a C++ "wrapper" executable](resource/doc/vmafossexec.md) offers running the prediction part of the algorithm in full, such that one can easily deploy VMAF in a production environment without needing to configure the Python dependencies. Additionally, `vmafossexec` offers a number of exclusive features, such as 1) speed optimization using multi-threading and skipping frames, 2) optionally computing PSNR, SSIM and MS-SSIM metrics in the output.
  - [`libvmaf.a` - a static library](resource/doc/libvmaf.md) offers an interface to incorporate VMAF into your C/C++ code. Using this library, VMAF is now included as a filter in [FFmpeg](http://ffmpeg.org/) main branch, and can be configured using: `./configure --enable-libvmaf --enable-version3`. See [this](https://ffmpeg.org/ffmpeg-filters.html#libvmaf) section for details. Using FFmpeg with `libvmaf` allows passing in compressed video bitstreams directly to VMAF.
  - [VMAF Dockerfile](Dockerfile) generates a VMAF docker image from the [VMAF Python library](resource/doc/VMAF_Python_library.md). Refer to [this](resource/doc/docker.md) document for detailed usages.

## Datasets

We also provide [two sample datasets](resource/doc/datasets.md) including the video files and the properly formatted dataset files in Python. They can be used as sample datasets to train and test custom VMAF models.

## Models

Besides the default VMAF model which predicts the quality of videos displayed on a 1080p HDTV in a living-room-like environment, VDK also includes a number of additional models, covering phone and 4KTV viewing conditions. Refer to the [models](resource/doc/models.md) page for more details.

## Confidence Interval

Since VDK v1.3.7 (June 2018), we have introduced a way to quantify the level of confidence that a VMAF prediction entails. Each VMAF prediction score now can come with a 95% confidence interval (CI), which quantifies the level of confidence that the prediction lies within the interval. Refer to the [VMAF confidence interval](resource/doc/conf_interval.md) page for more details.

## References

Refer to the [references](resource/doc/references.md) page.
