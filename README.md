VMAF - Video Multi-Method Assessment Fusion
===================

[![Build Status](https://travis-ci.org/Netflix/vmaf.svg?branch=master)](https://travis-ci.org/Netflix/vmaf) [![AppVeyor Build Status](https://ci.appveyor.com/api/projects/status/68i57b8ssasttngg?svg=true)](https://ci.appveyor.com/project/li-zhi/vmaf)

VMAF is a perceptual video quality assessment algorithm developed by Netflix. VMAF Development Kit (VDK) is a software package that contains the VMAF algorithm implementation, as well as a set of tools that allows a user to train and test a custom VMAF model. Read [this](https://medium.com/netflix-techblog/toward-a-practical-perceptual-video-quality-metric-653f208b9652) techblog post for an overview, or [this](https://medium.com/netflix-techblog/vmaf-the-journey-continues-44b51ee9ed12) post for the latest updates and tips for best practices.

![vmaf logo](resource/images/vmaf_logo.jpg)

## News

- (9/8/19) Added a [link to report VMAF bad cases](https://docs.google.com/forms/d/e/1FAIpQLSdJntNoBuucMSiYoK3SDWoY1QN0yiFAi5LyEXuOyXEWJbQBtQ/viewform?usp=sf_link). Over time, we have received feedbacks on when VMAF's prediction does not reflect the expected perceptual quality of videos, either they are corner cases where VMAF fails to cover, or new application scenarios which VMAF was not initially intended for. In response to that, we have created the Google form to allow users to upload their video samples and describe the scenarios. The bad cases are valuable for improving future versions of VMAF. Users can opt in or out for sharing their sample videos publicly.
- (1/31/19) Optimized C code for speed. Running in multithreading mode, `vmafossexec` achieves ~40% run time reduction compared to the previous version.
- (11/19/18) Added a BD-rate calculator implementation. See more details [here](resource/doc/VMAF_Python_library.md#bd-rate-calculator).
- (10/25/18) We have published our [second techblog on VMAF](https://medium.com/netflix-techblog/vmaf-the-journey-continues-44b51ee9ed12), with recommendations on best practices.
- (9/13/18) [SUREAL](https://github.com/Netflix/sureal) is no longer a submodule to VMAF.

## Frequently Asked Questions

Refer to the [FAQ](FAQ.md) page.

## Usages

The VDK package offers a number of ways for a user to interact with the VMAF algorithm implementations. The core feature extraction library is written in C. The rest scripting code including the classes for machine learning regression, training and testing VMAF models and etc., is written in Python. Besides, there is C++ an implementation partially replicating the logic in the regression classes, such that the VMAF prediction (excluding training) is fully implemented.

There are a number of ways one can use the package:

  - [VMAF Python library](resource/doc/VMAF_Python_library.md) offers full functionalities including running basic VMAF command line, running VMAF on a batch of video files, training and testing a VMAF model on video datasets, and visualization tools, etc.
  - [`vmafossexec` - a C++ executable](resource/doc/vmafossexec.md) offers running the prediction part of the algorithm in full, such that one can easily deploy VMAF in a production environment without needing to configure the Python dependencies. Additionally, `vmafossexec` offers a number of exclusive features, such as 1) speed optimization using multi-threading and skipping frames, 2) optionally computing PSNR, SSIM and MS-SSIM metrics in the output.
  - [`libvmaf.a` - a static library](resource/doc/libvmaf.md) offers an interface to incorporate VMAF into your C/C++ code. Using this library, VMAF is now included as a filter in [FFmpeg](http://ffmpeg.org/) main branch, and can be configured using: `./configure --enable-libvmaf --enable-version3`. See [this](resource/doc/libvmaf.md#use-libvmaf-with-ffmpeg) section for details. Using FFmpeg with `libvmaf` allows passing in compressed video bitstreams directly to VMAF.
  - [VMAF Dockerfile](Dockerfile) generates a VMAF docker image from the [VMAF Python library](resource/doc/VMAF_Python_library.md). Refer to [this](resource/doc/docker.md) document for detailed usages.
  - Build VMAF on Windows: follow instructions on [this](resource/doc/BuildForWindows.md) page.

## Datasets

We also provide [two sample datasets](resource/doc/datasets.md) including the video files and the properly formatted dataset files in Python. They can be used as sample datasets to train and test custom VMAF models.

## Models

Besides the default VMAF model which predicts the quality of videos displayed on a 1080p HDTV in a living-room-like environment, VDK also includes a number of additional models, covering phone and 4KTV viewing conditions. Refer to the [models](resource/doc/models.md) page for more details.

## Confidence Interval

Since VDK v1.3.7 (June 2018), we have introduced a way to quantify the level of confidence that a VMAF prediction entails. Each VMAF prediction score now can come with a 95% confidence interval (CI), which quantifies the level of confidence that the prediction lies within the interval. Refer to the [VMAF confidence interval](resource/doc/conf_interval.md) page for more details.

## Matlab Functionality

Besides the Python/C/C++ part of the repository, we also introduced a number of algorithms that are implemented in Matlab. For example, users can calculate ST-RRED, ST-MAD, SpEED-QA, and BRISQUE. For more details, see the [Matlab Usage](resource/doc/matlab_usage.md) page for more details.

## References

Refer to the [references](resource/doc/references.md) page.
