VMAF - Video Multi-Method Assessment Fusion
===================

[![Build Status](https://travis-ci.com/Netflix/vmaf.svg?branch=master)](https://travis-ci.com/Netflix/vmaf) [![libvmaf](https://github.com/Netflix/vmaf/workflows/libvmaf/badge.svg)](https://github.com/Netflix/vmaf/actions?query=workflow%3Alibvmaf) [![Windows](https://github.com/Netflix/vmaf/workflows/Windows/badge.svg)](https://github.com/Netflix/vmaf/actions?query=workflow%3AWindows)

VMAF is a perceptual video quality assessment algorithm developed by Netflix. This software package includes a stand-alone C library `libvmaf` and its wrapping Python library. The Python library also provides a set of tools that allows a user to train and test a custom VMAF model. Read [this](https://medium.com/netflix-techblog/toward-a-practical-perceptual-video-quality-metric-653f208b9652) tech blog post for an overview, [this](https://medium.com/netflix-techblog/vmaf-the-journey-continues-44b51ee9ed12) post for the tips of best practices, and [this](https://netflixtechblog.com/toward-a-better-quality-metric-for-the-video-community-7ed94e752a30) post for our latest efforts on speed optimization, new API design and the introduction of a codec evaluation-friendly NEG mode.

Also included in `libvmaf` are implementations of several other metrics: PSNR, PSNR-HVS, SSIM, MS-SSIM and CIEDE2000.

![vmaf logo](resource/images/vmaf_logo.jpg)

## News

- (2020-12-7) Check out our [latest tech blog](https://netflixtechblog.com/toward-a-better-quality-metric-for-the-video-community-7ed94e752a30) on speed optimization, new API design and the introduction of a codec evaluation-friendly NEG mode.
- (2020-12-3) We are releasing `libvmaf v2.0.0`. It has a new fixed-point and x86 SIMD-optimized (AVX2, AVX-512) implementation that achieves 2x speed up compared to the previous floating-point version. It also has a [new API](libvmaf/README.md) that is more flexible and extensible.
- (2020-7-13) We have created a [memo](https://docs.google.com/document/d/1dJczEhXO0MZjBSNyKmd3ARiCTdFVMNPBykH4_HMPoyY/edit?usp=sharing) to share our thoughts on VMAF's property in the presence of image enhancement operations, its impact on codec evaluation, and our solutions.
- (2020-2-27) We have changed VMAF's license from Apache 2.0 to [BSD+Patent](https://opensource.org/licenses/BSDplusPatent), a more permissive license compared to Apache that also includes an express patent grant.

## Frequently Asked Questions

Refer to the [FAQ](FAQ.md) page.

## Usages

The software package offers a number of ways to interact with the VMAF implementation.

  - The command-line tool [`vmaf`](libvmaf/tools/README.md) provides a complete algorithm implementation, such that one can easily deploy VMAF in a production environment. Additionally, the `vmaf` tool provides a number of auxillary metrics such as PSNR, SSIM and MS-SSIM.
  - The [C library `libvmaf`](libvmaf/README.md) provides an interface to incorporate VMAF into your code, and tools to integrate other feature extractors into the library.
  - The [Python library](resource/doc/VMAF_Python_library.md) offers a full array of wrapper classes and scripts for software testing, VMAF model training and validation, dataset processing, data visualization, etc.
  - VMAF is now included as a filter in FFmpeg, and can be configured using: `./configure --enable-libvmaf`. Refer to the [Using VMAF with FFmpeg](resource/doc/ffmpeg.md) page.
  - [VMAF Dockerfile](Dockerfile) generates a docker image from the [Python library](resource/doc/VMAF_Python_library.md). Refer to [this](resource/doc/docker.md) document for detailed usages.
  - To build VMAF on Windows, follow [this](resource/doc/BuildForWindows.md) instruction.

## Datasets

We also provide [two sample datasets](resource/doc/datasets.md) including the video files and the properly formatted dataset files in Python. They can be used as sample datasets to train and test custom VMAF models.

## Models

Besides the default VMAF model which predicts the quality of videos displayed on a 1080p HDTV in a living-room-like environment, VDK also includes a number of additional models, covering phone and 4KTV viewing conditions. Refer to the [models](resource/doc/models.md) page for more details.

## Confidence Interval

Since June 2018, we have introduced a way to quantify the level of confidence that a VMAF prediction entails. Each VMAF prediction score now can come with a 95% confidence interval, which quantifies the level of confidence of the prediction. Refer to the [VMAF Confidence Interval](resource/doc/conf_interval.md) page for more details.

## Matlab Functionality

Besides the C and Python part of the software package, we also introduced a number of algorithms that are implemented in Matlab. For example, users can calculate ST-RRED, ST-MAD, SpEED-QA, and BRISQUE. For more details, see the [Matlab Usage](resource/doc/matlab_usage.md) page for more details.

## Report of VMAF Bad Cases

Over time, we have received feedbacks on when VMAF's prediction does not reflect the expected perceptual quality of videos, either they are corner cases where VMAF fails to cover, or new application scenarios which VMAF was not initially intended for. In response to that, we have created a Google form to allow users to upload their video samples and describe their cases. The bad cases are valuable for improving future versions of VMAF. Users can opt in or out for sharing their sample videos publicly. Please submit the bad cases through [this link](https://docs.google.com/forms/d/e/1FAIpQLSdJntNoBuucMSiYoK3SDWoY1QN0yiFAi5LyEXuOyXEWJbQBtQ/viewform?usp=sf_link).

## Contribution Guide

Refer to the [contribution](CONTRIBUTING.md) page.

## References

Refer to the [references](resource/doc/references.md) page.

## External Resources

Refer to [this page](resource/doc/external_resource.md) for a list of 3rd-party tools built on top of VMAF.
