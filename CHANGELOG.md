# Change Log

## (2020-12-4) [2.0.0]

**New features:**
- Add PSNR-HVS and CIEDE2000 metrics.
- ci/actions: upload linux/macos artifacts (#738)
- libvmaf/feature: deprecate daala_ssim (#735)
- libvmaf: remove support for pkl models
- libvmaf/psnr: rewrite using integer types, 2x speedup
- vmaf: if no model is specified, enable v0.6.1 by default (#730)
- libvmaf/x86: add AVX2/AVX-512 optimizations for adm, vif and motion
- ci/actions: add xxd to build dependencies for Windows
- libvmaf: add support for built-in models
- libvmaf/integer_vif: use symmetrical mirroring on edges
- Fix log2 by replacing log2f_approx with log2f
- libvmaf_rc: provide a backwards compatible compute_vmaf(), link vmafossexec with libvmaf
- libvmaf: add framework support for json models
- libvmaf/libsvm: update libsvm to version 324
- libvmaf/motion: add motion_force_zero to motion fex
- return sha1 if Asset string is longer than 255
- Add CID/iCID Matlab source code
- build: unbreak x86 builds (Fixes: #374)
- Add 12bit and 16bit support for python YUV reader; add tests.
- Add PypsnrFeatureExtractor
- Add processes to FeatureAssembler. (#662)

**Fixed bugs:**
- fix motion flush for single frame input
- Fixing the perf_metric for a single entry list input

## (2020-8-24) [1.5.3]

(Updates since 1.5.1)

**Fixed bugs:**
- Fix inverted height and width in integer_motion in vmaf_rc (#650).

**New features:**
- libvmaf: add support for CSV and JSON logging
- Python: Add an (optional) step in Executor class to do python-based processing to ref/dis files (#523).
- Restructure python project and documentation (#544).
- Move test resource to Netflix/vmaf_resource repo (#552).
- Add Github CI (#558).
- Add vmaf_float_v0.6.1neg model; add vif_enhn_gain_limit and adm_enhn_gain_limit options to vmaf_rc.
- Update documentation for FFmpeg+libvmaf.
- Improvements to AucPerfMetric (#643).
- Add motion_force_zero option to vmaf_rc.

## (2020-6-30) [1.5.2]

**Fixed bugs:**
- Fix pkgconfig version sync issue (#572)

**New features:**
- libvmaf_rc general improvements

## (2020-2-27) [1.5.1]
    
**New features:**
- `libvmaf` has been relocated, and now has its own self-enclosed source tree (`./libvmaf/`) and build system (`meson`).
- Update license to BSD+Patent.
- Migrate the build system from makefile to meson.
- Introduce a new release candidate API with the associated library `libvmaf_rc` and executable `vmaf_rc` under `./libvmaf/build`. 
- Add SI and TI feature extractor python classes.
- Add fixed-point SSIM implementation.
- Migrate to python3.

## (2019-9-8) [1.3.15]

**Fixed bugs:**
- Fix a case when CPU cores > 128(MAX_NUM_THREADS) / 3 (#319).
- Avoid dis-filtering ref when not needed, fix return type (#325).
- Update name of file for failed dis_path fopen (#334).
- A few compilation fixes (warnings and errors) (#326).
- Bump up g++ version to 9 for travis (#352).
- Use stat struct instead of ftell to retrieve the file size (#350).

**New features:**
- Write aggregate scores, exec FPS to json output.
- Add support for python3 (#332).
- Print progress in vmafossexec (#337).
- Add VMAF logo.
- Add link to report VMAF bad cases.

## (2019-3-1) [1.3.14]

**Fixed bugs:**
- Fix VMAF value mismatch on 160x90 videos after optimization (#315).
- Fix w10 error with using uninitialized offset_flag variable (#302).

**New features:**
- Add automated Windows builds with AddVeyor (#313).
- Report aggregate CI scores and fix empty model name in log (#304).

## (2019-1-31) [1.3.13]

**New features:**
- Optimized C code for speed. Running in multithreading mode, `vmafossexec` achieves ~40% run time reduction compared to the previous version.
- Printed out individual vmaf bootstrap scores in text file from `vmafossexec`.
- refactored windows solution (#283) (#284) (#285) (#291) (#298).

## (2018-12-17) [1.3.11]

**New features:**
- Revise number of bootstrap models definition: model/vmaf_rb_v0.6.3/vmaf_rb_v0.6.3.pkl has 21 models (20 bootstrap models and one using the full data). From these 21 models, the 20 of them are same as v0.6.2, only added an additional bootstrap model.
- Output the per bootstrap model predictions from wrapper/vmafossexec.
- Print bootstrap individual scores in xml and json.
- Add BD-rate calculator and update documentation.
- Report aggregate PSNR, SSIM, and MS-SSIM scores.
- Add sklearn linear regression class to TrainTestModel.
- Enable BRISQUE feature in VMAF training with bootstrapping.
- Add --save-plot option to command line tools.
- Add ST-RREDOpt (time optimized), ST-MAD feature extractors, quality runners and unittestts. Refactor ST-RRED feature extractor. (#216)

**Fixed bugs:**
- Bug fixed. When start vmaf in multi-thread at the same time. (#239)
- Fix name of min function in vmaf.h and vmaf.cpp. (#227)
- Fix implicit declaration of functions (#225)

## (2018-9-13) [1.3.10]

**New features:**
- Remove sureal as a submodule to vmaf. sureal is now available through pip install.

## (2018-8-7) [1.3.9]

**Fixed bugs:**
- libvmaf: fix case where user defined read_frame() callback was being ignored.

## (2018-6-21) [1.3.8]

**Fixed bugs:**
- Fix compute_vmaf boolean type issue (#178).

## (2018-6-12) [1.3.7]

**New features:**
- Add the --ci option to calculate confidence intervals to predicted VMAF scores (run_vmaf, run_vmaf_in_batch, ffmpeg2vmaf, vmafossexec).
- Update libvmaf version to 1.3.7 after compute_vmaf() interface change (added enable_conf_interval option).
- Add new models: 1) model/vmaf_4k_v0.6.1.pkl for 4KTV viewing at distance 1.5H, 2) model/vmaf_rb_v0.6.2/vmaf_rb_v0.6.2.pkl for VMAF prediction with a confidence interval, 3) model/vmaf_4k_rb_v0.6.2/vmaf_4k_rb_v0.6.2.pkl for 4KTV viewing at distance 1.5H, with a confidence interval.

## (2018-6-4) [1.3.6]

**New features:**
- Update libvmaf version to 1.3.6 (to make consistent with VDK version from now on) after compute_vmaf() interface change (added thread and subsample options).
- Add the option to set the number of threads to use in vmafossexec.
- Add the option to subsample frames to save computation in vmafossexec.

## (2018-5-23) [1.3.5]

**New features:**
- Add multi-threading to vmafossexec.

## (2018-5-8) [1.3.4]

**Refactoring:**
- Refactor mos out of vmaf repo; rename to sureal as submodule.
- Refactor TrainTestModel to make predict() to output dictionary.
- Refactor TrainTestModel.
- Rename KFLK metric to AUC (Area Under the Curve) for better interpretability.

**New features:**
- Add bootstrapping to VMAF. Add two new classes BootstrapVmafQualityRunner and BaggingVmafQualityRunner
- Add Resolving Power Performance Metric.
- Add BRISQUE and NIQE feature extractors. Added two new classes BrisqueNorefFeatureExtractor and NiqeNorefFeatureExtractor. Add NiqeQualityRunner.

**Fixed bugs:**
- Add .gitattributes (#127). Force .pkl and .model files to retain LF line-ending. Required for use on Windows where model files would otherwise be checked out as CRLF which VMAF's parser doesn't handle.
- Allow MinGW compilation of ptools (#133). ptools doesn't build on MinGW as *nix socket headers are included. This patch selects Windows headers for MinGW builds.
- Update compute vmaf interface (#138). Update VMAF version in libvmaf.pc and etc. Catch logic error (resulted from wrong model file format) in compute_vmaf(). Use custom error code.

## (2017-12-3) [1.3.3]

**Fixed bugs:**
- Update VMAF version to 0.6.2 after compute_vmaf() interface change (#124).

## (2017-12-3) [1.3.2]

**Refactoring:**
- Lift check for exec existence during program load.
- Refactor psnr, ssim, ms_ssim and vmaf_feature to call ExternalProgramCaller.
- Refactor feature/Makefile to make executables depend on libvmaf.a.
- Refactor wrapper/Makefile to include additional objs in libvmaf.a but exclude main.o.
- Remove ar -d command after removing main.o from libvmaf.a.

**New features:**
- Generalize read_dataset.
- Update default Asset resampling method to bicubic (#116).
- Extend ffmpeg2vmaf script to allow ref/dis input to be YUV (#118).
- Improve README.md (#121).

**Fixed bugs:**
- Temporary fix Visual Studio builds (#112).
- Avoid unnecessary dependency on matplotlib in run_vmaf (#114).
- Remove unneeded dependencies in Dockerfile, fixes #115 (#117).
- MinGW support (#123).
- Change compute_vmaf() interface to return an error code instead of throw an error #124 (#126).

## (2017-8-12) [1.3.1]

**Refactoring:**
- Refactor NorefExecutorMixin to eliminate repeated codes.
- Refactor C code: get rid of unused double functions; uniformly use read_frame callback function to void repeated code;
- Add strip option to Makefile.

**New features:**
- Update Asset class: add copy functions to Asset; add ref/dis_yuv_type; deprecate yuv_type; add ref/dis_start_sec;
- Update subjective models: add confidence interval to subjective model parameters; refactor MLE model and make subclasses; add run_subj command line.
- Recommend pip, add ffmpeg2vmaf info and reorganize prerequisite installation (#88).
- Reduce sleep time in parallel_map.
- Add library interface for VMAF (#90).
- Add VisualStudio2015 support (#92).
- Add example of image dataset notyuv.
- Add pkgconfig file and changed Makefile.
- Add VmafPhoneQualityRunner class.
- Add DMOS_MLE_CO subjective model.

**Fixed bugs:**
- Update RegressionMixin to handle AUC exception for dicitonary-style dataset.
- Fix Makefile fedora libptools issue. (#98)

## (2017-4-13) [1.2.4]

**Refactoring:**
- Deprecate run_executors_in_parallel.
- Refactor NorefFeatureExtractor into NorefExecutorMixin so that it can be used for all executors.
- Add abstract methods to some base classes.

**New features:**
- Add ST-RRED runner (StrredQualityRunner), based on "Video Quality Assessment by Reduced Reference Spatio-Temporal Entropic Differencing", by R. Soundararaajan, A. Bovik.
- Add start/end frame support for Executor.

## (2017-3-8) [1.2.3]

**New features:**
- Refactor to replace config.ROOT with config.VmafConfig.

## (2017-3-1) [1.2.2]

**New features:**
- Generalize Result and FileSystemResultStore to allow None values.

## (2017-2-27) [1.2.1]

**Tasks:**
- Refactor to prepare for pypi packaging.

## (2017-2-20) [1.2.0]

**New features:**
- Updated VMAF model to version v0.6.1. Changes include: 1) added a custom model for cellular phone screen viewing; 2) trained using new dataset, covering more difficult content; 3) elementary metric fixes: ADM behavior at near-black frames, motion behavior at scene boundaries; 4) compressed quality score range by 20% to accommodate higher dynamic range; 5) Use MLE instead of DMOS as subjective model.

## (2017-1-24) [1.1.23]

**Fixed bugs:**
- Replace subprocess.call with run_process (checking return value).

## (2017-1-22) [1.1.22]

**New features:**
- Add command line ffmpeg2vmaf, which takes encoded videos as input.

## (2017-1-18) [1.1.21]

**New features:**
- Allow processing non-YUV input videos.

## (2016-12-20) [1.1.20]

**New features:**
- Add STRRED runner.

## (2016-12-19) [1.1.19]

**New features:**
- Allow specifying crop and pad parameter in dataset files.

## (2016-12-8) [1.1.18]

**Fixed bugs:**
- Replace pathos with custom function for parallel executor running.

## (2016-12-8) [1.1.17]

**Fixed bugs:**
- Fix command line run_testing issue. Add command line test cases.

## (2016-12-5) [1.1.16]

**New features:**
- Speed up VMAF convolution operation by AVX.

## (2016-11-30) [1.1.15]

**Fixed bugs:**
- Fix vmafossexec memory leakage.

## (2016-11-28) [1.1.14]

**New features:**
- Add enable_transform_score option to VmafQualityRunner, VmafossExecQualityRunner.

## (2016-11-18) [1.1.13]

**Fixed bugs:**
- Fix a bug in DatasetReader.to_aggregated_dataset_file.

## (2016-11-15) [1.1.12]

**New features:**
- Add Travis continuous integration.

## (2016-11-11) [1.1.11]

**New features:**
- Add implementation of AUC (Area Under the Curve) - quality metric evaluation method based on AUC. Refer to: L. Krasula, K. Fliegel, P. Le Callet, M.Klima, "On the accuracy of objective image and video quality models: New methodology for performance evaluation", QoMEX 2016.

## (2016-11-07) [1.1.10]

**New features:**
- Add options to use custom subjective models in run_vmaf_training and run_testing commands.

## (2016-11-02) [1.1.9]

**New features:**
- Add DatasetReader and subclasses; add SubjectiveModel and subclasses.

## (2016-10-19) [1.1.8]

**New features:**
- Add quality runners for each individual VMAF elementary metrics.

## (2016-10-14) [1.1.7]

**Fixed bugs:**
- Issue #36: SSIM and MS-SSIM sometimes get negative values.

## (2016-10-10) [1.1.6]

**New features:**
- Add Xcode project support.
- Add more pooling options (median, percx) to CLIs.

## (2016-10-8) [1.1.5]

**New features:**
- Add support for docker usage (#30).

## (2016-10-7) [1.1.4]

**Fixed bugs:**
- Issue #29: Make ptools build under Fedora.

## (2016-10-6) [1.1.3]

**New features:**
- Generalize dataset format to allow per-content YUV format.

## (2016-10-5) [1.1.2]

**Fixed bugs:**
- Make ptools work under Mac OS.
- Update SklearnRandomForestTrainTestModel test with sklearn 0.18.

## (2016-09-29) [1.1.1]

**New features:**
- Update command lines run_vmaf, run_psnr, run_vmaf_in_batch, run_cleaning_cache, run_vmaf_training and run_testing.

## (2016-09-28) [1.1.0]

**New features:**
- Update wrapper/vmafossexec: 1) it now takes pkl model file as input, so that slopes/intercepts are no longer hard-coded; 2) it now takes multiple YUV input formats; 3) add flag to enable/disable VMAF score clipping at 0/100; 4) allow customly running PSNR/SSIM/MS-SSIM; 5) allow customly outputing XML/JSON
- Add SSIM/MS-SSIM option in run_testing.

## (2016-09-09) [1.0.9]

**Fixed bugs:**
- Move VmafQualityRunnerWithLocalExplainer to quality_runner_adhoc to resolve multiple instances of VMAF found when calling QualityRunner.find_subclass.

**New features:**
- Add custom_clip_0to1 to TrainTestModel.

## (2016-09-07) [1.0.8]

**New features:**
- Generalize read_dataset to allow specifying width, height and resampling method on which to calculate quality.
- Add bicubic to SUPPORTED_RESAMPLING_TYPES for Asset.
- Update Asset rule with resampling_type in __str__ to avoid duplicates in data store.

## (2016-08-20) [1.0.7]

**New features:**
- Update VmafFeatureExtractor to 0.2.2b with scaled ADM features exposed (adm_scale0-3).

## (2016-08-20) [1.0.6]

**New features:**
- Add DisYUVRawVideoExtractor and related classes.
- Add NeuralNetworkTrainTestModel base class that integrates TensorFlow.
- Add example class ToddNoiseClassifierTrainTestModel.

## (2016-08-20) [1.0.5]

**New features:**
- Add LocalExplainer class.
- Add show_local_explanation option to run_vmaf script.

## (2016-07-21) [1.0.4]

**Fixed bugs:**
- Fix a series of numerical issues in VMAF features, increment VmafFeatureExtractor version number.
- Retrain VmafQualityRunner after feature update, increment version number.

## (2016-07-20) [1.0.3]

**New features:**
- Add base class NorefFeatureExtractor for any feature extractor that do not use a reference video.
- Add MomentNorefFeatureExtractor subclassing NorefFeatureExtractor as an example implementation.

## (2016-06-16) [1.0.2]

**New features:**
- Refactor feature code to expose ssim/ms-ssim, speed up ssim/ms-ssim.

## (2016-06-10) [1.0.1]

**Fixed bugs:**
- Fix feature while looping by moving feof to after read_image.
- Fix issue #2 use hashed string for log filename and result filename to avoid file names getting too long.

**New features:**
- Add SsimFeatureExtractor and MsSsimFeatureExtractor with intermediate features (luminence, contrast, structure).

