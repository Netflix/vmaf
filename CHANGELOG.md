# Change Log

## (3/1/2019) [1.3.14]

**Fixed bugs:**
- Fix VMAF value mismatch on 160x90 videos after optimization (#315).
- Fix w10 error with using uninitialized offset_flag variable (#302).

**New features:**
- Add automated Windows builds with AddVeyor (#313).
- Report aggregate CI scores and fix empty model name in log (#304).

## (1/31/2019) [1.3.13]

**New features:**
- Optimized C code for speed. Running in multithreading mode, `vmafossexec` achieves ~40% run time reduction compared to the previous version.
- Printed out individual vmaf bootstrap scores in text file from `vmafossexec`.
- refactored windows solution (#283) (#284) (#285) (#291) (#298).

## (12/17/2018) [1.3.11]

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

## (9/13/2018) [1.3.10]

**New features:**
- Remove sureal as a submodule to vmaf. sureal is now available through pip install.

## (8/7/2018) [1.3.9]

**Fixed bugs:**
- libvmaf: fix case where user defined read_frame() callback was being ignored.

## (6/21/2018) [1.3.8]

**Fixed bugs:**
- Fix compute_vmaf boolean type issue (#178).

## (6/12/2018) [1.3.7]

**New features:**
- Add the --ci option to calculate confidence intervals to predicted VMAF scores (run_vmaf, run_vmaf_in_batch, ffmpeg2vmaf, vmafossexec).
- Update libvmaf version to 1.3.7 after compute_vmaf() interface change (added enable_conf_interval option).
- Add new models: 1) model/vmaf_4k_v0.6.1.pkl for 4KTV viewing at distance 1.5H, 2) model/vmaf_rb_v0.6.2/vmaf_rb_v0.6.2.pkl for VMAF prediction with a confidence interval, 3) model/vmaf_4k_rb_v0.6.2/vmaf_4k_rb_v0.6.2.pkl for 4KTV viewing at distance 1.5H, with a confidence interval.

## (6/4/2018) [1.3.6]

**New features:**
- Update libvmaf version to 1.3.6 (to make consistent with VDK version from now on) after compute_vmaf() interface change (added thread and subsample options).
- Add the option to set the number of threads to use in vmafossexec.
- Add the option to subsample frames to save computation in vmafossexec.

## (5/23/2018) [1.3.5]

**New features:**
- Add multi-threading to vmafossexec.

## (5/8/2018) [1.3.4]

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

## (12/3/2017) [1.3.3]

**Fixed bugs:**
- Update VMAF version to 0.6.2 after compute_vmaf() interface change (#124).

## (12/3/2017) [1.3.2]

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

## (8/12/2017) [1.3.1]

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

## (4/13/2017) [1.2.4]

**Refactoring:**
- Deprecate run_executors_in_parallel.
- Refactor NorefFeatureExtractor into NorefExecutorMixin so that it can be used for all executors.
- Add abstract methods to some base classes.

**New features:**
- Add ST-RRED runner (StrredQualityRunner), based on "Video Quality Assessment by Reduced Reference Spatio-Temporal Entropic Differencing", by R. Soundararaajan, A. Bovik.
- Add start/end frame support for Executor.

## (3/8/2017) [1.2.3]

**New features:**
- Refactor to replace config.ROOT with config.VmafConfig.

## (3/1/2017) [1.2.2]

**New features:**
- Generalize Result and FileSystemResultStore to allow None values.

## (2/27/2017) [1.2.1]

**Tasks:**
- Refactor to prepare for pypi packaging.

## (2/20/2017) [1.2.0]

**New features:**
- Updated VMAF model to version v0.6.1. Changes include: 1) added a custom model for cellular phone screen viewing; 2) trained using new dataset, covering more difficult content; 3) elementary metric fixes: ADM behavior at near-black frames, motion behavior at scene boundaries; 4) compressed quality score range by 20% to accommodate higher dynamic range; 5) Use MLE instead of DMOS as subjective model.

## (1/24/2017) [1.1.23]

**Fixed bugs:**
- Replace subprocess.call with run_process (checking return value).

## (1/22/2017) [1.1.22]

**New features:**
- Add command line ffmpeg2vmaf, which takes encoded videos as input.

## (1/18/2017) [1.1.21]

**New features:**
- Allow processing non-YUV input videos.

## (12/20/2016) [1.1.20]

**New features:**
- Add STRRED runner.

## (12/19/2016) [1.1.19]

**New features:**
- Allow specifying crop and pad parameter in dataset files.

## (12/8/2016) [1.1.18]

**Fixed bugs:**
- Replace pathos with custom function for parallel executor running.

## (12/8/2016) [1.1.17]

**Fixed bugs:**
- Fix command line run_testing issue. Add command line test cases.

## (12/5/2016) [1.1.16]

**New features:**
- Speed up VMAF convolution operation by AVX.

## (11/30/2016) [1.1.15]

**Fixed bugs:**
- Fix vmafossexec memory leakage.

## (11/28/2016) [1.1.14]

**New features:**
- Add enable_transform_score option to VmafQualityRunner, VmafossExecQualityRunner.

## (11/18/2016) [1.1.13]

**Fixed bugs:**
- Fix a bug in DatasetReader.to_aggregated_dataset_file.

## (11/15/2016) [1.1.12]

**New features:**
- Add Travis continuous integration.

## (11/11/2016) [1.1.11]

**New features:**
- Add implementation of AUC (Area Under the Curve) - quality metric evaluation method based on AUC. Refer to: L. Krasula, K. Fliegel, P. Le Callet, M.Klima, "On the accuracy of objective image and video quality models: New methodology for performance evaluation", QoMEX 2016.

## (11/07/2016) [1.1.10]

**New features:**
- Add options to use custom subjective models in run_vmaf_training and run_testing commands.

## (11/02/2016) [1.1.9]

**New features:**
- Add DatasetReader and subclasses; add SubjectiveModel and subclasses.

## (10/19/2016) [1.1.8]

**New features:**
- Add quality runners for each individual VMAF elementary metrics.

## (10/14/2016) [1.1.7]

**Fixed bugs:**
- Issue #36: SSIM and MS-SSIM sometimes get negative values.

## (10/10/2016) [1.1.6]

**New features:**
- Add Xcode project support.
- Add more pooling options (median, percx) to CLIs.

## (10/8/2016) [1.1.5]

**New features:**
- Add support for docker usage (#30).

## (10/7/2016) [1.1.4]

**Fixed bugs:**
- Issue #29: Make ptools build under Fedora.

## (10/6/2016) [1.1.3]

**New features:**
- Generalize dataset format to allow per-content YUV format.

## (10/5/2016) [1.1.2]

**Fixed bugs:**
- Make ptools work under Mac OS.
- Update SklearnRandomForestTrainTestModel test with sklearn 0.18.

## (09/29/2016) [1.1.1]

**New features:**
- Update command lines run_vmaf, run_psnr, run_vmaf_in_batch, run_cleaning_cache, run_vmaf_training and run_testing.

## (09/28/2016) [1.1.0]

**New features:**
- Update wrapper/vmafossexec: 1) it now takes pkl model file as input, so that slopes/intercepts are no longer hard-coded; 2) it now takes multiple YUV input formats; 3) add flag to enable/disable VMAF score clipping at 0/100; 4) allow customly running PSNR/SSIM/MS-SSIM; 5) allow customly outputing XML/JSON
- Add SSIM/MS-SSIM option in run_testing.

## (09/09/2016) [1.0.9]

**Fixed bugs:**
- Move VmafQualityRunnerWithLocalExplainer to quality_runner_adhoc to resolve multiple instances of VMAF found when calling QualityRunner.find_subclass.

**New features:**
- Add custom_clip_0to1 to TrainTestModel.

## (09/07/2016) [1.0.8]

**New features:**
- Generalize read_dataset to allow specifying width, height and resampling method on which to calculate quality.
- Add bicubic to SUPPORTED_RESAMPLING_TYPES for Asset.
- Update Asset rule with resampling_type in __str__ to avoid duplicates in data store.

## (08/20/2016) [1.0.7]

**New features:**
- Update VmafFeatureExtractor to 0.2.2b with scaled ADM features exposed (adm_scale0-3).

## (08/20/2016) [1.0.6]

**New features:**
- Add DisYUVRawVideoExtractor and related classes.
- Add NeuralNetworkTrainTestModel base class that integrates TensorFlow.
- Add example class ToddNoiseClassifierTrainTestModel.

## (08/20/2016) [1.0.5]

**New features:**
- Add LocalExplainer class.
- Add show_local_explanation option to run_vmaf script.

## (07/21/2016) [1.0.4]

**Fixed bugs:**
- Fix a series of numerical issues in VMAF features, increment VmafFeatureExtractor version number.
- Retrain VmafQualityRunner after feature update, increment version number.

## (07/20/2016) [1.0.3]

**New features:**
- Add base class NorefFeatureExtractor for any feature extractor that do not use a reference video.
- Add MomentNorefFeatureExtractor subclassing NorefFeatureExtractor as an example implementation.

## (06/16/2016) [1.0.2]

**New features:**
- Refactor feature code to expose ssim/ms-ssim, speed up ssim/ms-ssim.

## (06/10/2016) [1.0.1]

**Fixed bugs:**
- Fix feature while looping by moving feof to after read_image.
- Fix issue #2 use hashed string for log filename and result filename to avoid file names getting too long.

**New features:**
- Add SsimFeatureExtractor and MsSsimFeatureExtractor with intermediate features (luminence, contrast, structure).

