# Change Log

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
- Add implementation of KFLK - quality metric evaluation method based on AUC. Refer to: L. Krasula, K. Fliegel, P. Le Callet, M.Klima, "On the accuracy of objective image and video quality models: New methodology for performance evaluation", QoMEX 2016.

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

