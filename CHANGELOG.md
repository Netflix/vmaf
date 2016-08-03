# Change Log

## (06/10/2016) [1.0.1]

**Fixed bugs:**
- Fix feature while looping by moving feof to after read_image.
- Fix issue #2 use hashed string for log filename and result filename to avoid file names getting too long.

**New features:**
- Add SsimFeatureExtractor and MsSsimFeatureExtractor with intermediate features (luminence, contrast, structure).

## (06/16/2016) [1.0.2]

**New features:**
- Refactor feature code to expose ssim/ms-ssim, speed up ssim/ms-ssim.

## (07/20/2016) [1.0.3]

**New features:**
- Add base class NorefFeatureExtractor for any feature extractor that do not use a reference video.
- Add MomentNorefFeatureExtractor subclassing NorefFeatureExtractor as an example implementation.

## (07/21/2016) [1.0.4]

**Fixed bugs:**
- Fix a series of numerical issues in VMAF features, increment VmafFeatureExtractor version number.
- Retrain VmafQualityRunner after feature update, increment version number.

