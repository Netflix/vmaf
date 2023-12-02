# CAMBI

CAMBI (Contrast Aware Multiscale Banding Index) is Netflix's detector for banding (aka contouring) artifacts.

## Background

For an introduction to CAMBI, please refer to the [tech blog](https://netflixtechblog.medium.com/cambi-a-banding-artifact-detector-96777ae12fe2). For a detailed technical description, please refer to the [technical paper](papers/CAMBI_PCS2021.pdf) published at PCS 2021. Note that the paper describes an initial version of CAMBI that no longer matches the code exactly, but it is still a good introduction.

By default, the current version of CAMBI is a [no-reference metric](https://en.wikipedia.org/wiki/Video_quality#Classification_of_objective_video_quality_models), and operates on a frame-by-frame basis (no temporal information is leveraged). To integrate it as part of the VMAF framework, which employs a [full-reference metric](https://en.wikipedia.org/wiki/Video_quality#Classification_of_objective_video_quality_models) API, CAMBI takes both a reference and a distorted video as its input. For simplicity, one can point the input arguments `--reference` and `--distorted` to the same video path. 

CAMBI also offers a full-reference mode which computes its score as `MAX(0, distorted_score - reference_score)`. This mode can be activated with the `--full_ref` command line option. In this case, both the `--reference` and `--distorted` inputs will be used.

## Scores

The CAMBI score starts at 0, meaning no banding is detected. A higher CAMBI score means more visible banding artifacts are identified. The maximum CAMBI observed in a sequence is 24 (unwatchable). As a rule of thumb, a CAMBI score around 5 is where banding starts to become slightly annoying (also note that banding is highly dependent on the viewing environment - the brigher the display, and the dimmer the ambient light, the more visible banding is).

## How to run CAMBI

To invoke CAMBI using the VMAF command line, follow the [instruction](../../libvmaf/tools/README.md) and use `cambi` as the feature name. For example, after downloading the input video [`src01_hrc01_576x324.yuv`](https://github.com/Netflix/vmaf_resource/blob/master/python/test/resource/yuv/src01_hrc01_576x324.yuv), invoke CAMBI via:

```shell script
libvmaf/build/tools/vmaf \
    --reference src01_hrc01_576x324.yuv \
    --distorted src01_hrc01_576x324.yuv \
    --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature cambi --output /dev/stdout
```

This will yield the output:
```
<VMAF version="4b42f672">
  <params qualityWidth="576" qualityHeight="324" />
  <fyi fps="52.47" />
  <frames>
    <frame frameNum="0" cambi="0.848047" />
    <frame frameNum="1" cambi="0.723467" />
    ...
    <frame frameNum="46" cambi="0.994815" />
    <frame frameNum="47" cambi="1.019691" />
  </frames>
  <pooled_metrics>
    <metric name="cambi" min="0.509878" max="1.019691" mean="0.689250" harmonic_mean="0.681308" />
  </pooled_metrics>
  <aggregate_metrics />
</VMAF>
```

## Bit depths

CAMBI supports the same input bit depths as VMAF: 8, 10, 12 and 16. However, the computations in CAMBI will always be performed at the 10-bit level, and the other formats will be converted to 10-bit as a preprocessing step.

## Options

The CAMBI feature extractor also supports additional optional parameters as listed below:

- `window_size` (min: 15, max: 127, default: 63): Window size to compute CAMBI (default: 63 corresponds to ~1 degree at 4K resolution and 1.5H)
- `topk` (min: 0, max: 1.0, default: 0.6): Ratio of pixels for the spatial pooling computation
- `tvi_threshold` (min: 0.0001, max: 1.0, default: 0.019): Visibilty threshold for luminance ΔL < tvi_threshold*L_mean for BT.1886
- `max_log_contrast` (min: 0, max: 5, default: 2): Maximum contrast in log luma level (2^max_log_contrast) at 10-bits. Default 2 is equivalent to 4 luma levels at 10-bit and 1 luma level at 8-bit. The default is recommended for banding artifacts coming from video compression.
- `full_ref`: optional flag (default: false) to run CAMBI as a full-reference metric, outputting the per-frame difference between the encoded and source images as well as the existing no-reference score.
- `enc_width` and `enc_height`: Encoding/processing resolution to compute the banding score, useful in cases where scaling was applied to the input prior to the computation of metrics
- `src_width` and `src_height`: Encoding/processing resolution to compute the banding score on the reference image, only used if `full_ref=true`.
- `heatmaps_path`: Set to a folder where the heatmaps for different scales will be stored as `.gray` files

An example using the `enc_width` and `enc_height` options on the input video [`KristenAndSara_1280x720_8bit_processed.yuv`](https://github.com/Netflix/vmaf_resource/blob/master/python/test/resource/yuv/KristenAndSara_1280x720_8bit_processed.yuv) which has been encoded at 540p and later upscaled to 1280p (specifying the accurate encoding width and height as input allows CAMBI to more accurately assess the banding artifact):

```shell script
libvmaf/build/tools/vmaf \
    --reference KristenAndSara_1280x720_8bit_processed.yuv \
    --distorted KristenAndSara_1280x720_8bit_processed.yuv \
    --width 1280 --height 720 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature cambi=enc_width=960:enc_height=540 --output /dev/stdout
```

This will yield the output:
```
<VMAF version="4b42f672">
  <params qualityWidth="1280" qualityHeight="720" />
  <fyi fps="40000.00" />
  <frames>
    <frame frameNum="0" cambi="1.218365" />
  </frames>
  <pooled_metrics>
    <metric name="cambi" min="1.218365" max="1.218365" mean="1.218365" harmonic_mean="1.218365" />
  </pooled_metrics>
  <aggregate_metrics />
</VMAF>
```

If no encoding width and height parameters are specified:

```shell script
libvmaf/build/tools/vmaf \
    --reference KristenAndSara_1280x720_8bit_processed.yuv \
    --distorted KristenAndSara_1280x720_8bit_processed.yuv \
    --width 1280 --height 720 --pixel_format 420 --bitdepth 8 \
    --no_prediction --feature cambi --output /dev/stdout
```

The output will be:
```
<VMAF version="4b42f672">
  <params qualityWidth="1280" qualityHeight="720" />
  <fyi fps="47619.05" />
  <frames>
    <frame frameNum="0" cambi="0.341833" />
  </frames>
  <pooled_metrics>
    <metric name="cambi" min="0.341833" max="0.341833" mean="0.341833" harmonic_mean="0.341833" />
  </pooled_metrics>
  <aggregate_metrics />
</VMAF>
```

## Generating and Decoding Heatmaps

To generate the heatmaps, run CAMBI with the `heatmaps_path` option set to a local folder. It will write files such as:

```
cambi_heatmap_scale_0_1280x720_16b.gray
cambi_heatmap_scale_1_640x360_16b.gray
cambi_heatmap_scale_2_320x180_16b.gray
cambi_heatmap_scale_3_160x90_16b.gray
cambi_heatmap_scale_4_80x45_16b.gray
```

containing the raw grayscale heatmap data.

You can use `ffmpeg` to convert them:

```
ffmpeg -f rawvideo -pix_fmt gray16le -s 1280x720 -i heatmaps/cambi_heatmap_scale_0_1280x720_16b.gray -frames:v 1 heatmaps/cambi_heatmap_scale_0_1280x720_16b.png
```

## Python Library

CAMBI can also be invoked in the [Python library](python.md). Use `CambiFeatureExtractor` as the feature extractor, and `CambiQualityRunner` as the quality runner. Use `CambiFullReferenceFeatureExtractor` and `CambiFullReferenceQualityRunner` to run the full-reference version of CAMBI.

```
    dis_path = VmafConfig.test_resource_path("yuv", "KristenAndSara_1280x720_8bit_processed.yuv")
    asset = Asset(dataset="test", content_id=0, asset_id=0,
                  workdir_root=VmafConfig.workdir_path(),
                  ref_path=dis_path,
                  dis_path=dis_path,
                  asset_dict={'width': 1280, 'height': 720,
                              'dis_enc_width': 960, 'dis_enc_height': 540})

        self.qrunner = CambiQualityRunner(
            [asset, asset_original],
            None, fifo_mode=False,
            result_store=None,
            optional_dict={}
        )
        self.qrunner.run(parallelize=True)
        results = self.qrunner.results

        # score: arithmetic mean score over all frames
        self.assertAlmostEqual(results[0]['Cambi_score'],
                               1.218365, places=4)
```

