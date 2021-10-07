# `vmaf`

`vmaf` is a command line tool which supports VMAF feature extraction and prediction. The tool takes a pair of input videos as well as a trained VMAF model and writes an output log containing per-frame and pooled VMAF scores. Input videos can be either `.y4m` or `.yuv` and output logs are available in a number of formats: `.xml`, `.json`, `.csv`, `.sub`.

## Compile

Refer to the [`libvmaf`](../README.md) Compile section.

## Usage

```
Usage: vmaf [options]

Supported options:
 --reference/-r $path:      path to reference .y4m or .yuv
 --distorted/-d $path:      path to distorted .y4m or .yuv
 --width/-w $unsigned:      width
 --height/-h $unsigned:     height
 --pixel_format/-p: $string pixel format (420/422/444)
 --bitdepth/-b $unsigned:   bitdepth (8/10/12)
 --model/-m $params:        model parameters, colon ":" delimited
                            `path=` path to model file
                            `version=` built-in model version
                            `name=` optional name used in logs
 --output/-o $path:         path to output file
 --xml:                     write output file as XML (default)
 --json:                    write output file as JSON
 --csv:                     write output file as CSV
 --sub:                     write output file as subtitle
 --threads $unsigned:       number of threads to use
 --feature $string:         additional feature
 --cpumask: $bitmask        restrict permitted CPU instruction sets
 --subsample: $unsigned     compute scores only every N frames
 --quiet/-q:                disable FPS meter when run in a TTY
 --no_prediction/-n:        no prediction, extract features only
 --version/-v:              print version and exit
```

## Input
Pass your reference/distorted pair of videos to the tool using the `--reference` and `--distorted` flags. If your inputs are `.y4m` this is all that is required. If your inputs are raw `.yuv` the following parameters are also required: `--width`, `--height`, `--pixel_format`, `--bitdepth`.

```shell script
# .y4m
--reference ducks.y4m \
--distorted ducks_dist.y4m \

# .yuv
--reference ducks.yuv \
--distorted ducks_dist.yuv \
--width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
```

## VMAF Models
`vmaf` now has a number of VMAF models built-in. This means that no external VMAF model files are required, and the models are read from the binary itself. Previous versions of `libvmaf` required a `.pkl` format model file. Since v2.0.0, these `.pkl` model files have been deprecated in favor of `.json` model files. If you have a previously trained `.pkl` model you would like to convert to `.json`, the following [Python conversion script](../../python/vmaf/script/convert_model_from_pkl_to_json.py) is available. If the `--model` parameter is not passed at all, `version=vmaf_v0.6.1` is enabled by default.

```shell script
# built-in model
--model version=vmaf_v0.6.1

# model file
--model path=../model/vmaf_v0.6.1.json
```

## Additional Metrics
A number of addtional metrics are supported. Enable these metrics with the `--feature` flag.

```shell script
# psnr, psnr_hvs, ssim, ms-ssim, ciede
--feature psnr \
--feature psnr_hvs \
--feature float_ssim \
--feature float_ms_ssim
--feature ciede
--feature cambi
```

## Example

The following example shows a comparison using a pair of yuv inputs ([`src01_hrc00_576x324.yuv`](https://github.com/Netflix/vmaf_resource/blob/master/python/test/resource/yuv/src01_hrc00_576x324.yuv), [`src01_hrc01_576x324.yuv`](https://github.com/Netflix/vmaf_resource/blob/master/python/test/resource/yuv/src01_hrc01_576x324.yuv)). In addition to VMAF, the `psnr` metric is also computed and logged.

```shell script
./build/tools/vmaf \
    --reference src01_hrc00_576x324.yuv \
    --distorted src01_hrc01_576x324.yuv \
    --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
    --model version=vmaf_v0.6.1 \
    --feature psnr \
    --output output.xml
```
Example output:
```text
VMAF version e1d466c
48 frames ⠀⠩ 44.72 FPS
vmaf_v0.6.1: 76.668905
```
with `output.xml`:
```xml
<VMAF version="e1d466c">
  <params qualityWidth="576" qualityHeight="324" />
  <fyi fps="41.98" />
  <frames>
    <frame frameNum="0" integer_adm2="0.962084" integer_adm_scale0="0.946338" integer_adm_scale1="0.939006" integer_adm_scale2="0.957474" integer_adm_scale3="0.980887" integer_motion2="0.000000" integer_motion="0.000000" integer_vif_scale0="0.505712" integer_vif_scale1="0.879061" integer_vif_scale2="0.937873" integer_vif_scale3="0.964301" psnr_y="34.760779" psnr_cb="39.229987" psnr_cr="41.349703" vmaf="83.856285" />
    ...
  </frames>
  <pooled_metrics>
    <metric name="integer_adm2" min="0.921008" max="0.962084" mean="0.934506" harmonic_mean="0.934463" />
    ...
    <metric name="psnr_y" min="29.640688" max="34.760779" mean="30.755064" harmonic_mean="30.727905" />
    <metric name="psnr_cb" min="38.019979" max="39.229987" mean="38.449441" harmonic_mean="38.447866" />
    <metric name="psnr_cr" min="40.649266" max="41.353846" mean="40.991910" harmonic_mean="40.991083" />
    <metric name="vmaf" min="71.176557" max="87.181420" mean="76.668905" harmonic_mean="76.510006" />
  </pooled_metrics>
  <aggregate_metrics />
</VMAF>
```
