# `vmaf`

`vmaf` is a command line tool which supports VMAF feature extraction and prediction. The tool takes a pair of input videos as well as a trained VMAF model and writes an output log containing per-frame and pooled VMAF scores. Input videos can be either `.y4m` or `.yuv` and output logs are available in a number of formats: `.xml`, `.json`, `.csv`, `.sub`.

## Usage

```
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

```sh
# .y4m
--reference ducks.y4m \
--distorted ducks_dist.y4m \

# .yuv
--reference ducks.yuv \
--distorted ducks_dist.yuv \
--width 1920 --height 1080 --pixel_format 420 --bitdepth 8 \
```

## Model
`vmaf` now has a number of VMAF models built-in. This means that no external VMAF model files are required, and the models are read from the binary itself. Previous versions of `libvmaf` required a `.pkl` format model file. Since v2.0.0, these `.pkl` model files have been depreciated in favor of `.json` model files. If you have a previously trained `.pkl` model you would like to convert to `.json`, the following [Python conversion script](../python/vmaf/script/convert_model_from_pkl_to_json.py) is available. If the `--model` parameter is not passed at all, `version=vmaf_v0.6.1` is enabled by default.

```
# built-in model
--model version=vmaf_v0.6.1

# model file
--model path=../model/vmaf_v0.6.1.json
```

## Example

The following example shows a comparison using a pair of yuv inputs (`src01_hrc00_576x324.yuv`, `src01_hrc01_576x324.yuv`). In addition to VMAF which is enabled with the model `../model/vmaf_float_v0.6.1.pkl`, the `psnr` metric is also computed and logged.

```sh
./build/tools/vmaf \
    --reference src01_hrc00_576x324.yuv \
    --distorted src01_hrc01_576x324.yuv \
    --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
    --model version=vmaf_v0.6.1 \
    --feature psnr \
    --output output.xml
```
