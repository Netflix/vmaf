# libvmaf

## Prerequisites

For building, you need the following:

- [Python3](https://www.python.org/download/releases/3.0/) (3.6 or higher)
- [Meson](https://mesonbuild.com/) (0.47 or higher)
- [Ninja](https://ninja-build.org/) (1.7.1 or higher)
- [NASM](https://www.nasm.us/) (for x86 builds only, 2.13.02 or higher)

Follow the steps below:
```
python3 -m pip install virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate
pip install meson
sudo [package-manager] install nasm ninja doxygen
```
You need to invoke `[package-manager]` depending on which system you are on: `apt-get` for Ubuntu and Debian, `yum` for CentOS and RHEL, `dnf` for Fedora, `zypper` for openSUSE, `brew` for MacOS (no `sudo`).

## Compile

Run:

```
meson build --buildtype release
```

Build with:

```
ninja -vC build
```

## Test

Build and run tests with:

```
ninja -vC build test
```

## Install

Install the libraries and models to `/usr/local` using:

```
ninja -vC build install
```

Under Linux, you may need `sudo` for the above command.

## Documentation

Generate HTML documentation with:

```
ninja -vC build doc/html
```

## API

Create a `VmafContext` with `vmaf_init()`. `VmafContext` is an opaque type, and `VmafConfiguration` is a options struct used to initialize the context. Be sure to clean up the `VmafContext` with `vmaf_close()` when you are done with it.

```c
int vmaf_init(VmafContext **vmaf, VmafConfiguration cfg);

int vmaf_close(VmafContext *vmaf);
```

Calculating a VMAF score requires a VMAF model. The next step is to create a `VmafModel`. There are a few ways to get a `VmafModel`. Use `vmaf_model_load()` when you would like to load one of the default built-in models. Use `vmaf_model_load_from_path()` when you would like to read a model file from a filesystem. After you are done using the `VmafModel`, clean it up with `vmaf_model_destroy()`.

```c
int vmaf_model_load(VmafModel **model, VmafModelConfig *cfg,
                     const char *version);

int vmaf_model_load_from_path(VmafModel **model, VmafModelConfig *cfg,
                              const char *path);

void vmaf_model_destroy(VmafModel *model);
```


A VMAF score is a fusion of several elementary features which are specified by a model file. The next step is to register all feature extractors required by your model or models with `vmaf_use_features_from_model()`. If there are auxillary metrics (i.e. `PSNR`) you would also like to extract use `vmaf_use_feature()` to register it directly.

```c
int vmaf_use_features_from_model(VmafContext *vmaf, VmafModel *model);

int vmaf_use_feature(VmafContext *vmaf, const char *feature_name,
                      VmafFeatureDictionary *opts_dict);
```

VMAF is a full-reference metric, meaning it is calculated on pairs of reference/distorted pictures. To allocate a `VmafPicture` use `vmaf_picture_alloc`. After allocation, you may fill the buffers with pixel data.

```c
int vmaf_picture_alloc(VmafPicture *pic, enum VmafPixelFormat pix_fmt,
                        unsigned bpc, unsigned w, unsigned h);
```

Read all of you input pictures in a loop with `vmaf_read_pictures()`. When you are done reading pictures, some feature extractors may have internal buffers may still need to be flushed. Call `vmaf_read_pictures()` again with `ref` and `dist` set to `NULL` to flush these buffers. Once buffers are flushed, all further calls to `vmaf_read_pictures()` are invalid.

```c
int vmaf_read_pictures(VmafContext *vmaf, VmafPicture *ref, VmafPicture *dist,
                       unsigned index);
```

After your pictures have been read, you can retrieve a vmaf score. Use `vmaf_score_at_index` to get the score at single index, and use `vmaf_score_pooled()` to get a pooled score across multiple frames.

```c
int vmaf_score_at_index(VmafContext *vmaf, VmafModel *model, double *score,
                        unsigned index);

int vmaf_score_pooled(VmafContext *vmaf, VmafModel *model,
                      enum VmafPoolingMethod pool_method, double *score,
                      unsigned index_low, unsigned index_high);
```

## Contributing a new VmafFeatureExtractor

To write a new VmafFeatureExtractor, please first familiarize yourself with the [VmafFeatureExtractor API documentation](https://github.com/Netflix/vmaf/blob/master/libvmaf/src/feature/feature_extractor.h#L36-L87). Implementing a new feature extractor should be relatively painless. Create a new `VmafFeatureExtractor` and add it to the build as well as the `feature_extractor_list[]`. See [this diff](https://github.com/Netflix/vmaf/commit/fd3c79697c7e06586aa5b9cda8db0d9aedfd70c5) for an example. Once you do this your feature extractor may be registered and used inside of `libvmaf` via `vmaf_use_feature()` or `vmaf_use_features_from_model()`. To invoke this feature extractor directly from the command line with `vmaf` use the `--feature` flag.

`VmafFeatureExtractor` is a feature extraction class with just a few callbacks. If you have preallocations and/or precomputations to make, it is best to do this in the `.init()` callback and store the output in `.priv`.  This is a place for custom data which is available for all subsequent callbacks. If you allocate anything in `.init()` be sure to clean it up in the `.close()` callback.

The remainder of your work should take place in the `.extract()` callback. This callback is called for every pair of input pictures. Read the pixel data make some computations and then write the output(s) to the `VmafFeatureCollector` via the `vmaf_feature_collector_append()` api. An important thing to know about this callback is that it can (and probably is) being called in an arbitrary order. If your feature extractor has a temporal requirement (i.e. `motion`), set the `VMAF_FEATURE_EXTRACTOR_TEMPORAL` flag and the `VmafFeatureExtractorContext` will ensure that this callback is executed in serial. For an example of a feature extractor with a temporal dependency see the [motion](https://github.com/Netflix/vmaf/blob/master/libvmaf/src/feature/integer_motion.c) feature extractor.

If the `VMAF_FEATURE_EXTRACTOR_TEMPORAL` is set, it is likely that you have buffers that need flushing. If this is the case, `.flush()` is called in a loop until something non-zero is returned.

## Example

The following example shows a comparison using a pair of yuv inputs (`src01_hrc00_576x324.yuv`, `src01_hrc01_576x324.yuv`). In addition to VMAF which is enabled with the model `../model/vmaf_float_v0.6.1.pkl`, the `psnr` metric is also computed and logged.

```sh
wget https://github.com/Netflix/vmaf_resource/raw/master/python/test/resource/yuv/src01_hrc00_576x324.yuv
wget https://github.com/Netflix/vmaf_resource/raw/master/python/test/resource/yuv/src01_hrc01_576x324.yuv
./build/tools/vmaf \
    --reference src01_hrc00_576x324.yuv \
    --distorted src01_hrc01_576x324.yuv \
    --width 576 --height 324 --pixel_format 420 --bitdepth 8 \
    --model path=../model/vmaf_float_v0.6.1.pkl \
    --feature psnr \
    --output /dev/stdout
```

The output should look like this:
```html
<VMAF version="1.5.3">
  <params qualityWidth="576" qualityHeight="324" />
  <fyi fps="59.60" />
  <frames>
    <frame frameNum="0" adm2="0.962086" adm_scale0="0.946315" adm_scale1="0.939017" adm_scale2="0.957478" adm_scale3="0.980893" motion2="0.000000" vif_scale0="0.505393" vif_scale1="0.878155" vif_scale2="0.937589" vif_scale3="0.964357" psnr_y="34.760779" psnr_cb="39.229987" psnr_cr="41.349703" vmaf="83.849994" />
    ...
    <frame frameNum="47" adm2="0.946169" adm_scale0="0.924315" adm_scale1="0.908033" adm_scale2="0.943376" adm_scale3="0.971125" motion2="5.443212" vif_scale0="0.416110" vif_scale1="0.811470" vif_scale2="0.893364" vif_scale3="0.934516" psnr_y="31.888613" psnr_cb="38.667124" psnr_cr="41.353846" vmaf="83.019174" />
  </frames>
</VMAF>
```
