# `libvmaf`

## Prerequisites

For building, you need the following:

- [Python3](https://www.python.org/download/releases/3.0/) (3.6 or higher)
- [Meson](https://mesonbuild.com/) (0.56.1 or higher)
- [Ninja](https://ninja-build.org/) (1.7.1 or higher)
- [NASM](https://www.nasm.us/) (for x86 builds only, 2.13.02 or higher)
- `xxd`

Follow the steps below to set up a clean virtual environment and install the tools:
```
python3 -m pip install virtualenv
python3 -m virtualenv .venv
source .venv/bin/activate
pip install meson
sudo [package-manager] install nasm ninja-build doxygen xxd
```
You need to invoke `[package-manager]` depending on which system you are on: `apt-get` for Ubuntu and Debian, `yum` for CentOS and RHEL, `dnf` for Fedora, `zypper` for openSUSE, `brew` for MacOS (no `sudo`).

Ninja package name might be `ninja` or `ninja-build`, depending on your system. Check [the pre-built ninja packages](https://github.com/ninja-build/ninja/wiki/Pre-built-Ninja-packages) before running the install command.

## Compile

Run:

```
meson setup build --buildtype release
```

Special cases:
- add `-Denable_float=true` flag in the rare case if you want to use the floating-point feature extractors.
- add `-Denable_avx512=true` to support wider SIMD instructions to achieve the fastest processing on supported CPUs
- add `-Denable_cuda=true` to build with CUDA support, which requires `nvcc` for compilation (tested with CUDA >= 11)
- add `-Denable_nvtx=true` to build with [NVTX](https://github.com/NVIDIA/NVTX) marker support, which enables easy profiling using Nsight Systems

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

Install the library, headers, and the `vmaf` command line tool:

```
ninja -vC build install
```

This will install the following files:

```
├── bin
│   └── vmaf
├── include
│   └── libvmaf
│       ├── feature.h
│       ├── libvmaf.h
│       ├── model.h
│       ├── picture.h
│       └── version.h
└── lib
    ├── libvmaf.1.dylib
    ├── libvmaf.a
    ├── libvmaf.dylib -> libvmaf.1.dylib
    └── pkgconfig
        └── libvmaf.pc
```

## Documentation

Generate HTML documentation with:

```
ninja -vC build doc/html
```

## VMAF Models

`libvmaf` now has a number of VMAF models built-in. This means that no external VMAF model files are required, since the models are compiled into and read directly from the library. If you do not wish to compile the built-in models into your build, you may disable them with `-Dbuilt_in_models=false`. Previous versions of this library required a `.pkl` model file. Since libvmaf v2.0.0, these `.pkl` model files have been deprecated in favor of `.json` model files. If you have a previously trained `.pkl` model you would like to convert to `.json`, this [Python conversion script](../python/vmaf/script/convert_model_from_pkl_to_json.py) is available. 

## `vmaf`

A command line tool called `vmaf` is included as part of the build/installation. See the `vmaf` [README.md](tools/README.md) for details.

## API Walkthrough

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

For complete API documentation, see [libvmaf.h](include/libvmaf/libvmaf.h). For an example of using the API to create the `vmaf` command line tool, see [vmaf.c](tools/vmaf.c).

## Contributing a new VmafFeatureExtractor

To write a new feature extractor, please first familiarize yourself with the [VmafFeatureExtractor API](https://github.com/Netflix/vmaf/blob/master/libvmaf/src/feature/feature_extractor.h#L36-L87) documentation.

Create a new `VmafFeatureExtractor` and add it to the build as well as the `feature_extractor_list[]`. See [this diff](https://github.com/Netflix/vmaf/commit/fd3c79697c7e06586aa5b9cda8db0d9aedfd70c5) for an example. Once you do this your feature extractor may be registered and used inside of `libvmaf` via `vmaf_use_feature()` or `vmaf_use_features_from_model()`. To invoke this feature extractor directly from the command line with `vmaf` use the `--feature` flag.

`VmafFeatureExtractor` is a feature extraction class with just a few callbacks. If you have preallocations and/or precomputations to make, it is best to do this in the `.init()` callback and store the output in `.priv`.  This is a place for custom data which is available for all subsequent callbacks. If you allocate anything in `.init()` be sure to clean it up in the `.close()` callback.

The remainder of your work should take place in the `.extract()` callback. This callback is called for every pair of input pictures. Read the pixel data make some computations and then write the output(s) to the `VmafFeatureCollector` via the `vmaf_feature_collector_append()` api. An important thing to know about this callback is that it can (and probably is) being called in an arbitrary order. If your feature extractor has a temporal requirement (i.e. `motion`), set the `VMAF_FEATURE_EXTRACTOR_TEMPORAL` flag and the `VmafFeatureExtractorContext` will ensure that this callback is executed in serial. For an example of a feature extractor with a temporal dependency see the [motion](https://github.com/Netflix/vmaf/blob/master/libvmaf/src/feature/integer_motion.c) feature extractor.

If the `VMAF_FEATURE_EXTRACTOR_TEMPORAL` is set, it is likely that you have buffers that need flushing. If this is the case, `.flush()` is called in a loop until something non-zero is returned.
