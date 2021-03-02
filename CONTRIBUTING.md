# Contributing to VMAF

If you would like to contribute code to the VMAF repository, you can do so through GitHub by forking the repository and sending a pull request. When submitting code, please make every effort to follow existing conventions and style in order to keep the code as readable as possible.

## License

By contributing your code, you agree to license your contribution under the terms of the [BSD+Patent](https://opensource.org/licenses/BSDplusPatent). Your contributions should also include the following header:

```
/**
 * Copyright 2016-2020 [the original author or authors].
 * 
 * Licensed under the BSD+Patent License (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * https://opensource.org/licenses/BSDplusPatent
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
```

## Ways to Contribute

There are many ways you can contribute, and no contribution is too small. To name a few:
- Submitting a bugfix
- Improving documentation
- Making the code run on a new platform
- Robustifying the build system
- Improving the CI loop
- Improving code coverage by adding tests
- Optimizing the speed
- Implementing a well-known quality metric
- Implementing a custom VMAF model for a specific use case

## Algorithmic Contribution

This section focuses on *algorithmic contribution*, which cover two main use cases: 
  - Implementing a *well-known quality metric* that can be found in the literature
  - Implementing a *custom VMAF model*, using new elementary features and trained on a specific dataset

For both cases, one can follow the procedure below:
- First, implement the feature extractor(s), by subclassing the `FeatureExtractor` Python class. A new `FeatureExtractor` class created could be either 1) native Python implementation, or 2) calling a subprocess implemented in a different language (in C or in Matlab, for example).
- Second, implement the quality runner, by 1) creating a new `QualityRunner` class as a thin wrapper around the new `FeatureExtractor` created, or 2) using the established `VmafQualityRunner` class but training a custom VMAF model.

For the concepts of `FeatureExtractor`, `QualityRunner` and `VmafQualityRunner`, please refer to the [Core Classes](resource/doc/VMAF_Python_library.md#core-classes) section of the VMAF Python library documentation.

### Creating A New `FeatureExtractor`

#### Native Python
To create a subclass of `FeatureExtractor` in native Python code, a minimalist example to follow is the `PypsnrFeatureExtractor` class ("Py-PSNR", see the [code diff](https://github.com/Netflix/vmaf/commit/e698b4d788fb3dcabdc4df2fd1bffe88dc0d3ecd)). The following steps discuss the implementation strategy.
  - Create a subclass of the `FeatureExtractor`. Make sure to specify the `TYPE`, `VERSION` and `ATOM_FEATURES`, which play a role in caching the features extracted in `ResultStore`. Optionally, one can specify a `DERIVED_ATOM_FEATURES` field (refer to the `_post_process_result()` method section for more details).
  - Implement the `_generate_result()` method, which is responsible for the heavy-lifting feature calculation. This method is called by the `_run_on_asset()` method in the parent `Executor` class, which preprocesses the reference and distorted videos to prepare them in a proper YUV format. It calls two `YuvReader` to read the video frame-by-frame, and run computations on them. The result is written to the `log_file_path` file path.
  - Implement the `_get_feature_scores()` method, which parses the result from the `log_file_path` and put it in a dictionary to prepare for a new `Result` object.
  - Optionally, implement the `_post_process_result()` method to compute the `DERIVED_ATOM_FEATURES` from the `ATOM_FEATURES`. Refer to the [Python Calling Matlab](#python-calling-matlab) section for a specific example.
  - Create [test cases](https://github.com/Netflix/vmaf/commit/e698b4d788fb3dcabdc4df2fd1bffe88dc0d3ecd#diff-5b58c2457df7e9b30b0a678d6f79b1caaad6c3f036edfadb6ca9fb0955bede33R630) to lock the numerical results.  
  - Notice that `FeatureExtractor` allows one to pass in optional parameters that has an effect on the numerical result. This is demonstrated by the `max_db` parameter in `PypsnrFeatureExtractor` (see this [test case](https://github.com/Netflix/vmaf/commit/e698b4d788fb3dcabdc4df2fd1bffe88dc0d3ecd#diff-5b58c2457df7e9b30b0a678d6f79b1caaad6c3f036edfadb6ca9fb0955bede33R730) for an example use case). By default, there is a bit depth-dependent maximum PSNR value (see [here](FAQ.md#q-why-are-the-psnr-values-capped-at-60-db-for-8-bit-inputs-and-72-db-for-12-bit-inputs-in-the-packages-implementation) for the motivation behind), but the `max_db` parameter specified in the `optional_dict` input allows one to specify a maximum value.

#### Python Calling `libvmaf` in C
Very often the feature extractor implementation is in the C library `libvmaf`. In this case we simply create a thin Python `FeatureExtractor` subclass to call the `vmaf` command line executable. For more information on implementing a new feature extractor in `libvmaf`, refer to [this section](libvmaf/README.md#contributing-a-new-vmaffeatureextractor). An example to follow is the PSNR-HVS feature extractor (see the [code diff](https://github.com/Netflix/vmaf/commit/ce2ad1af0b1ba8dd1fbae3e03da0329f078e6bc6)). The following steps discuss the implementation strategy.
  - Add a new feature extractor implementation `vmaf_fex_psnr_hvs` in file `libvmaf/src/feature/third_party/xiph/psnr_hvs.c`. It is recommended to put the code under directory `third_party/[org]`.
  - In `libvmaf/src/feature/feature_extractor.c`:
    - Declare the new feature extractor as `extern`:
        ```c
        extern VmafFeatureExtractor vmaf_fex_psnr_hvs;
        ```
    - Add the new feature extractor to the `feature_extractor_list`:
        ```c
        static VmafFeatureExtractor *feature_extractor_list[] = {
        ...
        &vmaf_fex_psnr_hvs,
        ...
        };
        ```
  - In `libvmaf/src/meson.build`, add the new `psnr_hvs.c` file to the `libvmaf_feature_sources` list:
      ```c
        libvmaf_feature_sources = [
        ...
        feature_src_dir + 'third_party/xiph/psnr_hvs.c',
        ...
        ]    
      ```
  - Create a Python wrapper class `PsnrhvsFeatureExtractor` in `python/vmaf/third_party/xiph/vmafexec_feature_extractor.py` (Note: you also need to make `vmaf.third_party.xiph a Python package by adding the `__init__.py` files in corresponding directories.)
  - Add a test case for `PsnrhvsFeatureExtractor` in `python/test/third_party/xiph/vmafexec_feature_extractor_test.py` to lock the numerical values.
      
#### Python Calling Matlab
Oftentimes for a well-known quality metric, its Matlab implementation already exists. The VMAF Python library allows directly plugging in the Matlab code by creating a thin Python `MatlabFeatureExtractor` subclass to call the Matlab script. An example to follow is the STRRED feature extractor (see [implementation](https://github.com/Netflix/vmaf/blob/master/python/vmaf/core/matlab_feature_extractor.py#L24-L113) and [test case](https://github.com/Netflix/vmaf/blob/e698b4d788fb3dcabdc4df2fd1bffe88dc0d3ecd/python/test/extra/feature_extractor_extratest.py#L25-L44)). The following steps discuss the implementation strategy.
  - First, Matlab must be pre-installed and its path specified in the `MATLAB_PATH` field in the `python/vmaf/externals.py` file. If not, a user will be prompt with the installation instructions.
  - Create a subclass of the `MatlabFeatureExtractor`. Make sure to specify the `TYPE`, `VERSION` and `ATOM_FEATURES`, which play a role in caching the features extracted in `ResultStore`. Optionally, one can specify a `DERIVED_ATOM_FEATURES` field (refer to the `_post_process_result()` method section for more details).
  - Implement the `_generate_result()` method, which is responsible for calling the Matlab command line to output the result to the file at `log_file_path`.
  - Implement the `_get_feature_scores()` method, which parses the result from the `log_file_path` and put it in a dictionary to prepare for a new `Result` object. In the case of the `StrredFeatureExtractor`, the default method provided by the `FeatureExtractor` superclass can be directly used as the Matlab script's data format is compatible with it, hence the implementation is skipped. But in general, this methods needs to be implemented.
  - Optionally, implement the `_post_process_result()` method to compute the `DERIVED_ATOM_FEATURES` from the `ATOM_FEATURES`. In the case of STRRED, the `strred` feature can be derived from the `srred` and `trred` features via simple computation:
    ```python
    strred = srred * trred
    ```
    Therefore, we define the `strred` feature as "derived" and skip the caching process.
  - Create [test cases]((https://github.com/Netflix/vmaf/blob/e698b4d788fb3dcabdc4df2fd1bffe88dc0d3ecd/python/test/extra/feature_extractor_extratest.py#L25-L44)) to lock the numerical results.

### Creating A Thin `QualityRunner` Wrapper
For the use case of implementing a *well-known quality metric*, after the feature extractor is created, the job is almost done. But to run tests and scripts uniformly, we need to create a thin wrapper of the `QualityRunner` subclass around the new `FeatureExtractor` already created. A good example of this is the `SsimQualityRunner` class (see [code](https://github.com/Netflix/vmaf/blob/master/python/vmaf/core/quality_runner.py#L815)). One simply needs to create a subclass from `QualityRunnerFromFeatureExtractor`, and override the `_get_feature_extractor_class()` and `_get_feature_key_for_score()` methods.

### Training A Custom VMAF Model

#### Creating A New `TrainTestModel`

#### Calling `run_vmaf_training` Script






