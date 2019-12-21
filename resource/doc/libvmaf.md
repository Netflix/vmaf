`libvmaf.a` -- A Static Library of VMAF
===================

VMAF is now packaged into a static library called `libvmaf.a`.

To build `libvmaf.a`, make the `ptools` library first, followed by `libvmaf`:

```
cd src/ptools; make; cd ../../src/libvmaf; make; cd ../..;
```

The static library built is at `src/libvmaf/libvmaf.a`. To install the library, run:

```
sudo make install
```

This copies the library header `libvmaf.h` under `usr/local/include`, library `libvmaf.a `under `usr/local/lib` and all the model files under `usr/local/share`. You can use the header `libvmaf.h` in your program. It contains an API which can be called from any C/C++ program:

```
int compute_vmaf(double* vmaf_score, char* fmt, int width, int height, 
int (*read_frame)(float *ref_data, float *main_data, float *temp_data, int stride, void *user_data), 
void *user_data, char *model_path, char *log_path, char *log_fmt, int disable_clip, 
int disable_avx, int enable_transform, int phone_model, int do_psnr, int do_ssim, 
int do_ms_ssim, char *pool_method, int thread, int subsample, int enable_conf_interval);
```

Here, `read_frame` is a callback function which can be used to pass data from a program to VMAF. `user_data` is a program specific data that can be used by the callback function. For sample usage of `compute_vmaf`, refer to [`libvmaf/tools/main.cpp`](../../libvmaf/tools/main.cpp).

To test the library, run:

```
make testlib
```

This command will build an executable `testlib` using this library together with [`libvmaf/tools/main.cpp`](../../libvmaf/tools/main.cpp).

To uninstall the library run:

```
make uninstall
```

### Use `libvmaf.a` with FFmpeg

After installing `libvmaf.a`, you can use it with FFmpeg. Under FFmpeg directory, configure, build and install FFmpeg with:

```
./configure --enable-libvmaf --enable-version3
make install
```

Using FFmpeg with libvmaf is very powerful, as you can create complex filters to calculate VMAF directly on videos of different encoding formats and resolutions. For the best practices of computing VMAF at the right resolution, refer to our [techblog](https://medium.com/netflix-techblog/vmaf-the-journey-continues-44b51ee9ed12). Below is an example on how you can compare a downscaled video with its original 1080p source:

```
ffmpeg -i main.mpg -i ref.mpg -filter_complex \
"[0:v]scale=1920x1080:flags=bicubic,setpts=PTS-STARTPTS[main];[1:v]setpts=PTS-STARTPTS[ref];[main][ref]libvmaf" -an -f null -
```

Here `main.mpg` is a downscaled and encoded video and `ref.mpg` is its reference source at 1080p. The command scales the first input video (`0:v`) and forwards it to VMAF (`libvmaf`) with the label `main`, where it is compared against the label `ref` which corresponds to the second input reference video, `1:v`. We enforce the scaling algorithm to bicubic, since the default scaling algorithm is bilinear (also see the [techblog](https://medium.com/netflix-techblog/vmaf-the-journey-continues-44b51ee9ed12) for the recommendation on upsampling methods). Note that we need to specify the `setpts=PTS-STARTPTS` option to ensure that the PTS is correctly
aligned. We can also the `-an` option to disregard the presence of audio.

See the [FFmpeg's guide to libvmaf](https://ffmpeg.org/ffmpeg-filters.html#libvmaf), the [FFmpeg Filtering Guide](https://trac.ffmpeg.org/wiki/FilteringGuide) for more examples of complex filters, and the [Scaling Guide](https://trac.ffmpeg.org/wiki/Scaling) for information about scaling and using different scaling algorithms. Please refer to the next section for some additional examples on more complex cases.

### Use `libvmaf.a` with FFmpeg - More examples

The above command line works well for most cases, but additional steps might be necessary.

For example, when the frame rates between reference and encoded video do not match, e.g., when the source is 60 fps and the encode is 30 fps:

```
ffmpeg -r 1 -i main.mpg -r 2 -i ref.mpg -filter_complex \
"[0:v]scale=1920x1080:flags=bicubic,setpts=PTS-STARTPTS[main];[1:v]select='not(mod(n,2))',setpts=PTS-STARTPTS[ref];[main][ref]libvmaf" -an -f null -
```

Here we explicitly state that the reference video has double the frame rate and also use the select filter to select half of the reference frames to calculate VMAF.

If we want a specific model to be used (e.g. the `baz.pkl` model file under the `/foo/bar/` directory):

```
ffmpeg -i main.mpg -i ref.mpg -filter_complex \
"[0:v]scale=1920x1080:flags=bicubic,setpts=PTS-STARTPTS[main];[1:v]setpts=PTS-STARTPTS[ref];[main][ref]libvmaf='model_path=/foo/bar/baz.pkl'" -an -f null -
```

If we want to specify a model that relies on bootstrapping (e.g. the 0.6.3 model file under the `/foo/bar/` directory), the `enable_confidence_interval` option needs to be set:

```
ffmpeg -i main.mpg -i ref.mpg -filter_complex \
"[0:v]scale=1920x1080:flags=bicubic,setpts=PTS-STARTPTS[main];[1:v]setpts=PTS-STARTPTS[ref];[main][ref]libvmaf='model_path=/foo/bar/vmaf_b_v0.6.3.pkl:enable_conf_interval=1'" -an -f null -
```

If we want to store the results in an XML file with a given name (e.g. under `/foo/bar/baz/results.xml`):

```
ffmpeg -i main.mpg -i ref.mpg -filter_complex \
"[0:v]scale=1920x1080:flags=bicubic,setpts=PTS-STARTPTS[main];[1:v]setpts=PTS-STARTPTS[ref];[main][ref]libvmaf='log_path=/foo/bar/baz/results.xml:model_path=/foo/bar/baz.pkl'" -an -f null -
```

If we want to enable the calculation of an additional metric, like SSIM:

```
ffmpeg -i main.mpg -i ref.mpg -filter_complex \
"[0:v]scale=1920x1080:flags=bicubic,setpts=PTS-STARTPTS[main];[1:v]setpts=PTS-STARTPTS[ref];[main][ref]libvmaf='ssim=1:model_path=/foo/bar/baz.pkl'" -an -f null -
```

If we want use a single thread:

```
ffmpeg -i main.mpg -i ref.mpg -filter_complex \
"[0:v]scale=1920x1080:flags=bicubic,setpts=PTS-STARTPTS[main];[1:v]setpts=PTS-STARTPTS[ref];[main][ref]libvmaf='n_threads=1:ssim=1:model_path=/foo/bar/baz.pkl'" -an -f null -
```
