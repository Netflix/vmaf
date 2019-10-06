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

Here, `read_frame` is a callback function which can be used to pass data from a program to VMAF. `user_data` is a program specific data that can be used by the callback function. For sample usage of `compute_vmaf`, refer to [`src/libvmaf/src/main.cpp`](../../libvmaf/src/main.cpp).

To test the library, run:

```
make testlib
```

This command will build an executable `testlib` using this library together with [`src/libvmaf/src/main.cpp`](../../libvmaf/src/main.cpp).

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
"[0:v]scale=1920x1080:flags=bicubic[main];[main][1:v]libvmaf" -f null -
```

Here `main.mpg` is a downscaled and encoded video and `ref.mpg` is its reference source at 1080p. The command scales the first input video (`0:v`) and forwards it to VMAF (`libvmaf`) with the label `main`, where it is compared against the second input reference video, `1:v`. Bicubic upsampling is used (also see the [techblog](https://medium.com/netflix-techblog/vmaf-the-journey-continues-44b51ee9ed12) for the recommendation on upsampling methods).

See the [FFmpeg's guide to libvmaf](https://ffmpeg.org/ffmpeg-filters.html#libvmaf), the [FFmpeg Filtering Guide](https://trac.ffmpeg.org/wiki/FilteringGuide) for more examples of complex filters, and the [Scaling Guide](https://trac.ffmpeg.org/wiki/Scaling) for information about scaling and using different scaling algorithms.
