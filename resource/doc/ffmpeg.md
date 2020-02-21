### Use `libvmaf` with FFmpeg

After installing `libvmaf`, you can use it with FFmpeg. Under FFmpeg directory, configure, build and install FFmpeg with:

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
