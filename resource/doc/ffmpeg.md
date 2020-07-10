Using libvmaf with FFmpeg
===================

After installing libvmaf, you can use it with FFmpeg. Under FFmpeg directory, configure, build and install FFmpeg with:

```
./configure --enable-libvmaf
make -j4
make install
```

Using FFmpeg+libvmaf is very powerful, as you can create complex filters to calculate VMAF directly on videos of different encoding formats and resolutions. For the best practices of computing VMAF at the right resolution, refer to our [techblog](https://medium.com/netflix-techblog/vmaf-the-journey-continues-44b51ee9ed12). 

We provide a few examples how you can construct the FFmpeg command line and use VMAF as a filter. Note that you may need to download the test videos from [vmaf_resource](https://github.com/Netflix/vmaf_resource/tree/master/python/test/resource).

Below is an example on how you can run FFmpeg+libvmaf on a pair of YUV files. Note that `src01_hrc00_576x324.yuv` is the reference video and `src01_hrc01_576x324.yuv` is the distorted video. `-r 24` sets the frame rate, and `PTS-STARTPTS` synchronizes the PTS (presentation timestamp) of the two videos. Note that it is important to set the frame rate and the PTS right, since FFmpeg filters synchronize based on timestamps instead of frames. 

The `log_path` is set to standard output `/dev/stdout`. It uses the `model_path` at location `/usr/local/share/model/vmaf_v0.6.1.pkl` (which is the default and can be omitted).

```
ffmpeg -video_size 576x324 -r 24 -pixel_format yuv420p -i yuv/src01_hrc00_576x324.yuv \
    -video_size 576x324 -r 24 -pixel_format yuv420p -i yuv/src01_hrc01_576x324.yuv \
    -lavfi "[0:v]setpts=PTS-STARTPTS[reference]; \
            [1:v]setpts=PTS-STARTPTS[distorted]; \
            [distorted][reference]libvmaf=log_fmt=xml:log_path=/dev/stdout:model_path=/usr/local/share/model/vmaf_v0.6.1.pkl" \
    -f null -
```

The expected output is:

```
...
[libvmaf @ 0x7fcfa3403980] VMAF score: 76.684256
```

Below is a more complicated example where the inputs are packaged .mp4 files. It takes in 1) a reference video `Seeking_30_480_1050.mp4` of 480p and 2) a distorted video `Seeking_10_288_375.mp4` of 288p upsampled to `720x480` using bicubic, and compute VMAF on the two 480p videos. Bicubic is used as the recommended upsampling method (also see the [techblog](https://medium.com/netflix-techblog/vmaf-the-journey-continues-44b51ee9ed12) for more details).

```
ffmpeg -r 24 -i  mp4/Seeking_30_480_1050.mp4 \
    -r 24 -i mp4/Seeking_10_288_375.mp4 \
    -lavfi "[0:v]setpts=PTS-STARTPTS[reference]; \
            [1:v]scale=720:480:flags=bicubic,setpts=PTS-STARTPTS[distorted]; \
            [distorted][reference]libvmaf=log_fmt=xml:log_path=/dev/stdout:model_path=/usr/local/share/model/vmaf_v0.6.1.pkl" \
    -f null -
```

The expected output is:

```
...
[libvmaf @ 0x7fb5b672bc00] VMAF score: 51.037407
```

See the [FFmpeg's guide to libvmaf](https://ffmpeg.org/ffmpeg-filters.html#libvmaf), the [FFmpeg Filtering Guide](https://trac.ffmpeg.org/wiki/FilteringGuide) for more examples of complex filters, and the [Scaling Guide](https://trac.ffmpeg.org/wiki/Scaling) for information about scaling and using different scaling algorithms.
