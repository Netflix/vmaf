# Using VMAF with FFmpeg

After [installing](../../libvmaf/README.md#install) `libvmaf`, you can use it with [FFmpeg](http://ffmpeg.org/). Under the FFmpeg directory, configure, build and install FFmpeg with:

```shell script
./configure --enable-libvmaf
make -j4
make install
```

Using FFmpeg+libvmaf is very powerful, as you can create complex filters to calculate VMAF directly on videos of different encoding formats and resolutions. For the best practices of computing VMAF at the right resolution, refer to our [tech blog](https://medium.com/netflix-techblog/vmaf-the-journey-continues-44b51ee9ed12).

We provide a few examples how you can construct the FFmpeg command line and use VMAF as a filter. Note that you may need to download the test videos from [vmaf_resource](https://github.com/Netflix/vmaf_resource/tree/master/python/test/resource).

Below is an example on how you can run FFmpeg+libvmaf on a pair of YUV files. First, download the reference video [`src01_hrc00_576x324.yuv`](https://github.com/Netflix/vmaf_resource/blob/master/python/test/resource/yuv/src01_hrc00_576x324.yuv) and the distorted video [`src01_hrc01_576x324.yuv`](https://github.com/Netflix/vmaf_resource/blob/master/python/test/resource/yuv/src01_hrc01_576x324.yuv). `-r 24` sets the frame rate (note that it needs to be before `-i`), and `PTS-STARTPTS` synchronizes the PTS (presentation timestamp) of the two videos (this is crucial if one of your videos does not start at PTS 0, for example, if you cut your video out of a long video stream). It is important to set the frame rate and the PTS right, since FFmpeg filters synchronize based on timestamps instead of frames.

The `log_path` is set to standard output `/dev/stdout`. It uses the `model_path` at location `/usr/local/share/model/vmaf_float_v0.6.1.json` (which is the default and can be omitted).

```shell script
ffmpeg -video_size 576x324 -r 24 -pixel_format yuv420p -i src01_hrc00_576x324.yuv \
    -video_size 576x324 -r 24 -pixel_format yuv420p -i src01_hrc01_576x324.yuv \
    -lavfi "[0:v]setpts=PTS-STARTPTS[reference]; \
            [1:v]setpts=PTS-STARTPTS[distorted]; \
            [distorted][reference]libvmaf=log_fmt=xml:log_path=/dev/stdout:model_path={your_vmaf_dir}/model/vmaf_v0.6.1.json:n_threads=4" \
    -f null -
```

The expected output is:

```shell script
[libvmaf @ 0x7fcfa3403980] VMAF score: 76.668905
```

Below is a more complicated example where the inputs are packaged `.mp4` files. It takes in 1) a reference video [`Seeking_30_480_1050.mp4`](https://github.com/Netflix/vmaf_resource/blob/master/python/test/resource/mp4/Seeking_30_480_1050.mp4) of 480p and 2) a distorted video [`Seeking_10_288_375.mp4`](https://github.com/Netflix/vmaf_resource/blob/master/python/test/resource/mp4/Seeking_10_288_375.mp4) of 288p upsampled to `720x480` using bicubic, and compute VMAF on the two 480p videos. Bicubic is used as the recommended upsampling method (also see the [techblog](https://medium.com/netflix-techblog/vmaf-the-journey-continues-44b51ee9ed12) for more details).

```shell script
ffmpeg \
    -r 24 -i Seeking_30_480_1050.mp4 \
    -r 24 -i Seeking_10_288_375.mp4 \
    -lavfi "[0:v]setpts=PTS-STARTPTS[reference]; \
            [1:v]scale=720:480:flags=bicubic,setpts=PTS-STARTPTS[distorted]; \
            [distorted][reference]libvmaf=log_fmt=xml:log_path=/dev/stdout:model_path={your_vmaf_dir}/model/vmaf_v0.6.1.json:n_threads=4" \
    -f null -
```

The expected output is:

```shell script
[libvmaf @ 0x7fb5b672bc00] VMAF score: 51.017497
```

See the [FFmpeg's guide to libvmaf](https://ffmpeg.org/ffmpeg-filters.html#libvmaf), the [FFmpeg Filtering Guide](https://trac.ffmpeg.org/wiki/FilteringGuide) for more examples of complex filters, and the [Scaling Guide](https://trac.ffmpeg.org/wiki/Scaling) for information about scaling and using different scaling algorithms.

## Note about the model path on Windows

Due to Windows not having a good default for where to pull the VMAF model from, you will always need to specify `model_path` when calling libvmaf through `ffmpeg`. However, you will need to be careful about the path you pass to `model_path`.

If you are using a relative path for your `model_path`, you can completely ignore this whole section, else if you are trying to use an absolute Windows path (`D:\mypath\vmaf_v0.6.1.json`) for your `model_path` argument, you will need to be careful so `ffmpeg` passes the right path to `libvmaf`.

The final command line will depend on what shell you are running `ffmpeg` through, so you will need to go through the following steps to make sure your path is okay.

1. Convert all of the backslashes `\` to forward slashes `/` (`D:/mypath/vmaf_v0.6.1.json`)
2. Escape the colon `:` character by using a backslash `\` (`D\:/mypath/vmaf_v0.6.1.json`)
3. Then escape that backslash with another backslash (`D\\:/mypath/vmaf_v0.6.1.json`)
4. The next step will depend on the shell that will run `ffmpeg`:
    - For PowerShell and Command Prompt, this will be enough and your final `ffmpeg` command line will look something like
    ```powershell
    ./ffmpeg.exe -i dist.y4m -i ref.y4m \
        -lavfi libvmaf=model_path="D\\:/mypath/vmaf_v0.6.1.json" \
        -f null -
    ```
    Note: I only quoted the path part for trivial reasons and in this specific case, it can be unquoted or you can quote the whole part after lavfi starting from `libvmaf` to `json` and it should give the same result due to neither shell treating the `\` as a special character

    - For bash or specifically msys2 bash, it has some additional considerations. The first thing to know is that bash treats the backslash character `\` a bit special in that it's an escape character normally when not put inside single quotes. The second thing to know is that msys2's bash attempts convert a posix-like path (`/mingw64/share/model/vmaf_v0.6.1.json`) to a Windows mixed path (`D:/msys2/mingw64/share/model/vmaf_v0.6.1.json`) when passing arguments to a program. Normally, this would be fine, however, in our case, this works against us since we cannot allow it to convert the path to a normal path with an un-escaped colon. For this, we will need to not only escape the escaped backslash, but we will also need to pass the `MSYS2_ARG_CONV_EXCL` environment variable with the value of `*` to make sure it doesn't apply that special conversion on any of the arguments

    ```bash
    MSYS2_ARG_CONV_EXCL="*" \
        ./ffmpeg.exe -i dist.y4m -i ref.y4m -lavfi \
        libvmaf=model_path="D\\\:/mypath/vmaf_v0.6.1.json" -f null -
    ```

    Note: in this case, the quotes are not as trivial as the PowerShell/cmd version, as removing the quotes entirely will require you to re-escape the backslash resulting in 4 total backslashes, but quoting the whole argument will be fine.

    Second Note: if you use single quotes around the path, it will be fine as well and the final command line would look like

    ```bash
    MSYS2_ARG_CONV_EXCL="*" \
        ./ffmpeg.exe -i dist.y4m -i ref.y4m -lavfi \
        libvmaf=model_path='D\\:/mypath/vmaf_v0.6.1.json' -f null -
    ```

    with only a double backslash instead of a triple.

## External resources

Refer to [this page](external_resource.md) for a list of FFmpeg-based 3rd-party tools.
