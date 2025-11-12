# VMAF Usage through Docker

Install Docker, then, from the VMAF directory, run:

```shell script
docker build -t vmaf .
```

And to use it, just run:

```shell script
docker run --rm vmaf [CLI]
```

Where `[CLI]` is one of `run_vmaf`.

For example, if you are under root, to run `run_vmaf` on a sample reference/distorted video pair under `resource/yuv`:

```shell script
docker run --rm -v $(pwd):/files vmaf \
    yuv420p 576 324 \
    /files/src01_hrc00_576x324.yuv \
    /files/src01_hrc01_576x324.yuv \
    --out-fmt json
```

Note that you need to first download the test videos from [vmaf_resource](https://github.com/Netflix/vmaf_resource/tree/master/python/test/resource).

```shell script
wget https://github.com/Netflix/vmaf_resource/raw/master/python/test/resource/yuv/src01_hrc00_576x324.yuv
wget https://github.com/Netflix/vmaf_resource/raw/master/python/test/resource/yuv/src01_hrc01_576x324.yuv
```

## Use docker with GPU support

To run docker containers with GPU support you have to install the [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The above built container already has CUDA support inside and can use CUDA when the nvidia runtime is in use:
```shell script
docker run --gpus all --rm -v $(pwd):/files vmaf \
    yuv420p 576 324 \
    /files/src01_hrc00_576x324.yuv \
    /files/src01_hrc01_576x324.yuv \
    --out-fmt json
```

While CUDA support is already fast using the `vmaf` tool the tool itself is heavily IO bottlenecked and a usage with ffmpeg and corresponding hardware decoders is much more efficient. To enable that we provide a separate Dockerfile that builds ffmpeg with cuda support and the vmaf filter. To build the container use:
```shell script
docker build -f Dockerfile.ffmpeg -t ffmpeg_vmaf .
```
Usage of the container is very similar to the above examples. For example to run the vmaf filter on two mp4 files use:
```shell script
wget https://ultravideo.fi/video/Beauty_3840x2160_120fps_420_8bit_HEVC_RAW.hevc

docker run --gpus all  -e NVIDIA_DRIVER_CAPABILITIES=compute,video -v $(pwd):/files ffmpeg_vmaf  \
  -y -hwaccel cuda -hwaccel_output_format cuda -i /files/Beauty_3840x2160_120fps_420_8bit_HEVC_RAW.hevc \
  -fps_mode vfr -c:a copy -c:v hevc_nvenc -b:v 2M /files/dist.mp4

docker run --gpus all  -e NVIDIA_DRIVER_CAPABILITIES=compute,video -v $(pwd):/files ffmpeg_vmaf \
    -hwaccel cuda -hwaccel_output_format cuda -i /files/Beauty_3840x2160_120fps_420_8bit_HEVC_RAW.hevc \
    -hwaccel cuda -hwaccel_output_format cuda -i /files/dist.mp4 \
    -filter_complex "[0:v]scale_cuda=format=yuv420p[ref];[1:v]scale_cuda=format=yuv420p[dist];[ref][dist]libvmaf_cuda" \
    -f null -
```

As you see for 420 video format we will have to convert from NV12 to 420 as seen above using `scale_cuda`. For other formats like yuv444p or yuv422p you can directly use the input from the decoder without conversion.
`-filter_complex [0:v][1:v]libvmaf_cuda`
 
