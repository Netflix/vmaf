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

To run `vmafossexec` with a specified model file:

```shell script
docker run --rm -v $(pwd):/files vmaf \
    --entrypoint ""
    vmafossexec yuv420p 576 324 \
    /files/src01_hrc00_576x324.yuv \
    /files/src01_hrc01_576x324.yuv \
    /files/model/vmaf_float_v0.6.1.pkl \
    --log /dev/stdout
```

## Docker with CUDA support 

To run docker containers with GPU support you have to install the [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
After that a CUDA enabled container can be built using the below command line:
```shell script
docker build -f Dockerfile.cuda -t vmaf_cuda .
```

Besides VMAF the container also build ffmpeg support to enable GPU enabled decode to be able to run VMAF at speed of light.

```shell script
REF_VIDEO=$PWD/data/text_ref.mp4 
DIS_VIDEO=$PWD/data/text_dis.mp4 

docker run --gpus all  -e NVIDIA_DRIVER_CAPABILITIES=compute,video -v $REF_VIDEO:/data/ref.mp4 -v $DIS_VIDEO:/data/dis.mp4 vmaf_cuda
```
To run a custom ffmpeg command line inside the container use: 
```shell script
docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,video --entrypoint=bash -it --rm vmaf_cuda 
```