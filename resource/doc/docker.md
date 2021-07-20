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
