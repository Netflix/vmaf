VMAF Usage through Docker
========================

Install Docker, then, from the VMAF directory, run:

```sh
docker build -t vmaf .
```

And to use it, just run:

```sh
docker run --rm vmaf [CLI]
```

Where `[CLI]` is one of `[run_vmaf, vmafossexec, vmaf_rc, run_vmaf_training, run_vmaf_in_batch, run_testing]`.

For example, if you are under root, to run `run_vmaf` on a sample reference/distorted video pair under `resource/yuv`:

```sh
docker run --rm -v $(pwd):/files vmaf \
  run_vmaf yuv420p 576 324 \
  /files/src01_hrc00_576x324.yuv \
  /files/src01_hrc01_576x324.yuv \
  --out-fmt json
```

Note that you need to first download the test videos from [vmaf_resource](https://github.com/Netflix/vmaf_resource/tree/master/python/test/resource).
```sh
wget https://github.com/Netflix/vmaf_resource/raw/master/python/test/resource/yuv/src01_hrc00_576x324.yuv
wget https://github.com/Netflix/vmaf_resource/raw/master/python/test/resource/yuv/src01_hrc01_576x324.yuv
```

To run `vmafossexec` with a specified model file:

```sh
docker run --rm -v $(pwd):/files vmaf \
  vmafossexec yuv420p 576 324 \
  /files/src01_hrc00_576x324.yuv \
  /files/src01_hrc01_576x324.yuv \
  /files/model/vmaf_v0.6.1.pkl \
  --log /dev/stdout
```
