VMAF Usage through Docker
========================

Install Docker, then, from the VMAF directory, run:

```
docker build -t vmaf .
```

And to use it, just run:

```
docker run --rm vmaf [CLI]
```

Where `[CLI]` is one of `[run_vmaf, run_vmaf_training, run_vmaf_in_batch, run_testing, vmafossexec]`.

For example, if you are under root, to run `run_vmaf` on a sample reference/distorted video pair under `resource/yuv`:

```
docker run --rm -v $(pwd):/files vmaf \
  run_vmaf yuv420p 576 324 \
  /files/python/test/resource/yuv/src01_hrc00_576x324.yuv \
  /files/python/test/resource/yuv/src01_hrc01_576x324.yuv \
  --out-fmt json
```

To run `vmafossexec` with a specified model file:

```
docker run --rm -v $(pwd):/files vmaf \
  vmafossexec yuv420p 576 324 \
  /files/python/test/resource/yuv/src01_hrc00_576x324.yuv \
  /files/python/test/resource/yuv/src01_hrc01_576x324.yuv \
  /files/model/nflxall_vmafv4.pkl
```
